import json
from collections import OrderedDict
from pathlib import Path
from typing import TypedDict, Literal, Union
import numpy as np

from DataManager import DataStream, TimeType, OT, DataSet


class LifeTime(TypedDict, total=False):
    oid: int
    price: float
    volume: int
    rest: int
    birth: int
    death: int
    life: int
    AS: str


class SnapShot(TypedDict):
    timestamp: TimeType
    ask: OrderedDict[float, float]
    bid: OrderedDict[float, float]


class OrderBook:
    __slots__ = (
        "snapshots",
        "last_snapshot",
        "data_api",
        "oid_map",
        "data_cache",
        "skip_order",
    )

    # noinspection PyTypeChecker
    def __init__(self, data_api: Union[DataStream, DataSet]):
        self.snapshots: OrderedDict[TimeType, SnapShot] = OrderedDict()
        self.last_snapshot: SnapShot = None
        self.data_api: Union[DataStream, DataSet] = data_api
        self.oid_map: dict[int, LifeTime] = dict()
        self.data_cache: list[OT] = []
        self.skip_order: list[int] = []

    def next_batch(self, until: int = None) -> list[OT]:


        if self.data_cache:
            # 如果缓存有数据，先处理缓存的数据
            data = self.data_cache.pop(0)
        else:
            data = self.data_api.fresh()[0]
        timestamp_this = data["time"]

        if until is not None and timestamp_this > until:
            self.data_cache.append(data)
            return []

        res = [data]
        while True:
            try:
                next_data = self.data_api.fresh()
            except StopIteration:
                break
            timestamp_next = next_data[0]["time"]

            if timestamp_next != timestamp_this:
                self.data_cache.extend(next_data)
                break
            else:
                res.extend(next_data)

        return res

    def update(self, until: int = None):
        while True:
            try:
                res = self.next_batch(until=until)
                self.single_update(res)
            except StopIteration:
                break

    def single_update(self, datas: list[OT] = None):
        if datas is None:
            datas = self.next_batch()
        datas: list[OT] = sorted(datas, key=lambda x: x["oid"], reverse=True)
        for data in datas:

            if self.last_snapshot is None:
                self.last_snapshot = SnapShot(
                    timestamp=data[self.data_api.date_column],
                    ask=OrderedDict(),
                    bid=OrderedDict(),
                )

            if data["oid"] != 0 or data["ptype"] != 0:
                self.oid_map[data["oid"]] = LifeTime(
                    oid=data["oid"],
                    price=data["price"],
                    volume=data["volume"],
                    birth=data["time"],
                    rest=data["volume"],
                    AS="bid" if data["flag"] == 1 else "ask",
                )
                self._update_from_order(data)
            else:
                if self.data_api.ticker.endswith("SZ"):
                    if data["flag"] == 3:
                        # 异常订单：000001.SZ, 2746350，从未有人发起
                        self._log_oid(data, key="oidb")
                    elif data["flag"] == 4:
                        self._log_oid(data, key="oids")
                    else:
                        self._cal_log_oid(data, key="oidb")
                        self._cal_log_oid(data, key="oids")

                    tmp_oid_idx = "oidb" if data["flag"] in [1, 3] else "oids"
                    if data["price"] == 0.0:
                        if data[tmp_oid_idx] in self.oid_map:
                            data["price"] = self.oid_map[data[tmp_oid_idx]]["price"]
                        else:
                            continue
                    assert data["price"] != 0.0, data
                    self._update_from_tick(data)

                elif self.data_api.ticker.endswith("SH"):
                    # data.flag may be 0,1,2
                    if data["flag"] == 2:
                        self._cal_log_oid(data, key="oidb")
                    elif data["flag"] == 1:
                        self._cal_log_oid(data, key="oids")
                    else:
                        self._cal_log_oid(data, key="oidb")
                        self._cal_log_oid(data, key="oids")
                    tmp_oid_idx = "oidb" if data["flag"] in [1, 3] else "oids"
                    if data["price"] == 0.0:
                        if data[tmp_oid_idx] in self.oid_map:
                            data["price"] = self.oid_map[data[tmp_oid_idx]]["price"]
                        else:
                            continue
                    self._update_from_tick(data)

    def _cal_log_oid(self, data: OT, key: str):
        try:
            tmp_val = self.oid_map[data[key]]["volume"] - data["volume"]
            if tmp_val != 0:
                self.oid_map[data[key]]["rest"] = tmp_val
            elif tmp_val == 0:
                self.oid_map[data[key]]["rest"] = 0
                self.oid_map[data[key]]["death"] = data["time"]
                self.oid_map[data[key]]["life"] = (
                        data["time"] - self.oid_map[data[key]]["birth"]
                )
        except KeyError:
            pass

    def _log_oid(self, data: OT, key: str):
        try:
            self.oid_map[data[key]]["rest"] = 0
            self.oid_map[data[key]]["death"] = data["time"]
            self.oid_map[data[key]]["life"] = (
                    data["time"] - self.oid_map[data[key]]["birth"]
            )
        except KeyError:
            return

    @staticmethod
    def _order_change(
            snap: SnapShot, AS: Literal["ask", "bid"], direction: Literal[1, -1], data: OT
    ):
        snap[AS][data["price"]] = (
                snap[AS].get(data["price"], 0) + direction * data["volume"]
        )
        if snap[AS][data["price"]] == 0:
            del snap[AS][data["price"]]

        snap["ask"] = OrderedDict(
            sorted(snap["ask"].items(), key=lambda x: x[0], reverse=False)
        )

        snap["bid"] = OrderedDict(
            sorted(snap["bid"].items(), key=lambda x: x[0], reverse=True)
        )

    @staticmethod
    def _tick_change(snap: SnapShot, data: OT, direction: Union[str, list[str]] = None):
        if direction is None:
            direction = ["ask", "bid"]
        assert data["price"] != 0.0, data
        TradePrice = data["price"]
        TradeQty = data["volume"]
        A_p_v = list(snap["ask"].items())
        B_p_v = list(snap["bid"].items())

        if "bid" in direction:
            rest = TradeQty
            for p, v in B_p_v:
                if p >= TradePrice:
                    if v >= rest:
                        v -= rest
                        if v == 0:
                            del snap["bid"][p]
                        else:
                            snap["bid"][p] = v
                    else:
                        rest -= v
                        del snap["bid"][p]
                else:
                    break

        if "ask" in direction:
            rest = TradeQty
            for p, v in A_p_v:
                if p <= TradePrice:
                    if v >= rest:
                        v -= rest
                        if v == 0:
                            del snap["ask"][p]
                        else:
                            snap["ask"][p] = v
                    else:
                        rest -= v
                        del snap["ask"][p]
                else:
                    break

        snap["ask"] = OrderedDict(
            sorted(snap["ask"].items(), key=lambda x: x[0], reverse=False)
        )

        snap["bid"] = OrderedDict(
            sorted(snap["bid"].items(), key=lambda x: x[0], reverse=True)
        )

    def _update_from_order(self, data: OT):
        snap: SnapShot = self.last_snapshot.copy()
        snap["timestamp"]: int = data[self.data_api.date_column]
        AS: Literal["bid", "ask"] = "bid" if data["flag"] in [1, 3] else "ask"  # type: ignore
        direction: Literal[1, -1] = 1 if data["flag"] in [1, 2] else -1  # type: ignore
        # assert data["price"] != 0.0, data
        if data["price"] == 0.0:
            self.skip_order.append(data["oid"])
            return
        if self.data_api.ticker.endswith("SZ") or self.data_api.ticker.endswith("SH"):
            self._order_change(snap, AS, direction, data)
        else:
            raise NotImplementedError

        self.last_snapshot = snap
        self.snapshots[data[self.data_api.date_column]] = self.last_snapshot.copy()  # type: ignore

    def _update_from_tick(self, data: OT):
        snap: SnapShot = self.last_snapshot.copy()
        snap["timestamp"]: int = data[self.data_api.date_column]
        # assert data["price"] != 0.0, data
        if self.data_api.ticker.endswith("SZ"):
            if data["flag"] in [1, 2]:
                direction = []
                if data["oids"] not in self.skip_order:
                    direction.append("ask")
                if data["oidb"] not in self.skip_order:
                    direction.append("bid")
                self._tick_change(snap, data, direction=direction)
            elif data["flag"] == 3:
                self._order_change(snap, "bid", -1, data)
            elif data["flag"] == 4:
                self._order_change(snap, "ask", -1, data)
        elif self.data_api.ticker.endswith("SH"):
            if data["flag"] == 2:
                self._tick_change(snap, data, direction=["bid"])
            elif data["flag"] == 1:
                self._tick_change(snap, data, direction=["ask"])
            else:
                self._tick_change(snap, data)
        else:
            raise NotImplementedError

    def search_closet_time(self, query_stamp: int):
        logged_timestamp: np.ndarray = np.array(list(self.snapshots.keys()))
        search_timestamp = logged_timestamp[logged_timestamp <= query_stamp]
        time_difference = np.abs(search_timestamp - query_stamp)
        closest_index = np.argmin(time_difference)
        closest_time = search_timestamp[closest_index]
        return closest_time

    @staticmethod
    def print_json(dict_like: dict):
        print(json.dumps(dict_like, indent=2))

    def search_snapshot(self, query_stamp: int):
        closest_time = self.search_closet_time(query_stamp)
        return self.snapshots[closest_time]


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    data_api = Path(__file__).parent / "../../datas/000001.SZ/tick/gtja/2023-03-01.csv"
    tick = DataSet(data_api, date_column="time", ticker="000001.SZ")

    ob = OrderBook(data_api=tick)
    ob.single_update()
    print(ob.last_snapshot)
    ob.single_update()
    print(ob.last_snapshot)
    ob.single_update()
    print(ob.last_snapshot)

    # datas = tick.fresh(10)
    # ob.single_update(datas)
    # print(ob.last_snapshot)
    # datas = tick.fresh(1)
    # ob.single_update(datas)
    # print(ob.last_snapshot)

    # timestamp = 20230508093103000
    # ob.update(until=None)
    # near = ob.search_snapshot(timestamp)
    # print(near["timestamp"])
    # print(near["bid"])
    # print(near["ask"])
