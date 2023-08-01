import json
from collections import OrderedDict
from pathlib import Path
from typing import TypedDict, Literal
import numpy as np
from DataManager import DataStream, TimeType, OT


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
    __slots__ = "snapshots", "last_snapshot", "data_api", "oid_map", "data_cache"

    # noinspection PyTypeChecker
    def __init__(self, data_api: DataStream):
        self.snapshots: OrderedDict[TimeType, SnapShot] = OrderedDict()
        self.last_snapshot: SnapShot = None
        self.data_api: DataStream = data_api
        self.oid_map: dict[int, LifeTime] = dict()
        self.data_cache: list[OT] = []

    def update(self, until: int = None):
        while True:
            if self.data_cache:
                # 如果缓存有数据，先处理缓存的数据
                data = self.data_cache.pop(0)
            else:
                data = self.data_api.fresh()
            timestamp_this = data["time"]
            if until is not None and timestamp_this > until:
                self.data_cache.append(data)
                break

            if self.last_snapshot is None:
                self.last_snapshot = SnapShot(
                    timestamp=data[self.data_api.date_column],
                    ask=OrderedDict(),
                    bid=OrderedDict(),
                )
            res = [data]
            while True:
                try:
                    next_data = self.data_api.fresh()
                except StopIteration:
                    break
                timestamp_next = next_data["time"]
                if timestamp_next != timestamp_this:
                    self.data_cache.append(next_data)
                    break
                else:
                    res.append(next_data)

            self.single_update(res)

    def single_update(self, datas: list[OT]):
        datas: list[OT] = sorted(datas, key=lambda x: x["oid"], reverse=True)
        for data in datas:
            if data["oid"] != 0 or data["ptype"] != 0:
                self.oid_map[data["oid"]] = LifeTime(
                    oid=data["oid"],
                    price=data["price"],
                    volume=data["volume"],
                    birth=data["time"],
                    rest=data["volume"],
                    AS= "bid" if data["flag"] == 1 else "ask"
                )
                self._update_from_order(data)
            else:
                if self.data_api.ticker.endswith(".SZ"):
                    if data["flag"] == 3:
                        assert data["oidb"] in self.oid_map, data
                        self.oid_map[data["oidb"]]["rest"] = 0
                        self.oid_map[data["oidb"]]["death"] = data["time"]
                        self.oid_map[data["oidb"]]["life"] = (
                            self.oid_map[data["oidb"]]["birth"] - data["time"]
                        )
                    elif data["flag"] == 4:
                        assert data["oids"] in self.oid_map, data
                        self.oid_map[data["oids"]]["rest"] = 0
                        self.oid_map[data["oids"]]["death"] = data["time"]
                        self.oid_map[data["oids"]]["life"] = (
                            self.oid_map[data["oids"]]["birth"] - data["time"]
                        )
                    else:
                        assert data["oidb"] in self.oid_map, data
                        tmp_val = self.oid_map[data["oidb"]]["volume"] - data["volume"]
                        if tmp_val != 0:
                            self.oid_map[data["oidb"]]["rest"] = tmp_val
                        elif tmp_val == 0:
                            self.oid_map[data["oidb"]]["rest"] = 0
                            self.oid_map[data["oidb"]]["death"] = data["time"]
                            self.oid_map[data["oidb"]]["life"] = (
                                    self.oid_map[data["oidb"]]["birth"] - data["time"]
                            )

                        assert data["oids"] in self.oid_map, data
                        tmp_val = self.oid_map[data["oids"]]["volume"] - data["volume"]
                        if tmp_val != 0:
                            self.oid_map[data["oids"]]["rest"] = tmp_val
                        elif tmp_val == 0:
                            self.oid_map[data["oids"]]["rest"] = 0
                            self.oid_map[data["oids"]]["death"] = data["time"]
                            self.oid_map[data["oids"]]["life"] = (
                                    self.oid_map[data["oids"]]["birth"] - data["time"]
                            )

                    tmp_oid_idx = "oidb" if data["flag"] in [1, 3] else "oids"
                    if data["price"] == 0.0:
                        data["price"] = self.oid_map[data[tmp_oid_idx]]["price"]
                    self.oid_map[tmp_oid_idx] = LifeTime()
                    self._update_from_tick(data)

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

    def _tick_change(self, snap: SnapShot, data: OT):
        assert data["price"] != 0.0, data
        TradePrice = data["price"]
        TradeQty = data["volume"]
        A_p_v = list(snap["ask"].items())
        B_p_v = list(snap["bid"].items())

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
        snap["timestamp"]: int = data[self.data_api.date_column]  # type: ignore
        if self.data_api.ticker.endswith("SZ"):
            AS: Literal["bid", "ask"] = "bid" if data["flag"] == 1 else "ask"  # type: ignore
            direction: Literal[1, -1] = 1 if data["flag"] in [1, 2] else -1  # type: ignore
            # TODO:
            if data["price"] == 0.0:
                pass
            else:
                self._order_change(snap, AS, direction, data)
        elif self.data_api.ticker.endswith("SH"):
            pass
        else:
            raise NotImplementedError

        self.last_snapshot = snap
        self.snapshots[data[self.data_api.date_column]] = self.last_snapshot.copy()  # type: ignore

    def _update_from_tick(self, data: OT):
        snap: SnapShot = self.last_snapshot.copy()
        snap["timestamp"]: int = data[self.data_api.date_column]
        assert data["price"] != 0.0, data
        if self.data_api.ticker.endswith("SZ"):
            if data["flag"] in [1, 2]:
                self._tick_change(snap, data)
            elif data["flag"] == 3:
                self._order_change(snap, "bid", -1, data)
            elif data["flag"] == 4:
                self._order_change(snap, "ask", -1, data)
        elif self.data_api.ticker.endswith("SH"):
            pass
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
    data_dir = Path(__file__).parent / "../../datas/000001.SZ/tick/gta"
    tick = DataStream(data_dir, date_column="time")
    ob = OrderBook(data_api=tick)
    timestamp = 20230508093103200
    ob.update(until=timestamp)
    near = ob.search_snapshot(timestamp)
    print(near["timestamp"])
    print(near["bid"])
    print(near["ask"])
    print(ob.oid_map[1712])
