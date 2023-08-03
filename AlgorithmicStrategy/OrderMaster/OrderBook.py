import json
from collections import OrderedDict
from typing import Literal, Union

import numpy as np

from .DataManager import DataStream, DataSet
from .Schema import TimeType, SnapShot, LifeTime, OrderTick, OrderFlag


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
        self.data_cache: list[OrderTick] = []
        self.skip_order: list[int] = []

    def next_batch(self, until: int = None) -> list[OrderTick]:
        if self.data_cache:
            # 如果缓存有数据，先处理缓存的数据
            data = self.data_cache.pop(0)
        else:
            data = self.data_api.fresh()[0]
        timestamp_this = data[self.data_api.date_column]

        if until is not None and timestamp_this > until:
            self.data_cache.append(data)
            return []

        res = [data]
        while True:
            try:
                next_data = self.data_api.fresh()
            except StopIteration:
                break
            timestamp_next = next_data[0][self.data_api.date_column]

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
                if res:
                    self.single_update(res)
                else:
                    break
            except StopIteration:
                break

    def single_update(self, datas: list[OrderTick] = None):
        if datas is None:
            datas = self.next_batch()
        datas: list[OrderTick] = sorted(datas, key=lambda x: x["oid"], reverse=True)
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
                    birth=data[self.data_api.date_column],
                    rest=data["volume"],
                    AS="bid" if data["flag"] == OrderFlag.SELL else "ask",
                )
                self._update_from_order(data)
            else:
                if self.data_api.ticker.endswith("SZ"):
                    if data["flag"] == OrderFlag.CANCEL_SELL:
                        # 异常订单：000001.SZ, 2746350，从未有人发起
                        self._log_oid(data, key="oidb")
                    elif data["flag"] == OrderFlag.CANCEL_BUY:
                        self._log_oid(data, key="oids")
                    else:
                        self._cal_log_oid(data, key="oidb")
                        self._cal_log_oid(data, key="oids")

                    tmp_oid_idx = "oidb" if data["flag"] in [OrderFlag.CANCEL_BUY, OrderFlag.BUY] else "oids"
                    if data["price"] == 0.0:
                        if data[tmp_oid_idx] in self.oid_map:
                            data["price"] = self.oid_map[data[tmp_oid_idx]]["price"]
                        else:
                            continue
                    assert data["price"] != 0.0, data
                    self._update_from_tick(data)

                elif self.data_api.ticker.endswith("SH"):
                    # data.flag may be 0,1,2
                    if data["flag"] == OrderFlag.SELL:
                        self._cal_log_oid(data, key="oidb")
                    elif data["flag"] == OrderFlag.BUY:
                        self._cal_log_oid(data, key="oids")
                    elif data["flag"] == OrderFlag.NEUTRAL:
                        self._cal_log_oid(data, key="oidb")
                        self._cal_log_oid(data, key="oids")
                    tmp_oid_idx = "oidb" if data["flag"] == OrderFlag.SELL else "oids"
                    if data["price"] == 0.0:
                        if data[tmp_oid_idx] in self.oid_map:
                            data["price"] = self.oid_map[data[tmp_oid_idx]]["price"]
                        else:
                            continue
                    self._update_from_tick(data)
    def _cal_log_oid(self, data: OrderTick, key: str):
        try:
            tmp_val = self.oid_map[data[key]]["volume"] - data["volume"]
            if tmp_val != 0:
                self.oid_map[data[key]]["rest"] = tmp_val
            elif tmp_val == 0:
                self.oid_map[data[key]]["rest"] = 0
                self.oid_map[data[key]]["death"] = data[self.data_api.date_column]
                self.oid_map[data[key]]["life"] = (
                    data[self.data_api.date_column] - self.oid_map[data[key]]["birth"]
                )
        except KeyError:
            pass

    def _log_oid(self, data: OrderTick, key: str):
        try:
            self.oid_map[data[key]]["rest"] = 0
            self.oid_map[data[key]]["death"] = data[self.data_api.date_column]
            self.oid_map[data[key]]["life"] = (
                data[self.data_api.date_column] - self.oid_map[data[key]]["birth"]
            )
        except KeyError:
            return

    @staticmethod
    def _order_change(
        snap: SnapShot,
        AS: Literal["ask", "bid"],
        direction: Literal[1, -1],
        data: OrderTick,
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
    def _tick_change(
        snap: SnapShot, data: OrderTick, direction: Union[str, list[str]] = None
    ):
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

    def _update_from_order(self, data: OrderTick):
        snap: SnapShot = self.last_snapshot.copy()
        snap["timestamp"]: int = data[self.data_api.date_column]
        AS: Literal["bid", "ask"] = "bid" if data["flag"] in [OrderFlag.BUY, OrderFlag.CANCEL_BUY] else "ask"  # type: ignore
        direction: Literal[1, -1] = 1 if data["flag"] in [OrderFlag.BUY, OrderFlag.SELL] else -1  # type: ignore
        # assert data["price"] != 0.0, data
        if data["price"] == 0.0:
            self.skip_order.append(data["oid"])
            return
        elif data["price"] == 1.0:
            self.skip_order.append(data["oid"])
            return
        if self.data_api.ticker.endswith("SZ") or self.data_api.ticker.endswith("SH"):
            self._order_change(snap, AS, direction, data)
        else:
            raise NotImplementedError

        self.last_snapshot = snap
        self.snapshots[data[self.data_api.date_column]] = self.last_snapshot.copy()  # type: ignore

    def _update_from_tick(self, data: OrderTick):
        snap: SnapShot = self.last_snapshot.copy()
        snap["timestamp"]: int = data[self.data_api.date_column]
        # assert data["price"] != 0.0, data
        if self.data_api.ticker.endswith("SZ"):
            if data["flag"] in [OrderFlag.SELL, OrderFlag.BUY]:
                direction = ["ask", "bid"]
                if data["oids"] in self.skip_order:
                    direction.remove("ask")
                    self.oid_map[data["oids"]]["volume"] = data["volume"]
                if data["oidb"] in self.skip_order:
                    direction.remove("bid")
                    self.oid_map[data["oidb"]]["volume"] = data["volume"]
                self._tick_change(snap, data, direction=direction)
            elif data["flag"] == OrderFlag.CANCEL_BUY:
                self._order_change(snap, "bid", -1, data)
            elif data["flag"] == OrderFlag.CANCEL_SELL:
                self._order_change(snap, "ask", -1, data)
        elif self.data_api.ticker.endswith("SH"):
            if data["flag"] == OrderFlag.SELL:
                self._tick_change(snap, data, direction=["bid"])
            elif data["flag"] == OrderFlag.BUY:
                self._tick_change(snap, data, direction=["ask"])
            else:
                self._tick_change(snap, data)
        else:
            raise NotImplementedError

    def search_closet_time(self, query_stamp: int):
        logged_timestamp: np.ndarray = np.array(list(self.snapshots.keys()))
        search_timestamp = logged_timestamp[logged_timestamp <= query_stamp]
        return search_timestamp[-1]

    @staticmethod
    def print_json(dict_like: dict):
        print(json.dumps(dict_like, indent=2))

    def search_snapshot(self, query_stamp: int):
        closest_time = self.search_closet_time(query_stamp)
        return self.snapshots[closest_time]
