import copy
import json
from collections import OrderedDict
from typing import Literal, Union
from datetime import datetime, timedelta

import numpy as np

from DataManager import DataStream, DataSet
from Schema import TimeType, SnapShot, LifeTime, OrderTick, OrderFlag, Excecuted_trade
from Schema import OrderDepth, Excecuted_trade
from Depth import OrderDepthCalculator


class OrderBook:
    __slots__ = (
        "snapshots",
        "last_snapshot",
        "data_api",
        "candle_tick",
        "oid_map",
        "skip_order",
        "kwargs",
    )

    # noinspection PyTypeChecker
    def __init__(self, data_api: Union[DataStream, DataSet], **kwargs):
        self.snapshots: OrderedDict[TimeType, SnapShot] = OrderedDict()
        self.last_snapshot: SnapShot = None
        self.data_api: Union[DataStream, DataSet] = data_api
        self.oid_map: dict[int, LifeTime] = dict()
        self.skip_order: list[int] = []
        self.candle_tick: OrderedDict[TimeType, list[float]] = OrderedDict()
        self.kwargs = kwargs

    def update(self, until: int = None):
        while True:
            try:
                res = self.data_api.next_batch(until=until)
                if res:
                    self.single_update(res)
                else:
                    break
            except StopIteration:
                break

    def single_update(self, datas: list[OrderTick] = None):
        if datas is None:
            datas = self.data_api.next_batch()
        datas: list[OrderTick] = sorted(datas, key=lambda x: x["oid"], reverse=True)
        for data in datas:
            if self.last_snapshot is None:
                self.last_snapshot = SnapShot(
                    timestamp=data[self.data_api.date_column],
                    ask=OrderedDict(),
                    bid=OrderedDict(),
                    ask_num=OrderedDict(),  # 初始化卖单数
                    bid_num=OrderedDict(),  # 初始化买单数
                    ask_order_stale=OrderedDict(),
                    ask_num_death=OrderedDict(),
                    bid_order_stale=OrderedDict(),
                    bid_num_death=OrderedDict(),
                    total_trade=Excecuted_trade(
                        order_num=0,
                        volume=0,
                        price=0.0,
                        total_price=0.0,
                        passive_num=0,
                        passive_stale_total=0,
                    ),  # 初始化总成交单数
                    order_depth=OrderDepth(
                        n_depth=list([]), total_volume=0, weighted_average_depth=0.0
                    ),
                )

            if data["oid"] != 0 or data["ptype"] != 0:

                if data["flag"] in [OrderFlag.BUY, OrderFlag.SELL]:
                    self.oid_map[data["oid"]] = LifeTime(
                        oid=data["oid"],
                        price=data["price"],
                        volume=data["volume"],
                        birth=data[self.data_api.date_column],
                        rest=data["volume"],
                        AS="bid" if data["flag"] == OrderFlag.BUY else "ask",
                        iscall=True if data["iscall"] == True else False
                    )
                self._update_from_order(data)
                if data["flag"] in [OrderFlag.BUY, OrderFlag.SELL]:
                    self._order_num_change(data, None, 1)
                elif data["flag"] in [
                    OrderFlag.CANCEL_BUY,
                    OrderFlag.CANCEL_SELL,
                ]:  # 如果是沪市的撤买单/撤卖单
                    self._log_oid(data, key="oid")
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
                        try:
                            self._trade_update(data)  # 对成交单数量进行更新
                        except KeyError as e:
                            print(data)
                            raise e

                    tmp_oid_idx = (
                        "oidb"
                        if data["flag"] in [OrderFlag.CANCEL_BUY, OrderFlag.BUY]
                        else "oids"
                    )
                    if data["price"] == 0.0:
                        if data[tmp_oid_idx] in self.oid_map:
                            data["price"] = self.oid_map[data[tmp_oid_idx]]["price"]
                        else:
                            continue
                    # assert data["price"] != 0.0, data
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
                    self._trade_update(data)  # 对成交单数量进行更新
                    tmp_oid_idx = "oidb" if data["flag"] == OrderFlag.SELL else "oids"
                    if data["price"] == 0.0:
                        if data[tmp_oid_idx] in self.oid_map:
                            data["price"] = self.oid_map[data[tmp_oid_idx]]["price"]
                        else:
                            continue
                    self._update_from_tick(data)

    def _order_depth_update(self, **kwargs):
        order_depth_calculator = OrderDepthCalculator(
            self.last_snapshot, kwargs.get("n", 10)
        )
        # self.last_snapshot["order_depth"]["n_depth"] = order_depth_calculator.n_depth
        # self.last_snapshot["order_depth"]["total_volume"] = order_depth_calculator.total_volume
        self.last_snapshot["order_depth"][
            "weighted_average_depth"
        ] = order_depth_calculator.calculate_weighted_average_depth(
            kwargs.get("decay_rate", 0.5)
        )

    def _order_num_change(self, data: OrderTick, key: str, direction: Literal[1, -1]):
        """
        使ordernum增加的direction为1,使ordernum减少的direction为-1
        """
        if direction == 1:
            if data["flag"] == OrderFlag.BUY:
                if data["price"] not in self.last_snapshot["bid_num"]:
                    self.last_snapshot["bid_num"][data["price"]] = 1
                else:
                    self.last_snapshot["bid_num"][data["price"]] += 1
            if data["flag"] == OrderFlag.SELL:
                if data["price"] not in self.last_snapshot["ask_num"]:
                    self.last_snapshot["ask_num"][data["price"]] = 1
                else:
                    self.last_snapshot["ask_num"][data["price"]] += 1

        if direction == -1:
            price = self.oid_map[data[key]]["price"]
            if key == "oidb":
                self.last_snapshot["bid_num"][price] -= 1
            if key == "oids":
                self.last_snapshot["ask_num"][price] -= 1
        return

    def _trade_update(self, data: OrderTick):
        # 计算所有成交单的数据
        # 如果主动成交单的rest = volume则为第一次计算，仅计算一次成交单数量的增加，volume仍然继续计算
        timestamp = data["time"]

        if not self.candle_tick:
            self.candle_tick[timestamp] = [
                None,
                data["price"],
                data["price"],
                data["price"],
                data["price"],
            ]

        else:
            timestamp_last = max(self.candle_tick.keys())
            if timestamp not in self.candle_tick:
                if len(self.snapshots.keys()) == 1:
                    self.candle_tick[timestamp] = [
                        None,
                        data["price"],
                        data["price"],
                        data["price"],
                        data["price"],
                    ]
                else:
                    self.candle_tick[timestamp] = [
                        self.candle_tick[timestamp_last][4],
                        data["price"],
                        data["price"],
                        data["price"],
                        data["price"],
                    ]

            else:
                candle = self.candle_tick[timestamp]
                if data["price"] > candle[2]:
                    candle[2] = data["price"]
                if data["price"] < candle[3]:
                    candle[3] = data["price"]
                candle[4] = data["price"]

        try:
            oidp = data["oidb"] if data["flag"] == OrderFlag.BUY else data["oids"]
            if self.oid_map[oidp]["rest"] == 0:
                self.last_snapshot["total_trade"]["order_num"] += 1
            self.last_snapshot["total_trade"]["volume"] += data["volume"]
            total_price = data["price"] * data["volume"]
            self.last_snapshot["total_trade"]["total_price"] += total_price
            # 再计算被动成交单，如果全部成交了就在passive部分增加
            oid = data["oids"] if data["flag"] == OrderFlag.BUY else data["oidb"]
            if self.oid_map[oid]["rest"] == 0:
                self.last_snapshot["total_trade"]["order_num"] += 1
                self.last_snapshot["total_trade"]["passive_num"] += 1
                self.last_snapshot["total_trade"][
                    "passive_stale_total"
                ] += self.oid_map[oid]["life"]
                self.last_snapshot["total_trade"]["volume"] += self.oid_map[oid]["rest"]
                total_price = data["price"] * self.oid_map[oid]["rest"]
                self.last_snapshot["total_trade"]["total_price"] += total_price

            self.last_snapshot["total_trade"]["price"] = (
                self.last_snapshot["total_trade"]["total_price"]
                / self.last_snapshot["total_trade"]["volume"]
            )
        except KeyError:
            return

    def _order_stale_update(self, data: OrderTick, key: str):
        oid = data[key]
        life = self.oid_map[oid]["life"]
        price = self.oid_map[oid]["price"]
        AS = self.oid_map[oid]["AS"]
        if AS == "bid":
            if price not in self.last_snapshot["bid_order_stale"]:
                self.last_snapshot["bid_order_stale"][price] = life
                self.last_snapshot["bid_num_death"][price] = 1
            else:
                self.last_snapshot["bid_order_stale"][price] += life
                self.last_snapshot["bid_num_death"][price] += 1
        if AS == "ask":
            if price not in self.last_snapshot["ask_order_stale"]:
                self.last_snapshot["ask_order_stale"][price] = life
                self.last_snapshot["ask_num_death"][price] = 1
            else:
                self.last_snapshot["ask_order_stale"][price] += life
                self.last_snapshot["ask_num_death"][price] += 1

    def _cal_log_oid(self, data: OrderTick, key: str):
        try:
            tmp_val = self.oid_map[data[key]]["rest"] - data["volume"]
            if tmp_val != 0:
                self.oid_map[data[key]]["rest"] = tmp_val
            elif tmp_val == 0:
                self.oid_map[data[key]]["rest"] = 0
                self.oid_map[data[key]]["death"] = data[self.data_api.date_column]
                t1 = datetime.strptime(str(data[self.data_api.date_column]), '%Y%m%d%H%M%S%f')
                t2 = datetime.strptime(str(self.oid_map[data[key]]["birth"]), '%Y%m%d%H%M%S%f')
                life = (t1-t2)/timedelta(milliseconds=1)
                if self.oid_map[data[key]]["iscall"] == True and data["iscall"] == False:
                    self.oid_map[data[key]]["life"] = life - 3*1e5
                else:
                    self.oid_map[data[key]]["life"] = life
                self._order_num_change(data, key, -1)
                self._order_stale_update(data, key)
        except KeyError:
            pass

    def _log_oid(self, data: OrderTick, key: str):
        try:
            self.oid_map[data[key]]["rest"] = 0
            self.oid_map[data[key]]["death"] = data[self.data_api.date_column]
            t1 = datetime.strptime(str(data[self.data_api.date_column]), '%Y%m%d%H%M%S%f')
            t2 = datetime.strptime(str(self.oid_map[data[key]]["birth"]), '%Y%m%d%H%M%S%f')
            life = (t1-t2)/timedelta(milliseconds=1)
            if self.oid_map[data[key]]["iscall"] == True and data["iscall"] == False:
                self.oid_map[data[key]]["life"] = life - 3*1e5
            else:
                self.oid_map[data[key]]["life"] = life
            self._order_num_change(data, key, -1)
            self._order_stale_update(data, key)
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
        self._order_depth_update(**self.kwargs)
        snap: SnapShot = self.last_snapshot.copy()
        snap["timestamp"]: int = data[self.data_api.date_column]
        AS: Literal["bid", "ask"] = (
            "bid" if data["flag"] in [OrderFlag.BUY, OrderFlag.CANCEL_BUY] else "ask"
        )  # type: ignore
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
        self.snapshots[data[self.data_api.date_column]] = copy.deepcopy(self.last_snapshot)  # type: ignore

    def _update_from_tick(self, data: OrderTick):
        self._order_depth_update(**self.kwargs)
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
                if data["oidb"] in self.skip_order:
                    return
                self._order_change(snap, "bid", -1, data)
            elif data["flag"] == OrderFlag.CANCEL_SELL:
                if data["oids"] in self.skip_order:
                    return
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
        self.snapshots[data[self.data_api.date_column]] = copy.deepcopy(
            self.last_snapshot
        )

    # 计算两个时间戳之间的成交订单平均新陈代谢
    def _get_avg_stale(self, timestamp_1: TimeType, timestamp_2: TimeType):
        snap2 = self.search_snapshot(timestamp_2)
        snap1 = self.search_snapshot(timestamp_1)
        ask_total_stale2 = copy.deepcopy(snap2["ask_order_stale"])
        bid_total_stale2 = copy.deepcopy(snap2["bid_order_stale"])
        for key in list(ask_total_stale2.keys()):
            if key in snap1["ask_order_stale"].keys():
                ask_total_stale2[key] -= snap1["ask_order_stale"][key]
                if ask_total_stale2[key] == 0:
                    del ask_total_stale2[key]
                    continue
                ask_death_change = (
                    snap2["ask_num_death"][key] - snap1["ask_num_death"][key]
                )
                ask_total_stale2[key] = ask_total_stale2[key] / ask_death_change
            else:
                ask_total_stale2[key] = (
                    ask_total_stale2[key] / snap2["ask_num_death"][key]
                )

        for key in list(bid_total_stale2.keys()):
            if key in snap1["bid_order_stale"].keys():
                bid_total_stale2[key] -= snap1["bid_order_stale"][key]
                if bid_total_stale2[key] == 0:
                    del bid_total_stale2[key]
                    continue
                bid_death_change = (
                    snap2["bid_num_death"][key] - snap1["bid_num_death"][key]
                )
                bid_total_stale2[key] = bid_total_stale2[key] / bid_death_change
            else:
                bid_total_stale2[key] = (
                    bid_total_stale2[key] / snap2["bid_num_death"][key]
                )

        return ask_total_stale2, bid_total_stale2

    def get_avg_trade(self, timestamp_1, timestamp_2):
        snap1 = self.search_snapshot(timestamp_1)
        snap2 = self.search_snapshot(timestamp_2)

        total_trade_2 = copy.deepcopy(snap2["total_trade"])
        total_trade_2["order_num"] -= snap1["total_trade"]["order_num"]
        total_trade_2["passive_num"] -= snap1["total_trade"]["passive_num"]
        total_trade_2["total_price"] -= snap1["total_trade"]["total_price"]
        total_trade_2["volume"] -= snap1["total_trade"]["volume"]
        total_trade_2["passive_stale_total"] -= snap1["total_trade"][
            "passive_stale_total"
        ]

        if total_trade_2["passive_num"] == 0:
            passive_stale_avg = -1
            total_trade_2["passive_stale_total"] = -1
            if total_trade_2["volume"] == 0:
                total_trade_2["price"] = -1
        else:
            total_trade_2["price"] = (
                total_trade_2["total_price"] / total_trade_2["volume"]
            )
            passive_stale_avg = (
                total_trade_2["passive_stale_total"] / total_trade_2["passive_num"]
            )

        return total_trade_2, passive_stale_avg

    @staticmethod
    def print_json(dict_like: dict):
        print(json.dumps(dict_like, indent=2))

    def search_closet_time(
        self, query_stamp: int, stamps: list[int] = None, backwards: bool = True
    ):
        if stamps is None:
            stamps = list(self.snapshots.keys())
        logged_timestamp: np.ndarray = np.array(stamps)
        if backwards:
            search_timestamp = logged_timestamp[logged_timestamp <= query_stamp]
            return search_timestamp[-1]
        else:
            search_timestamp = logged_timestamp[logged_timestamp >= query_stamp]
            return search_timestamp[0]

    def search_snapshot(self, query_stamp: int):
        closest_time = self.search_closet_time(query_stamp)
        return self.snapshots[closest_time]

    def search_candle(self, query_stamp: int):
        closest_time = self.search_closet_time(
            query_stamp, list(self.candle_tick.keys())
        )
        return self.candle_tick[closest_time]

    def get_candle_slot(self, timestamp1: int, timestamp2: int):
        logged_timestamp: np.ndarray = np.array(list(self.candle_tick.keys()))
        search_timestamp = logged_timestamp[(timestamp1 <= logged_timestamp) & (logged_timestamp <= timestamp2)]
        if search_timestamp.any():
            candle = [0.0, 0.0, 0.0, 0.0, 0.0]
            candle[4] = self.candle_tick[search_timestamp[-1]][4]
            candle[0] = self.candle_tick[search_timestamp[0]][4]
            candle[1] = self.candle_tick[search_timestamp[0]][1]
            candle[2] = self.candle_tick[search_timestamp[0]][2]
            candle[3] = self.candle_tick[search_timestamp[0]][3]
            for timestamp in search_timestamp:
                if candle[2] < self.candle_tick[timestamp][2]:
                    candle[2] = self.candle_tick[timestamp][2]
                if candle[3] > self.candle_tick[timestamp][3]:
                    candle[3] = self.candle_tick[timestamp][3]
        else:
            closest_time1 = self.search_closet_time(
                timestamp1, list(self.candle_tick.keys()))
            closest_time2 = self.search_closet_time(
                timestamp2, list(self.candle_tick.keys()), -1
            )
            candle = self.get_candle_slot(closest_time1, closest_time2)
        return candle

    @staticmethod
    def get_max_n_keys_and_values(ordered_dict: OrderedDict, n: int):
        keys = list(ordered_dict.keys())
        values = list(ordered_dict.values())

        if n >= len(keys):
            missing_count = n - len(keys)
            keys += [None] * missing_count
            values += [None] * missing_count
        else:
            keys = keys[:n]
            values = values[:n]

        return keys, values

    @staticmethod
    def get_values_from_ordered_dict(
        input_list: list, ordered_dict: OrderedDict, default: int = 0
    ):
        result = []
        for key in input_list:
            result.append(ordered_dict.get(key, default))

        return result

    @staticmethod
    def divide_ordered_dicts(od1: OrderedDict, od2: OrderedDict):
        result = OrderedDict()
        for key in od1:
            if key in od2:
                result[key] = od1[key] / od2[key]
            else:
                result[key] = 0

        return result

    def get_super_snapshot(self, level: int, timestamp: int):
        ob_snap = self.search_snapshot(timestamp)
        super_snapshot = dict()
        (
            super_snapshot["ask_price"],
            super_snapshot["ask_volume"],
        ) = self.get_max_n_keys_and_values(ob_snap["ask"], level)
        (
            super_snapshot["bid_price"],
            super_snapshot["bid_volume"],
        ) = self.get_max_n_keys_and_values(ob_snap["bid"], level)
        _, super_snapshot["ask_order_num"] = self.get_max_n_keys_and_values(
            ob_snap["ask_num"], level
        )
        _, super_snapshot["bid_order_num"] = self.get_max_n_keys_and_values(
            ob_snap["bid_num"], level
        )
        super_snapshot["ask_order_stale"] = self.get_values_from_ordered_dict(
            super_snapshot["ask_price"],
            self.divide_ordered_dicts(
                ob_snap["ask_order_stale"], ob_snap["ask_num_death"]
            ),
            default=-1,
        )
        super_snapshot["bid_order_stale"] = self.get_values_from_ordered_dict(
            super_snapshot["bid_price"],
            self.divide_ordered_dicts(
                ob_snap["bid_order_stale"], ob_snap["bid_num_death"]
            ),
            default=-1,
        )
        return super_snapshot
