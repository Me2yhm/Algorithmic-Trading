import csv
from pathlib import Path

from tqdm import tqdm

from .OrderBook import OrderBook
from .DataManager import DataSet
from datetime import datetime, timedelta
from collections import OrderedDict


class Writer:
    def __init__(self, filename: Path, features: list[str] = None, **kwargs):
        self.filename: Path = filename
        self.file = open(self.filename, "w", newline="", encoding="utf-8")
        self.csvwriter = csv.writer(self.file)
        self.features = (
            features
            if features is not None
            else [
                "candle",
                "candle_range",
                "snapshot",
                "VWAP",
                "VWAP_range",
                "depth",
            ]
        )
        self.rollback = kwargs.get("rollback", 3000)
        self.bid_ask_num = kwargs.get("bid_ask_num", 10)
        self.columns = self.init_columns()
        self.csvwriter.writerow(self.columns)

    def collect_data_by_timestamp(
        self, ob: OrderBook, timestamp: int, timestamp_prev: int
    ):
        # logger.info(f"WRITING DATA: {timestamp_prev}-{timestamp}")
        nearest_snapshot = ob.search_snapshot(timestamp)
        res = [timestamp]
        for f in self.features:
            if f == "candle":
                res.extend(ob.search_candle(timestamp))
            elif f == "candle_range":
                res.extend(ob.get_candle_slot(timestamp_prev, timestamp))
            elif f == "snapshot":
                for i in ob.get_super_snapshot(self.bid_ask_num, timestamp).values():
                    res.extend(i)
            elif f == "VWAP":
                res.extend(nearest_snapshot["total_trade"].values())
            elif f == "VWAP_range":
                tmp = ob.get_avg_trade(timestamp_prev, timestamp)
                res.extend(tmp[0].values())
                res.append(tmp[1])
            elif f == "depth":
                res.append(nearest_snapshot["order_depth"]["weighted_average_depth"])
        assert len(res) == len(self.columns)
        return res

    def get_prev_timestamp(self, timestamp: int):
        timestamp = datetime.strptime(str(timestamp), "%Y%m%d%H%M%S%f")
        prev_timestamp = timestamp - timedelta(microseconds=self.rollback * 1e3)
        return int(prev_timestamp.strftime("%Y%m%d%H%M%S%f")[:-3])

    def collect_data_order_book(self, ob: OrderBook):
        begin_stamp = ob.data_api.file_date_num + 9_30_00_000
        dt_begin = datetime.strptime(str(begin_stamp), "%Y%m%d%H%M%S%f")
        end_stamp = ob.last_snapshot["timestamp"]
        dt_end = datetime.strptime(str(end_stamp), "%Y%m%d%H%M%S%f")

        time_diff = dt_end - dt_begin
        total_iterations = time_diff // timedelta(microseconds=self.rollback * 1e3)

        with tqdm(total=total_iterations, desc="Processing", unit="iteration") as pbar:
            while dt_begin < dt_end:
                timestamp_prev = int(dt_begin.strftime("%Y%m%d%H%M%S%f")[:-3])
                new_dt = dt_begin + timedelta(microseconds=self.rollback * 1e3)
                timestamp = int(new_dt.strftime("%Y%m%d%H%M%S%f")[:-3])
                res = self.collect_data_by_timestamp(ob, timestamp, timestamp_prev)
                self.csvwriter.writerow(res)
                dt_begin = new_dt
                pbar.update(1)

    def init_columns(self):
        columns = ["timestamp"]
        for feature in self.features:
            if feature == "candle":
                """
                前一次收盘、开盘、最高、最低、收盘
                """
                columns.extend(["preclose", "open", "high", "low", "close"])
            elif feature == "candle_range":
                """
                时段的candle数据
                """
                columns.extend(
                    [
                        "preclose_range",
                        "open_range",
                        "high_range",
                        "low_range",
                        "close_range",
                    ]
                )
            elif feature == "snapshot":
                """
                超级盘口，包含买卖十档、买卖十档交易量、买卖十档订单数、买卖十档累计新陈代谢
                """
                columns.extend(["ask_price_" + str(i) for i in range(self.bid_ask_num)])
                columns.extend(
                    ["ask_volume_" + str(i) for i in range(self.bid_ask_num)]
                )
                columns.extend(["bid_price_" + str(i) for i in range(self.bid_ask_num)])
                columns.extend(
                    ["bid_volume_" + str(i) for i in range(self.bid_ask_num)]
                )
                columns.extend(
                    ["ask_order_num_" + str(i) for i in range(self.bid_ask_num)]
                )
                columns.extend(
                    ["bid_order_num_" + str(i) for i in range(self.bid_ask_num)]
                )
                columns.extend(
                    ["ask_order_stale_" + str(i) for i in range(self.bid_ask_num)]
                )
                columns.extend(
                    ["bid_order_stale_" + str(i) for i in range(self.bid_ask_num)]
                )
            elif feature == "VWAP":
                """
                任意时刻成交单VWAP、成交量、成交单数、被动方新陈代谢
                """
                columns.extend(
                    [
                        "order_num",
                        "volume",
                        "VWAP",
                        "amount",
                        "passive_num",
                        "passive_stale_total",
                    ]
                )
            elif feature == "VWAP_range":
                """
                任意时间段成交单VWAP、成交量、成交单数、被动方新陈代谢
                """
                columns.extend(
                    [
                        "order_num_range",
                        "volume_range",
                        "VWAP_range",
                        "amount_range",
                        "passive_num_range",
                        "passive_stale_total_range",
                        "passive_stale_avg_range",
                    ]
                )
            elif feature == "depth":
                """
                任意时刻的平均市场深度
                """
                columns.append("depth")
        return columns

    def __del__(self):
        self.file.close()

class LLobWriter:
    def __init__(self, tick: DataSet, orderbook: OrderBook, file_name: Path):
        self.tick = tick
        self.ob = orderbook
        self.columns = ["Timestamp", "ask1", "bid1", "VWAP"]
        self.little_ob = OrderedDict()
        self.file_name = file_name

    def write_llob(self):
        for ts in self.ob.snapshots.keys():
            if ts % 1000000000 >= 93000000 and ts % 1000000000 <= 145700000:
                if list(self.ob.snapshots[ts]["ask"].keys()):
                    self.little_ob[ts] = OrderedDict()
                    self.little_ob[ts]["ask1"] = list(
                        self.ob.snapshots[ts]["ask"].keys()
                    )[0]
                if list(self.ob.snapshots[ts]["bid"].keys()):
                    self.little_ob[ts]["bid1"] = list(
                        self.ob.snapshots[ts]["bid"].keys()
                    )[0]
                if list(self.ob.snapshots[ts]["total_trade"].values()):
                    self.little_ob[ts]["VWAP"] = list(
                        self.ob.snapshots[ts]["total_trade"].values()
                    )[2]
        data_list = []
        for ts, values in self.little_ob.items():
            data_list.append([ts] + [values.get(key, "") for key in self.columns[1:]])
        with open(str(self.file_name), mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(self.columns)
            writer.writerows(data_list)
