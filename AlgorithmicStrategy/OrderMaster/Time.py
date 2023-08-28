from datetime import timedelta, datetime
from collections import OrderedDict
from typing import Literal

import numpy as np
import random
from .DataManager import DataSet


class TradeTime:
    def __init__(self, begin: int, end: int, tick: DataSet):
        self.begin: int = begin
        self.end: int = end
        self.tick: DataSet = tick

    @classmethod
    def is_trade_time(cls, timestamp: datetime):
        time_num = int(timestamp.strftime("%Y%m%d%H%M%S%f")[8:])
        if (time_num < 91500000000) or (time_num > 145700000000) or (130000000000 > time_num > 113000000000):
            return False
        else:
            return True

    def generate_timestamps(
        self,
        key: Literal["update", "trade"],
        interval: int = 6000,
        limits=(-3000, 3000),
    ):
        interval = timedelta(microseconds=interval * 1e3)
        trade_timestamps = OrderedDict()
        current_time = datetime.strptime(
            str(self.tick.file_date_num + self.begin), "%Y%m%d%H%M%S%f"
        )
        end_time = datetime.strptime(
            str(self.tick.file_date_num + self.end), "%Y%m%d%H%M%S%f"
        )
        while current_time <= end_time:

            tmp = current_time
            if any(limits):
                tmp += timedelta(
                    microseconds=int(
                        np.random.randint(low=limits[0], high=limits[1])
                    )
                    * 1e3
                )
            if self.is_trade_time(tmp):
                if key == "update":
                    trade_timestamps[int(tmp.strftime("%Y%m%d%H%M%S%f")[:-3])] = {
                        key: True,
                        "trade": False,
                    }
                elif key == "trade":
                    trade_timestamps[int(tmp.strftime("%Y%m%d%H%M%S%f")[:-3])] = {
                        key: True,
                    }
            current_time += interval

        return trade_timestamps

    def generate_signals(self, **kwargs):
        trade = self.generate_timestamps(
            key="trade",
            interval=kwargs.get("trade_interval", 6000),
            limits=kwargs.get("trade_limits", (-2500, 2500)),
        )
        update = self.generate_timestamps(
            key="update",
            interval=kwargs.get("update_interval", 3000),
            limits=kwargs.get("update_limits", (0, 0)),
        )
        for k, v in trade.items():
            if k in update:
                update[k].update(v)
            else:
                update[k] = v
                update[k].update({"update": False})
        return sorted(update.items(), key=lambda x: x[0])


class TimestampConverter:
    @staticmethod
    def to_milliseconds(timestamp: str):
        year = int(timestamp[0:4])
        month = int(timestamp[4:6])
        day = int(timestamp[6:8])
        hours = int(timestamp[8:10])
        minutes = int(timestamp[10:12])
        seconds = int(timestamp[12:14])
        milliseconds = int(timestamp[14:17])

        total_milliseconds = (
            year * 365 * 24 * 60 * 60 * 1000
            + month * 30 * 24 * 60 * 60 * 1000
            + day * 24 * 60 * 60 * 1000
            + hours * 60 * 60 * 1000
            + minutes * 60 * 1000
            + seconds * 1000
            + milliseconds
        )
        return total_milliseconds

    @staticmethod
    def to_timestamp(milliseconds):
        years = milliseconds // (365 * 24 * 60 * 60 * 1000)
        milliseconds %= 365 * 24 * 60 * 60 * 1000
        months = milliseconds // (30 * 24 * 60 * 60 * 1000)
        milliseconds %= 30 * 24 * 60 * 60 * 1000
        days = milliseconds // (24 * 60 * 60 * 1000)
        milliseconds %= 24 * 60 * 60 * 1000
        hours = milliseconds // (60 * 60 * 1000)
        milliseconds %= 60 * 60 * 1000
        minutes = milliseconds // (60 * 1000)
        milliseconds %= 60 * 1000
        seconds = milliseconds // 1000
        milliseconds %= 1000

        timestamp = f"{years:04d}{months:02d}{days:02d}{hours:02d}{minutes:02d}{seconds:02d}{milliseconds:03d}"
        return timestamp


class SignalDeliverySimulator:
    def __init__(self, start_timestamp: str, end_timestamp: str, **kwargs):
        self.start_timestamp: str = start_timestamp
        self.end_timestamp: str = end_timestamp
        self.current_timestamp: int = TimestampConverter.to_milliseconds(
            start_timestamp
        )
        self.update_interval = kwargs.get("interval", 3000)
        self.std_deviation = kwargs.get("std", 1)

    def select_signal_delivery_time(self):
        return int(
            abs(np.random.normal(self.update_interval / 1000, self.std_deviation))
        )

    def simulate_signal_delivery(self):
        data = []

        while self.current_timestamp <= TimestampConverter.to_milliseconds(
            self.end_timestamp
        ):
            signal_interval = self.select_signal_delivery_time()
            self.current_timestamp += 1000 * signal_interval

            data.append(("trade", self.current_timestamp))
            self.current_timestamp += 1000 * (6 - signal_interval)

        fixed_timestamp = TimestampConverter.to_milliseconds(self.start_timestamp)
        while fixed_timestamp <= TimestampConverter.to_milliseconds(self.end_timestamp):
            data.append(("update", fixed_timestamp))
            fixed_timestamp += self.update_interval

        data.sort(key=lambda x: x[1])

        return data
