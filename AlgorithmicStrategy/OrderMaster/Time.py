import numpy as np
import csv


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

    @property
    def time_dict(self):
        time_dict = {}
        simulated_data = simulator.simulate_signal_delivery()
        for entry in simulated_data:
            timestamp = int(TimestampConverter.to_timestamp(entry[1]))
            if 20230704113000000 <= timestamp <= 20230704130000000:
                continue
            if timestamp in time_dict:
                time_dict[timestamp]["trade"] = time_dict[timestamp]["trade"] or (
                    entry[0] == "trade"
                )
                time_dict[timestamp]["update"] = time_dict[timestamp]["update"] or (
                    entry[0] == "update"
                )
            else:
                time_dict[timestamp] = {
                    "trade": (entry[0] == "trade"),
                    "update": (entry[0] == "update"),
                }


if __name__ == "__main__":
    start_timestamp = "20230704093000000"
    end_timestamp = "20230704145700000"

    simulator = SignalDeliverySimulator(start_timestamp, end_timestamp)
    simulated_data = simulator.simulate_signal_delivery()

    print(simulator.time_dict)
