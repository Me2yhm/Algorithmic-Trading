import numpy as np

class TimestampConverter:
    @staticmethod
    def to_milliseconds(timestamp):
        hours = int(timestamp[0:2])
        minutes = int(timestamp[2:4])
        seconds = int(timestamp[4:6])
        milliseconds = int(timestamp[6:9])

        total_milliseconds = hours * 60 * 60 * 1000 + minutes * 60 * 1000 + seconds * 1000 + milliseconds
        return total_milliseconds

    @staticmethod
    def to_timestamp(milliseconds):
        hours = milliseconds // (60 * 60 * 1000)
        milliseconds %= 60 * 60 * 1000
        minutes = milliseconds // (60 * 1000)
        milliseconds %= 60 * 1000
        seconds = milliseconds // 1000
        milliseconds %= 1000

        timestamp = f"{hours:02d}{minutes:02d}{seconds:02d}{milliseconds:03d}"
        return timestamp

class SignalDeliverySimulator:
    def __init__(self, start_timestamp, end_timestamp):
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.current_timestamp = TimestampConverter.to_milliseconds(start_timestamp)
        self.update_interval = 3 * 1000
        self.std_deviation = 1

    def select_signal_delivery_time(self):
        return int(abs(np.random.normal(self.update_interval / 1000, self.std_deviation)))

    def simulate_signal_delivery(self):
        data = []

        while self.current_timestamp <= TimestampConverter.to_milliseconds(self.end_timestamp):
            signal_interval = self.select_signal_delivery_time()
            self.current_timestamp += 1000 * signal_interval

            if "113000000" <= TimestampConverter.to_timestamp(self.current_timestamp) <= "130000000":
                continue

            data.append(("trade", self.current_timestamp))
            self.current_timestamp += 1000 * (6 - signal_interval)

        fixed_timestamp = TimestampConverter.to_milliseconds("093000000")
        while fixed_timestamp <= TimestampConverter.to_milliseconds(self.end_timestamp):
            data.append(("update", fixed_timestamp))
            fixed_timestamp += self.update_interval

        data.sort(key=lambda x: x[1])

        return data

if __name__ == "__main__":
    start_timestamp = "093000000"
    end_timestamp = "145700000"

    simulator = SignalDeliverySimulator(start_timestamp, end_timestamp)
    simulated_data = simulator.simulate_signal_delivery()

    Time_dict = {}

    for entry in simulated_data:
        timestamp = TimestampConverter.to_timestamp(entry[1])
        if timestamp in Time_dict:
            Time_dict[timestamp]["trade"] = Time_dict[timestamp]["trade"] or (entry[0] == "trade")
            Time_dict[timestamp]["update"] = Time_dict[timestamp]["update"] or (entry[0] == "update")
        else:
            Time_dict[timestamp] = {"trade": (entry[0] == "trade"), "update": (entry[0] == "update")}

    for timestamp, attributes in Time_dict.items():
        print(f"Timestamp: {timestamp}, Attributes: {attributes}")
