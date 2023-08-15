import numpy as np
import csv

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
        self.mean_time = 3
        self.std_deviation = 1

    def select_signal_delivery_time(self):
        return int(abs(np.random.normal(self.mean_time, self.std_deviation)))

    def simulate_signal_delivery(self):
        data = []
        while self.current_timestamp <= TimestampConverter.to_milliseconds(self.end_timestamp):
            signal_interval = self.select_signal_delivery_time()
            self.current_timestamp += 1000 * signal_interval

            if "113000000" <= TimestampConverter.to_timestamp(self.current_timestamp) <= "130000000":
                continue

            data.append(self.current_timestamp)
            self.current_timestamp += 1000 * (6 - signal_interval)

        return data

if __name__ == "__main__":
    start_timestamp = "093000000"
    end_timestamp = "145700000"

    simulator = SignalDeliverySimulator(start_timestamp, end_timestamp)
    simulated_data = simulator.simulate_signal_delivery()

    with open("timestamp1.csv", "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Timestamp"])

        for timestamp in simulated_data:
            csv_writer.writerow([TimestampConverter.to_timestamp(timestamp)])
