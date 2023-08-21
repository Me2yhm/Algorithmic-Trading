from pathlib import Path

from tqdm import tqdm

from AlgorithmicStrategy import (
    DataSet,
    OrderBook,
    Writer,
    SignalDeliverySimulator,
    TimestampConverter,
)

tick_folder = Path.cwd() / "../datas/000157.SZ/tick/gtja/"
tick_files = list(tick_folder.glob("*.csv"))

raw_data_folder = Path.cwd() / "RAW"
if not raw_data_folder.exists():
    raw_data_folder.mkdir(parents=True, exist_ok=True)


for tick_file in tqdm(tick_files):
    tick = DataSet(data_path=tick_file, ticker="SZ")
    begin_time = str(tick.file_date_num + 93003000)
    end_time = str(tick.file_date_num + 145700000)
    simulator = SignalDeliverySimulator(
        start_timestamp=begin_time, end_timestamp=end_time
    )
    simulated_data = simulator.simulate_signal_delivery()

    time_dict = {}
    for entry in simulated_data:
        timestamp = int(TimestampConverter.to_timestamp(entry[1]))
        if (tick.file_date_num + 113000000) <= timestamp <= (tick.file_date_num + 130000000):
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

    writer = Writer(filename=raw_data_folder / tick_file.name)
    ob = OrderBook(data_api=tick)
    ob.update()
    for ts, action in time_dict.items():
        if action['update']:
            newest_data = writer.collect_data_by_timestamp(
                ob, timestamp=ts, timestamp_prev=writer.get_prev_timestamp(ts)
            )

            writer.csvwriter.writerow(newest_data)
