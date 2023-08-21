import argparse
import warnings
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import torch as t

from AlgorithmicStrategy import (
    DataSet,
    OrderBook,
    Writer,
    Standarder,
    SignalDeliverySimulator,
    LimitedQueue,
)
from tqdm import tqdm

tick_folder = Path.cwd() / "../datas/000157.SZ/tick/gtja/"
tick_files = list(tick_folder.glob("*.csv"))

raw_data_folder = Path.cwd() / "RAW"
if not raw_data_folder.exists():
    raw_data_folder.mkdir(parents=True, exist_ok=True)


for tick_file in tqdm(tick_files):
    tick = DataSet(data_path=tick_file, ticker="SZ")
    begin_time = str(tick.file_date_num + 93000000)
    end_time = str(tick.file_date_num + 145700000)
    simulator = SignalDeliverySimulator(
        start_timestamp=begin_time, end_timestamp=end_time
    )
    simulated_data = simulator.simulate_signal_delivery()

    writer = Writer(filename=raw_data_folder / tick_file.name)
    ob = OrderBook(data_api=tick)
    ob.update()
    for ts, action in simulator.time_dict.items():
        newest_data = writer.collect_data_by_timestamp(
            ob, timestamp=ts, timestamp_prev=writer.get_prev_timestamp(ts)
        )

        writer.csvwriter.writerow(newest_data)
