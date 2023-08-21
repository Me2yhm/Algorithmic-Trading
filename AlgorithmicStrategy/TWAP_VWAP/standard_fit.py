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
    Trade_Update_time,
    LimitedQueue

)
from tqdm import tqdm

tick_folder = Path.cwd() / "../datas/000157.SZ/tick/gtja/"
tick_files = list(tick_folder.glob("*.csv"))

raw_data_folder = Path.cwd() / "RAW"
if not raw_data_folder.exists():
    raw_data_folder.mkdir(parents=True, exist_ok=True)

timer = Trade_Update_time(start_timestamp="093006000", end_timestamp="145700000")
timer.get_trade_update_time()
for tick_file in tqdm(tick_files[1:]):
    tick = DataSet(data_path=tick_file, ticker="SZ")
    writer = Writer(filename=raw_data_folder / tick_file.name)
    ob = OrderBook(data_api=tick)
    ob.update()
    for ts, action in timer.time_dict.items():
        print(ts)
        timestamp = int(ts) + tick.file_date_num
        try:
            newest_data = writer.collect_data_by_timestamp(
                ob,
                timestamp=timestamp,
                timestamp_prev=writer.get_prev_timestamp(timestamp)
            )
        except IndexError as ie:
            print(timestamp)
            raise ie
        writer.csvwriter.writerow(newest_data)
