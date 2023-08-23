from pathlib import Path

from tqdm import tqdm

from AlgorithmicStrategy import (
    DataSet,
    OrderBook,
    Writer,
    TradeTime
)

tick_folder = Path.cwd() / "../datas/000157.SZ/tick/gtja/"
tick_files = list(tick_folder.glob("*.csv"))

raw_data_folder = Path.cwd() / "RAW"
if not raw_data_folder.exists():
    raw_data_folder.mkdir(parents=True, exist_ok=True)


for tick_file in tqdm(tick_files):
    tick = DataSet(data_path=tick_file, ticker="SZ")
    tt = TradeTime(begin=93000000, end=145700000)
    time_dict = tt.generate_signals()
    writer = Writer(filename=raw_data_folder / tick_file.name)
    ob = OrderBook(data_api=tick)
    ob.update()
    for ts, action in time_dict:
        if action['update']:
            newest_data = writer.collect_data_by_timestamp(
                ob, timestamp=ts, timestamp_prev=writer.get_prev_timestamp(ts)
            )

            writer.csvwriter.writerow(newest_data)
