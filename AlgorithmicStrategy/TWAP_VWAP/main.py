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
)
from log import logger, log_eval, log_train
from utils import setup_seed, plotter
from tqdm import tqdm

warnings.filterwarnings("ignore")


@logger.catch
def main(opts: argparse.Namespace):
    logger.info("Starting".center(40, "="))

    dataset_path: Path = Path(__file__).parent / opts.dataset
    assert dataset_path.exists(), "Dataset path does not exist!"
    logger.info(f"Reading dataset from {dataset_path}")

    model_save_path: Path = Path(__file__).parent / opts.model_save
    if not model_save_path.exists():
        model_save_path.mkdir()
    logger.info(f"Saving model parameters to {model_save_path}")

    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    logger.info(f"Set device: {device}")

    setup_seed(opts.seed)
    logger.info("Set seed: {}".format(opts.seed))

    log_train(epoch=1, epochs=opts.epoch, step=1, steps=20, loss=1.5, acc=0.65)
    log_train(epoch=1, epochs=opts.epoch, step=10, steps=20, loss=1.5, acc=0.65)
    log_eval(epoch=1, acc=0.53)

    plotter(range(8), ylabel="acc", show=False, path="./PICS/test.png")


def show_total_order_number(ob: OrderBook):
    logger.info(f"TOTAL BID NUMBER: {sum(ob.last_snapshot['bid_num'].values())}")
    logger.info(f"TOTAL ASK NUMBER: {sum(ob.last_snapshot['ask_num'].values())}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Arguments for the strategy", add_help=True)
    parser.add_argument("-s", "--seed", type=int, default=2333, help="set random seed")
    parser.add_argument("-e", "--epoch", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="./DATA/ML")
    parser.add_argument("--model-save", type=str, default="./MODEL_SAVE")
    args = parser.parse_args()

    tick_folder = Path.cwd() / "../datas/002703.SZ/tick/gtja/"
    tick_files = list(tick_folder.glob("*.csv"))

    raw_data_folder = Path.cwd() / "RAW"
    if not raw_data_folder.exists():
        raw_data_folder.mkdir(parents=True, exist_ok=True)


    """
    Scripts begin
    """
    timer = Trade_Update_time(start_timestamp = "093006000", end_timestamp = "145700000")
    for tick_file in tqdm(tick_files):
        tick = DataSet(data_path=tick_file, ticker="SZ")
        ob = OrderBook(data_api=tick)
        writer = Writer(filename=raw_data_folder / tick_file.name)
        timer.get_trade_update_time()
        print(len(timer.time_dict))
        signals = []
        # for ts, action in timer.time_dict.items():
        #     timestamp = int(ts) + tick.file_date_num
        #     ob.update(until=timestamp)
        #     if action['trade']:
        #         pass
        #
        #     if action['update']:
        #         newest_data = writer.collect_data_by_timestamp(
        #             ob,
        #             timestamp = timestamp,
        #             timestamp_prev= writer.get_prev_timestamp(timestamp)
        #         )
        #         writer.csvwriter.writerow(newest_data)
        #         # newest_data = pd.DataFrame([newest_data], columns=writer.columns)
        break