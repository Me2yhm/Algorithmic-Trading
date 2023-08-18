import argparse
import warnings
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch as t

from AlgorithmicStrategy import DataSet, OrderBook, Writer, Normalized_reader
from log import logger, log_eval, log_train
from utils import setup_seed, plotter

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
    parser.add_argument(
        "--train",
        type=str,
        default=Path(__file__).parent / "DATA" / "ML" / "000157" / "train",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=Path(__file__).parent / "DATA" / "ML" / "000157" / "test",
    )
    parser.add_argument("--model-save", type=str, default="./MODEL_SAVE")
    args = parser.parse_args()

    logger.info(f"Using training data: {args.train}")
    logger.info(f"Using testing data: {args.test}")

    norm_reader = Normalized_reader(file_folder=args.train)

    logger.info(norm_reader.dfs.keys())
    inputs, trade_info = norm_reader.generate_inputs('0703')
    logger.info(list(inputs.values())[0].shape)

    # tick_path = Path.cwd() / "../datas/002703.SZ/tick/gtja/2023-07-03.csv"
    # tick = DataSet(data_path=tick_path, ticker="SZ")
    # ob = OrderBook(data_api=tick, decay_rate=5)
    #
    # until = None
    # ob.update(until=until)
    # w = Writer(filename='example.csv')
    # w.collect_data_order_book(ob)

    # path = r"D:\Fudan\Work\JoyE_work\AlgorithmicStrategy\AlgorithmicStrategy\TWAP_VWAP\DATA\KangYang\归一化测试集3s\000157_3s_0726.csv"
    # df = pd.read_csv(path)
    # print(df.loc[1451, 'depth'])
    # print(sum(df['depth'] > 0))
