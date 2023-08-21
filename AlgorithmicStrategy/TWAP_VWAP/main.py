import argparse
import warnings
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch as t

from AlgorithmicStrategy import DataSet, OrderBook, Writer, Normalized_reader, Normalizer,LimitedQueue
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
    parser.add_argument("--dataset", type=str, default="./DATA/ML")
    parser.add_argument("--model-save", type=str, default="./MODEL_SAVE")
    args = parser.parse_args()

    # normal_folder = Path.cwd() / "Example"
    # normal_folder = Path('D:\Fudan\Work\JoyE_work\AlgorithmicStrategy\AlgorithmicStrategy\TWAP_VWAP\DATA\KangYang\训练')

    # original_folder = Path.cwd() / "DATA" / "KangYang" / "测试集3s"

    # nm = Normalizer(file_folder=original_folder)

    # nm.initialize_output(output_path=Path.cwd() / "Example")

    # nr = Normalized_reader(normal_folder)

    # print(nr.generate_inputs('0704').shape)

    # tick_path = Path.cwd() / "../datas/002703.SZ/tick/gtja/2023-07-03.csv"
    # tick = DataSet(data_path=tick_path, ticker="SZ")
    # ob = OrderBook(data_api=tick, decay_rate=5)

    # until = None
    # ob.update(until=until)
    # w = Writer(filename='example.csv')
    # w.collect_data_order_book(ob)


    lq = LimitedQueue(100)
    df = pd.read_csv(r'D:\算法交易\Algorithmic-Trading\AlgorithmicStrategy\TWAP_VWAP\DATA\KangYang\训练集3s\000157_3s_0704.csv')
    for i in range(105):
        data = df.iloc[i,:]
        lq.push(data)
    matrix = lq.form_matrix()
    print(matrix[:,0] == df.loc[5:104,'timestamp'])
    # for i in range(1,100):
    #     value1 = matrix[0,-1]
    #     value2 = matrix[0,-2]
    #     print(matrix[-i,0] > matrix[-i-1,0])