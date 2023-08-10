import argparse
import csv
import warnings
from argparse import ArgumentParser
from pathlib import Path

import torch as t

from AlgorithmicStrategy import DataSet, OrderBook
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

    tick_path = Path.cwd() / "../datas/000001.SZ/tick/gtja/2023-07-03.csv"
    tick = DataSet(data_path=tick_path, ticker="SZ")
    ob = OrderBook(data_api=tick, decay_rate=5)

    until = 2023_07_03_09_37_00_130
    ob.update(until=until)
    time_stamp = 2023_07_03_09_25_00_130

    class Writer:
        def __init__(self, filename:str, features: list[str] = None,  **kwargs):
            self.filename = filename
            self.file = open(self.filename, 'w', newline='', encoding='utf-8')
            self.csvwriter = csv.writer(self.file)
            self.features = (
                features
                if features is not None
                else [
                    "candle",
                    "candle_range",
                    "snapshot",
                    "VWAP",
                    "VWAP_range",
                    "depth",
                ]
            )
            self.rollback = kwargs.get("rollback", 5000)
            self.bid_ask_num = kwargs.get("bid_ask_num", 10)
            self.columns = self.init_columns()
            self.csvwriter.writerow(self.columns)


        def collect_data_by_timestamp(self, ob: OrderBook, timestamp: int):
            timestamp_prev = timestamp - self.rollback
            print(timestamp_prev, timestamp)
            nearest_snapshot = ob.search_snapshot(timestamp)
            res = []
            for f in self.features:
                if f == 'candle':
                    res.extend(ob.search_candle(timestamp))
                elif f == 'candle_range':
                    res.extend(ob.get_candle_slot(timestamp_prev, timestamp))
                elif f == 'snapshot':
                    for i in ob.get_super_snapshot(self.bid_ask_num, timestamp).values():
                        res.extend(i)
                elif f == "VWAP":
                    res.extend(nearest_snapshot["total_trade"])
                elif f == "VWAP_range":
                    tmp = ob.get_avg_trade(timestamp_prev, timestamp)
                    res.extend(tmp[0].values())
                    res.append(tmp[1])
                elif f == 'depth':
                    res.append(nearest_snapshot["order_depth"]["weighted_average_depth"])
            assert len(res) == len(self.columns)
            return res

        def collect_data_order_book(self, ob: OrderBook):
            begin_stamp = ob.data_api.file_date_num + 9_30_00_000
            end_stamp = ob.last_snapshot['timestamp']
            for ts in range(begin_stamp + self.rollback, end_stamp, self.rollback):
                tmp_data = self.collect_data_by_timestamp(ob, ts)
                self.csvwriter.writerow(tmp_data)

        def init_columns(self):
            columns = []
            for feature in self.features:
                if feature == "candle":
                    """
                    前一次收盘、开盘、最高、最低、收盘
                    """
                    columns.extend(["preclose", "open", "high", "low", "close"])
                elif feature == "candle_range":
                    """
                    时段的candle数据
                    """
                    columns.extend(
                        [
                            "preclose_range",
                            "open_range",
                            "high_range",
                            "low_range",
                            "close_range",
                        ]
                    )
                elif feature == "snapshot":
                    """
                    超级盘口，包含买卖十档、买卖十档交易量、买卖十档订单数、买卖十档累计新陈代谢
                    """
                    columns.extend(
                        ["ask_price_" + str(i) for i in range(self.bid_ask_num)]
                    )
                    columns.extend(
                        ["ask_volume_" + str(i) for i in range(self.bid_ask_num)]
                    )
                    columns.extend(
                        ["bid_price_" + str(i) for i in range(self.bid_ask_num)]
                    )
                    columns.extend(
                        ["bid_volume_" + str(i) for i in range(self.bid_ask_num)]
                    )
                    columns.extend(
                        ["ask_order_num_" + str(i) for i in range(self.bid_ask_num)]
                    )
                    columns.extend(
                        ["bid_order_num_" + str(i) for i in range(self.bid_ask_num)]
                    )
                    columns.extend(
                        ["ask_order_stale_" + str(i) for i in range(self.bid_ask_num)]
                    )
                    columns.extend(
                        ["bid_order_stale_" + str(i) for i in range(self.bid_ask_num)]
                    )
                elif feature == "VWAP":
                    """
                    任意时刻成交单VWAP、成交量、成交单数、被动方新陈代谢
                    """
                    columns.extend(
                        [
                            "order_num",
                            "volume",
                            "VWAP",
                            "amount",
                            "passive_num",
                            "passive_stale_total",
                        ]
                    )
                elif feature == "VWAP_range":
                    """
                    任意时间段成交单VWAP、成交量、成交单数、被动方新陈代谢
                    """
                    columns.extend(
                        [
                            "order_num_range",
                            "volume_range",
                            "VWAP_range",
                            "amount_range",
                            "passive_num_range",
                            "passive_stale_total_range",
                            "passive_stale_avg_range"
                        ]
                    )
                elif feature == "depth":
                    """
                    任意时刻的平均市场深度
                    """
                    columns.append("depth")
            return columns

    def __del__(self):
        self.file.close()

    w = Writer(filename='example.csv')
    w.collect_data_order_book(ob)

    logger.info("任意时刻的candle数据")
    logger.info("前一次收盘、开盘、最高、最低、收盘")
    logger.info(ob.search_candle(time_stamp))
    tmp = []
    tmp.extend(ob.get_super_snapshot(10, time_stamp).values())
    print(tmp)
    logger.info("任意时段的candle数据")
    logger.info(ob.get_candle_slot(time_stamp, until))
    logger.info("任意时刻的超级盘口，包含买卖十档、买卖十档交易量、买卖十档订单数、买卖十档累计新陈代谢")
    logger.info(ob.get_super_snapshot(10, time_stamp))
    logger.info("任意时刻成交单VWAP、成交量、成交单数、被动方新陈代谢")
    logger.info(ob.search_snapshot(time_stamp)["total_trade"])
    logger.info("任意时间段成交单VWAP、成交量、成交单数、被动方新陈代谢")
    logger.info(ob.get_avg_trade(time_stamp, until))
    logger.info("任意时刻的平均市场深度")
    logger.info(ob.search_snapshot(time_stamp)["order_depth"]["weighted_average_depth"])

    # logger.info(ob.last_snapshot['timestamp'])
    # logger.info(ob.last_snapshot['ask'])
    # logger.info(ob.last_snapshot['bid'])
    # logger.info(ob.data_api.data_cache[0])
    # logger.info(tick.fresh())
    # logger.info(tick.fresh())
    # logger.info(tick.fresh())
    # logger.info(tick.fresh())

    # until = 2023_07_03_09_31_00_010
    # ob.update(until=until)
    #
    # show_total_order_number(ob)
    # logger.info(ob.last_snapshot['timestamp'])
    # logger.info(ob.last_snapshot['ask'])
    # logger.info(ob.last_snapshot['bid'])
    # logger.info(ob.data_cache[0])
    # main(args)
