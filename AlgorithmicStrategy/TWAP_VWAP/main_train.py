import argparse
import warnings
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import torch as t
from torch import optim

from AlgorithmicStrategy import (
    DataSet,
    OrderBook,
    Writer,
    Standarder,
    LimitedQueue,
    TradeTime
)

from MODELS import JoyeLOB, OCET, LittleOB, MultiTaskLoss

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
    parser.add_argument("-e", "--epoch", type=int, default=2)
    parser.add_argument("--dataset", type=str, default="./DATA/ML")
    parser.add_argument("--model-save", type=str, default="./MODEL_SAVE")
    args = parser.parse_args()

    logger.info("Starting".center(40, "="))
    t.set_default_tensor_type(t.cuda.FloatTensor)
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    logger.info(f"Set device: {device}")

    setup_seed(args.seed)
    logger.info("Set seed: {}".format(args.seed))

    tick_folder = Path.cwd() / "../datas/000157.SZ/tick/gtja/"
    tick_files = list(tick_folder.glob("*.csv"))

    raw_data_folder = Path.cwd() / "DATA/ML/RAW"
    norm_data_folder = Path.cwd() / "DATA/ML/NORM"
    label_data_folder = Path.cwd() / "DATA/ML/LABEL"
    little_lob_folder = Path.cwd() / "DATA/ML/LittleOB"

    train_folder = Path().cwd() / "DATA/ML/000157/train"
    test_folder = Path.cwd() / "DATA/ML/000157/test"
    train_files = list(train_folder.glob("*.csv"))
    test_files = list(test_folder.glob("*.csv"))

    joye_data = JoyeLOB(window=100)
    llob = LittleOB(direction='BUY')
    loss_func = MultiTaskLoss(

    )
    ocet = OCET(
        num_classes=1,
        dim=100,
        depth=2,
        heads=4,
        dim_head=25,
        mlp_dim=200,
    )
    ocet.to(device=device)
    optimizer = optim.Adam(ocet.parameters(), lr=0.001, weight_decay=0.0005)

    # logger.info(f"Model = {str(ocet)}")
    # logger.info("Model parameters = %d" % sum(p.numel() for p in ocet.parameters()))

    """
    Scripts begin
    """
    loss_log = []
    for epc in tqdm(range(args.epoch)):
        for file in train_files:
            if file not in joye_data:
                joye_data.push(file)
            tick = DataSet(file, ticker='000157.SZ')
            llob.push(file)
            tt = TradeTime(begin=9_30_00_000, end=14_57_00_000, tick=tick)
            pred_trade_volume_fracs = []
            true_trade_volume_fracs = []
            hist_trade_volume_fracs = []
            trade_price = []
            for ts, action in tt.generate_signals():
                if action['trade']:
                    time_search, X, volume_hist, volume_today = joye_data.batch(file, timestamp=ts)
                    if time_search is not None:
                        X = t.tensor(X, device=device, dtype=t.float32)
                        pred_frac = ocet(X)

                        _, price = llob.batch(file, ts)

                        trade_price.append(price)

                        hist_trade_volume_fracs.append(volume_hist)
                        pred_trade_volume_fracs.append(pred_frac)
                        true_trade_volume_fracs.append(volume_today)

            trade_price = t.tensor(trade_price, dtype=t.float32)
            pred_trade_volume_fracs = t.stack(pred_trade_volume_fracs)

            market_vwap = llob.get_VWAP(file)
            pred_vwap = t.sum(pred_trade_volume_fracs * trade_price)
            # pred_vwap = t.mm(pred_trade_volume_fracs, trade_price)
            # loss = pred_vwap - market_vwap

            # pred_trade_volume_fracs = t.tensor(pred_trade_volume_fracs, requires_grad=True,  dtype=t.float32)
            # trade_price = t.tensor(trade_price)
            # pred_vwap = t.sum(pred_trade_volume_fracs * trade_price)

            hist_trade_volume_fracs = t.tensor(hist_trade_volume_fracs, dtype=t.float32)
            hist_trade_volume_fracs = hist_trade_volume_fracs/t.sum(hist_trade_volume_fracs)

            true_trade_volume_fracs = t.tensor(true_trade_volume_fracs, dtype=t.float32)
            true_trade_volume_fracs = true_trade_volume_fracs/t.sum(true_trade_volume_fracs)

            loss = loss_func.calculate_loss(
                pred_trade_volume_fracs,
                true_trade_volume_fracs,
                hist_trade_volume_fracs,
                market_vwap,
                pred_vwap
            )
            loss.backward()
            optimizer.step()
            loss_log.append(loss.item())
            log_train(epoch=epc, epochs=args.epoch, file=file.stem, loss=loss.item())





    # for tick_file in tqdm(tick_files):
    #     lq = LimitedQueue(max_size=100)
    #     tick = DataSet(data_path=tick_file, ticker="SZ")
    #     ob = OrderBook(data_api=tick)
    #     writer = Writer(filename=raw_data_folder / tick_file.name)
    #     time_dict = {}
    #     signals = []
    #     for ts, action in time_dict.items():
    #         timestamp = int(ts) + tick.file_date_num
    #         ob.update(until=timestamp)
    #         if action["trade"]:
    #             pass
    #
    #         if action["update"]:
    #             newest_data = writer.collect_data_by_timestamp(
    #                 ob,
    #                 timestamp=timestamp,
    #                 timestamp_prev=writer.get_prev_timestamp(timestamp),
    #             )
    #             writer.csvwriter.writerow(newest_data)
    #             newest_data = pd.DataFrame([newest_data], columns=writer.columns)
    #             lq.push(newest_data)
    #
    #         if lq.size == 100:
    #             df = lq.to_df()
    #             print(df.shape)
    #             break
    #     break
