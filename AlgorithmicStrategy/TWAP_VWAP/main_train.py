import argparse
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch as t
from matplotlib import pyplot as plt
from torch import optim

sys.path.append(str(Path(__file__).parent.parent.parent))

from AlgorithmicStrategy import DataSet, OrderBook, TradeTime

from MODELS import JoyeLOB, OCET, LittleOB, MultiTaskLoss

from log import logger, log_eval, log_train
from utils import setup_seed, plotter, save_model
from tqdm import tqdm

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = ArgumentParser(description="Arguments for the strategy", add_help=True)
    parser.add_argument("-s", "--seed", type=int, default=2333, help="set random seed")
    parser.add_argument("-e", "--epoch", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="./DATA/ML")
    parser.add_argument("--model-save", type=str, default="./MODEL_SAVE")
    args = parser.parse_args()

    logger.info("Starting".center(40, "="))
    # t.set_default_tensor_type(t.cuda.FloatTensor)
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    logger.info(f"Set device: {device}")

    setup_seed(args.seed)
    logger.info("Set seed: {}".format(args.seed))

    tick_folder = Path.cwd() / "../datas/000157.SZ/tick/gtja/"
    tick_files = list(tick_folder.glob("*.csv"))

    model_save_path: Path = Path().cwd() / args.model_save
    if not model_save_path.exists():
        model_save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving model parameters to {model_save_path}")

    raw_data_folder = Path.cwd() / "DATA/ML/RAW"
    norm_data_folder = Path.cwd() / "DATA/ML/NORM"
    label_data_folder = Path.cwd() / "DATA/ML/LABEL"
    little_lob_folder = Path.cwd() / "DATA/ML/LittleOB"

    train_folder = Path().cwd() / "DATA/ML/000157/train"
    test_folder = Path.cwd() / "DATA/ML/000157/test"
    train_files = list(train_folder.glob("*.csv"))
    test_files = list(test_folder.glob("*.csv"))

    joye_data = JoyeLOB(window=100)
    llob = LittleOB(direction="BUY")
    loss_func = MultiTaskLoss()
    ocet = OCET(
        num_classes=1,
        dim=100,
        depth=2,
        heads=4,
        dim_head=25,
        mlp_dim=200,
    )
    newest_model = model_save_path / "1.ocet"
    para_dict = t.load(newest_model, map_location=device)
    ocet.load_state_dict(para_dict["model_state_dict"])
    ocet.to(device=device)

    optimizer = optim.Adam(ocet.parameters(), lr=0.001, weight_decay=0.0005)
    optimizer.load_state_dict(para_dict["optimizer_state_dict"])

    # logger.info(f"Model = {str(ocet)}")
    # logger.info("Model parameters = %d" % sum(p.numel() for p in ocet.parameters()))

    """
    Scripts begin
    """
    loss_global = []
    true_vwaps = []
    pred_vwaps = []
    for epc in tqdm(range(int(newest_model.stem) + 1, args.epoch)):
        loss_log = []
        for file in train_files:
            if file not in joye_data:
                joye_data.push(file)
            tick = DataSet(file, ticker="000157.SZ")
            llob_file = little_lob_folder / file.name
            if llob_file not in llob:
                llob.push(llob_file)
            tt = TradeTime(begin=9_30_00_000, end=14_57_00_000, tick=tick)
            pred_trade_volume_fracs = []
            true_trade_volume_fracs = []
            hist_trade_volume_fracs = []
            trade_price = []

            for ts, action in tt.generate_signals():
                if action["trade"]:
                    time_search, X, volume_hist, volume_today = joye_data.batch(
                        file, timestamp=ts
                    )
                    if time_search is not None:
                        X = t.tensor(X, device=device, dtype=t.float32)
                        pred_frac = ocet(X)

                        _, price = llob.batch(llob_file, ts)

                        trade_price.append(price)

                        hist_trade_volume_fracs.append(volume_hist)
                        pred_trade_volume_fracs.append(pred_frac)
                        true_trade_volume_fracs.append(volume_today)

            market_vwap = llob.get_VWAP(llob_file)
            true_vwaps.append(market_vwap)
            pred_trade_volume_fracs = t.squeeze(t.stack(pred_trade_volume_fracs))

            additional_vwap = 0
            if t.sum(pred_trade_volume_fracs) < 1:
                rest = 1 - t.sum(pred_trade_volume_fracs)
                _, final_price = llob.batch(
                    llob_file, tick.file_date_num + 14_57_00_000
                )
                additional_vwap = rest * final_price

            trade_price = t.cuda.FloatTensor(trade_price)
            pred_vwap = t.sum(pred_trade_volume_fracs * trade_price) + additional_vwap
            pred_vwaps.append(pred_vwap.item())

            hist_trade_volume_fracs = t.cuda.FloatTensor(hist_trade_volume_fracs)
            hist_trade_volume_fracs = hist_trade_volume_fracs / t.sum(
                hist_trade_volume_fracs
            )

            true_trade_volume_fracs = t.cuda.FloatTensor(true_trade_volume_fracs)
            true_trade_volume_fracs = true_trade_volume_fracs / t.sum(
                true_trade_volume_fracs
            )

            loss = loss_func.calculate_loss(
                pred_trade_volume_fracs,
                true_trade_volume_fracs,
                hist_trade_volume_fracs,
                market_vwap,
                pred_vwap,
            )
            loss.backward()
            optimizer.step()
            loss_log.append(loss.item())
            log_train(epoch=epc, epochs=args.epoch, file=file.stem, loss=loss.item())

        loss_global.extend(loss_log)

        save_model(
            model=ocet,
            optimizer=optimizer,
            epoch=epc,
            loss=float(np.mean(loss_log)),
            path=model_save_path / f"{epc}.ocet",
        )
    plotter(loss_global, ylabel="loss")
    # plotter(loss_global, ylabel='VWAP')
