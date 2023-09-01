import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path

import torch as t

sys.path.append(str(Path(__file__).parent.parent.parent))

from AlgorithmicStrategy import DataSet, TradeTime

from MODELS import (
    JoyeLOB,
    OCET,
    LittleOB,
    MultiTaskLoss,
    logger,
    log_eval,
    setup_seed,
)

from tqdm import tqdm

warnings.filterwarnings("ignore")


@t.no_grad()
def evaluate(ocet: OCET):
    loss_global = []
    true_vwaps = []
    pred_vwaps = []
    for file in tqdm(test_files):
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
                    X = t.tensor(X, dtype=t.float32)
                    # logger.info(X.size())
                    pred_frac = ocet(X)

                    _, price = llob.batch(llob_file, ts)

                    trade_price.append(price)

                    hist_trade_volume_fracs.append(volume_hist)
                    pred_trade_volume_fracs.append(pred_frac)
                    true_trade_volume_fracs.append(volume_today)
                    if sum(pred_trade_volume_fracs) > 1:
                        pred_trade_volume_fracs[-1] = pred_trade_volume_fracs[-1] - (
                            sum(pred_trade_volume_fracs) - 1
                        )
                        break
                    if sum(pred_trade_volume_fracs) == 1:
                        break

        market_vwap = llob.get_VWAP(llob_file)
        true_vwaps.append(market_vwap)
        pred_trade_volume_fracs = t.squeeze(t.stack(pred_trade_volume_fracs))

        additional_vwap = 0
        if t.sum(pred_trade_volume_fracs) < 1:
            rest = 1 - t.sum(pred_trade_volume_fracs)
            _, final_price = llob.batch(llob_file, tick.file_date_num + 14_57_00_000)
            additional_vwap = rest * final_price

        trade_price = t.Tensor(trade_price)
        pred_vwap = t.sum(pred_trade_volume_fracs * trade_price) + additional_vwap
        pred_vwaps.append(pred_vwap.item())

        hist_trade_volume_fracs = t.Tensor(hist_trade_volume_fracs)
        hist_trade_volume_fracs = hist_trade_volume_fracs / t.sum(
            hist_trade_volume_fracs
        )

        true_trade_volume_fracs = t.Tensor(true_trade_volume_fracs)
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
        # loss_global.append(loss.item())
        log_eval(file=file.stem, loss=loss.item())
    # plotter(loss_global, ylabel="loss")
    # plt.figure()
    # plt.plot(pred_vwaps, label="pred")
    # plt.plot(true_vwaps, label="true")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(description="Arguments for the strategy", add_help=True)
    parser.add_argument("-s", "--seed", type=int, default=2333, help="set random seed")
    parser.add_argument("-e", "--epoch", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="./DATA/ML")
    parser.add_argument("--model-save", type=str, default="./MODEL_SAVE_1")
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

    newest_model = model_save_path / "500.ocet"
    # para_dict = t.load(newest_model, map_location=device)
    para_dict = t.load(newest_model, map_location=t.device("cpu"))
    ocet.load_state_dict(para_dict["model_state_dict"])
    # ocet.to(device='cpu')
    # with t.no_grad():
    #     ocet.eval()

    # optimizer = optim.Adam(ocet.parameters(), lr=0.0001, weight_decay=0.0005)

    """
    Scripts begin
    """
    evaluate(ocet)
