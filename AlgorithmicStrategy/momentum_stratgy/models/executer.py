from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import os
import sys
from collections import defaultdict, deque
from multiprocessing import Lock
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skorch import NeuralNet
import tushare as ts
from loguru import logger

sys.path.append(os.path.abspath(Path(__file__).parent.parent.parent.parent))
from AlgorithmicStrategy.momentum_stratgy.anfis_pytorch.experimental import plot_all_mfs
from AlgorithmicStrategy.momentum_stratgy.dataloader import (
    add_indi,
    data_to_zscore,
    get_price_dat,
    get_volume_dat,
    train_loader,
)
from AlgorithmicStrategy.momentum_stratgy.models.cnn_anfis import (
    make_model,
    train_model,
)
from AlgorithmicStrategy.momentum_stratgy.models.schema import (
    account,
    get_line,
    is_full,
)
from AlgorithmicStrategy.momentum_stratgy.models.utils import (
    get_fail_list,
    get_net,
    get_success_list,
    roll_date,
    save_fail_list,
    save_success_list,
)

lock = Lock()


def parameter_init(
    para_code: str,
    para_date_train: str,
    para_date_test: str,
    para_indicators: list[str],
):
    global code, date_train, date_test, indicators
    code = para_code
    date_train = para_date_train
    date_test = para_date_test
    indicators = para_indicators


def get_model():
    logger.info(f"Start to get train model for {code} on {date_train}.")
    global net, X, y, model
    data_path = Path(__file__) / f"../../../datas/{code}/snapshot/gtja/{date_train}.csv"
    model_path = Path(__file__) / f"../../../cache/{code}/{date_train}_model.pth"
    with lock:
        data = pd.read_csv(data_path)
    data = data[data["time"] >= 93000000].reset_index(drop=True)
    data.iloc[:, 1:] = data.iloc[:, 1:].astype(float)
    datalader = train_loader(data, 60, -1, 64)
    X, y = datalader.dataset.tensors
    model = make_model(44, 60)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path))
        net = get_net(model, 400, 1e-7)
    else:
        logger.info("No model found, start to train.")
        net = train_model(datalader, model, 500, 1e-3)
        net = train_model(datalader, model, 400, 1e-7)
        if not model_path.parent.exists():
            model_path.parent.mkdir(parents=True)
        torch.save(model.state_dict(), model_path)


def get_test_model():
    logger.info(f"Start to get test data for {code} on {date_test}.")
    global zscore_pri_ind, zscore_vol_ind, vwap, data
    data_path = Path(__file__) / f"../../../datas/{code}/snapshot/gtja/{date_test}.csv"
    with lock:
        data = pd.read_csv(data_path)
    data = data[data["time"] >= 93000000].reset_index(drop=True)
    data.iloc[:, 1:] = data.iloc[:, 1:].astype(float)
    vwap = data["money"].sum() / data["volume"].sum()
    # prepare pricdat and voldat to make dataset
    pric_dat = get_price_dat(data)
    volum_dat = get_volume_dat(data)
    pric_ind = add_indi(pric_dat, indicators)
    volum_ind = add_indi(volum_dat, indicators)
    zscore_pri_ind = data_to_zscore(pric_ind, ["RSI"])
    zscore_vol_ind = data_to_zscore(volum_ind)


def get_last_date():
    global last_date
    last_date = roll_date(date_test.replace("-", ""))


def make_dat(deq_pri: deque, deq_vol: deque):
    pri = torch.tensor(deq_pri)
    vol = torch.tensor(deq_vol)
    dat = torch.stack([pri, vol], dim=0).unsqueeze(0)
    return dat


def get_market_price():
    global market_price, last_price
    pro = ts.pro_api()
    df = pro.daily(
        ts_code=code,
        start_date=date_test.replace("-", ""),
        end_date=date_test.replace("-", ""),
    )
    df2 = pro.daily(
        ts_code=code,
        start_date=last_date,
        end_date=last_date,
    )
    market_price = round(((df["high"] + df["low"] + df["close"]) / 3).values[0], 2)
    last_price = df2["close"].values[0]


def execute_cpama():
    logger.info(f"Start to execute CPAMA for {code} on {date_test}.")
    global acc
    acc = account(vwap=vwap)
    signal = []
    # deque use to contain data
    pri_deq = deque(maxlen=60)
    vol_deq = deque(maxlen=60)
    adn = len(data) - len(zscore_pri_ind)
    for i in range(len(data)):
        if i < adn:
            continue
        if i == len(data) - 2:
            break
        snap = data.iloc[i, :]
        price_line = get_line(zscore_pri_ind, i - adn)
        volume_line = get_line(zscore_vol_ind, i - adn)
        pri_deq.append(price_line)
        vol_deq.append(volume_line)
        if not is_full(pri_deq):
            continue
        if acc.has_signal:
            acc.execute_signal(signal, snap)
        if (i - 1) % 20 == 0:
            dat = make_dat(pri_deq, vol_deq)
            pred = model(dat).squeeze()
            # print(pred)
            signal = acc.generate_sig(pred.tolist(), snap)


def execute_twap():
    logger.info(f"Start to execute TWAP for {code} on {date_test}.")
    global acc_twap
    acc_twap = account(vwap=vwap)
    signal = []
    adn = len(data) - len(zscore_pri_ind)
    for i in range(len(data)):
        snap = data.iloc[i, :]
        if (i - 1) % 20 == 0 and acc_twap.remain_volume > 0:
            signal = [snap["ask1_price"], acc_twap.unit]
            acc_twap.has_signal = True
        if acc_twap.has_signal:
            acc_twap.execute_signal(signal, snap)


def execute_sl():
    logger.info(f"Start to execute S&L for {code} on {date_test}.")
    global acc_sl
    acc_sl = account(vwap=vwap)
    signal = []
    price = last_price
    for i in range(len(data)):
        snap = data.iloc[i, :]
        if snap["ask1_price"] <= price:
            acc_sl.has_signal = True if acc_sl.remain_volume > 0 else False
            signal = [snap["ask1_price"], snap["ask1_volume"]]
        if acc_sl.has_signal:
            acc_sl.execute_signal(signal, snap)
    if acc_sl.remain_volume > 0:
        for i in range(-1, -len(data), -1):
            snap = data.iloc[i, :]
            if acc_sl.remain_volume > 0:
                signal = [snap["ask1_price"], snap["ask1_volume"]]
                acc_sl.execute_signal(signal, snap)
            else:
                break


def write_result():
    logger.success(
        f"Algorithm has executed, starting to write result for {code} on {date_test}."
    )
    file = Path(__file__).parent / "./algorithom_result.csv"
    vwap_sig = round(vwap, 2)
    row1 = [
        code,
        date_test,
        "CPAMA",
        vwap_sig,
        market_price,
        acc.cal_bp(),
        acc.cal_win_rate(vwap),
    ]
    row2 = [
        code,
        date_test,
        "S&L",
        vwap_sig,
        market_price,
        acc_sl.cal_bp(),
        acc_sl.cal_win_rate(vwap),
    ]
    row3 = [
        code,
        date_test,
        "TWAP",
        vwap_sig,
        market_price,
        acc_twap.cal_bp(),
        acc_twap.cal_win_rate(vwap),
    ]
    for i in range(1, 4):
        row = eval(f"row{i}")
        row = [str(v) for v in row]
        exec(f"row{i} = row")
    with lock, open(file, mode="a", newline="") as file:
        writer = csv.writer(file)
        for row in [row1, row2, row3]:
            writer.writerow(row)


def contrast():
    show_atrri = ["avrage_cost", "market_price", "vwap", "remain_volume"]
    cpama = {
        show_atrri[0]: acc.cal_avcost(),
        show_atrri[1]: market_price,
        show_atrri[2]: vwap,
        show_atrri[3]: acc.remain_volume,
    }
    twap = {
        show_atrri[0]: acc_twap.cal_avcost(),
        show_atrri[1]: market_price,
        show_atrri[2]: vwap,
        show_atrri[3]: acc_twap.remain_volume,
    }
    sl = {
        show_atrri[0]: acc_sl.cal_avcost(),
        show_atrri[1]: market_price,
        show_atrri[2]: vwap,
        show_atrri[3]: acc_sl.remain_volume,
    }
    logger.info(f"CPAMA:{cpama}")
    logger.info(f"TWAP:{twap}")
    logger.info(f"S&L:{sl}")


def run(
    para_code: str,
    para_date_train: str,
    para_date_test: str,
    para_indicators: list[str] = ["sma60", "sma120", "rsi"],
):
    parameter_init(para_code, para_date_train, para_date_test, para_indicators)
    get_model()
    get_test_model()
    get_last_date()
    get_market_price()
    execute_cpama()
    execute_twap()
    execute_sl()
    write_result()
    contrast()


def main():
    datas_path = Path(__file__).parent.parent.parent / "datas"
    codes = [dir.name for dir in datas_path.iterdir() if dir.is_dir()]

    futures = {}
    with ProcessPoolExecutor() as executor:
        for code in codes:
            dates = [
                file.stem
                for file in (datas_path / f"{code}/snapshot/gtja").iterdir()
                if file.suffix == ".csv"
            ]
            train_dates = dates[:-1]
            test_dates = dates[1:]
            success_list = get_success_list(code)
            fail_list = get_fail_list(code)
            for date_train, date_test in zip(train_dates, test_dates):
                record = f"{code}-{date_train}-{date_test}"
                if record in success_list or record in fail_list:
                    continue
                future = executor.submit(run, code, date_train, date_test)
                futures[future] = (code, date_train, date_test)
        for future in as_completed(futures):
            try:
                record = f"{code}-{date_train}-{date_test}"
                code, date_train, date_test = futures[future]
                success_list.append(record)
                save_success_list(code, success_list)
                logger.success(f"{record} is done.")
            except Exception as e:
                logger.error(f"[{e.__class__.__name__}] {e}: {record} execute failed.")
                fail_list.append(record)
                save_fail_list(code, fail_list)


if __name__ == "__main__":
    main()
