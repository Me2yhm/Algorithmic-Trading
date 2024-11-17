from datetime import datetime, timedelta
from pathlib import Path

from chinese_calendar import is_holiday
import skorch
import torch
import pickle


def is_trading_day(today: str):
    weekday = datetime.strptime(today, "%Y%m%d").weekday()
    today = datetime.strptime(today, "%Y%m%d").date()
    if weekday >= 5:
        return False
    if is_holiday(today):
        return False
    return True


def roll_date(date: str, trend=-1):
    date_format = "%Y%m%d"
    old_date = datetime.strptime(date, date_format)
    new_date = old_date + timedelta(days=trend)
    while not is_trading_day(new_date.strftime(date_format)):
        new_date = new_date + timedelta(days=trend)
        return new_date.strftime(date_format)


def get_net(model, epoch=700, lr=5e-3):
    net = skorch.NeuralNet(
        model,
        max_epochs=epoch,
        train_split=None,
        criterion=torch.nn.MSELoss,
        # criterion__reduction="sum",
        optimizer=torch.optim.SGD,
        optimizer__lr=lr,
        optimizer__momentum=0.99,
    )
    return net


def get_success_list(code: str):
    succ_file = Path(__file__).parent.parent.parent / f"cache/{code}/success.pkl"
    if succ_file.exists():
        with open(succ_file, "rb") as f:
            success_list = pickle.load(f)
    else:
        success_list = []
        if not succ_file.parent.exists():
            succ_file.parent.mkdir(parents=True)
    return success_list


def save_success_list(code: str, success_list: list):
    succ_file = Path(__file__).parent.parent.parent / f"cache/{code}/success.pkl"
    with open(succ_file, "wb") as f:
        pickle.dump(success_list, f)


def get_fail_list(code: str):
    fail_file = Path(__file__).parent.parent.parent / f"cache/{code}/fail.pkl"
    if fail_file.exists():
        with open(fail_file, "rb") as f:
            fail_list = pickle.load(f)
    else:
        fail_list = []
        if not fail_file.parent.exists():
            fail_file.parent.mkdir(parents=True)
    return fail_list


def save_fail_list(code: str, fail_list: list):
    fail_file = Path(__file__).parent.parent.parent / f"cache/{code}/fail.pkl"
    with open(fail_file, "wb") as f:
        pickle.dump(fail_list, f)


if __name__ == "__main__":
    get_success_list("601155.SH")
