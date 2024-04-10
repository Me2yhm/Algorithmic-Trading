import os
from typing import Iterable, Literal
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import RandomOverSampler

import sys

import talib as TA


import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

indicators = ["sma60", "sma120", "rsi"]


def tag_zs(zs: float) -> list:
    if zs >= 1.5:
        return 2
    elif 0.5 <= zs < 1.5:
        return 1
    elif -0.3 < zs < 0.3:
        return 0
    elif -1.5 < zs <= -0.3:
        return -1
    else:
        return -2


def make_seqs(seq_len: int, data: Iterable) -> torch.Tensor:
    num_samp = len(data)
    return torch.stack([data[i : i + seq_len] for i in range(num_samp - seq_len + 1)])


def cal_zscore(pcg: list):
    pcg = np.where(np.isinf(pcg), 0, pcg)
    zscores = [
        (pcg[i] - np.mean(pcg[1:])) / np.std(pcg[1:]) for i in range(1, len(pcg))
    ]
    zscores.insert(0, 0)
    return zscores


def add_indi(dat: pd.DataFrame, inids: list[str]) -> pd.DataFrame:
    data = dat.copy()
    for i in range(1, len(data.columns)):
        col = data.columns[i]
        for ind in inids:
            ind = ind.upper()
            new_col = col + "-" + ind
            if ind[:3] == "SMA":
                per = int(ind[3:]) if len(ind) > 3 else 30
                data[new_col] = getattr(TA, ind[:3])(
                    data.iloc[:, i].values, timeperiod=per
                )
                continue
            data[new_col] = getattr(TA, ind)(data.iloc[:, i].values)
    data.dropna(inplace=True)
    return data.reset_index(drop=True)


def get_price_dat(data: pd.DataFrame) -> pd.DataFrame:
    return data.iloc[:, [*range(9), *range(14, 19)]].drop(columns=["volume", "money"])


def get_volume_dat(data: pd.DataFrame) -> pd.DataFrame:
    dat = data.drop(columns=["last_price", "money"]).iloc[
        :, [0, 1, *range(7, 12), *range(17, len(data.columns) - 2)]
    ]
    dat.loc[1:, "volume"] = dat["volume"].diff()[1:].values
    return dat


def data_to_zscore(dat: pd.DataFrame, nodiff: list[str] | str = "all") -> pd.DataFrame:
    data = dat.copy()
    if nodiff == "all":
        for col in data.columns[1:]:
            data.loc[:, col] = cal_zscore(data.loc[:, col].values)
    else:
        for col in data.columns[1:]:
            if col.split("-")[-1] in nodiff:
                data.loc[:, col] = cal_zscore(data.loc[:, col].values)
                continue
            data.loc[:, col] = cal_zscore(data.loc[:, col].diff().values)
    return data.iloc[1:, 1:].reset_index(drop=True)


def target_data(data: pd.Series, period: int = 20) -> list:
    target = data.diff(periods=period).dropna()
    target = cal_zscore(target.values)
    return target


def mark_target(target: list):
    return list(map(tag_zs, target))


def get_target(dat: pd.Series, period: int = 20) -> list:
    tag_dat = target_data(dat, period)
    return mark_target(tag_dat)


def make_seq_dataset(data: pd.DataFrame, seq_len: int) -> TensorDataset:
    x_dat = torch.tensor(data.iloc[:, :-1].to_numpy(), dtype=torch.float32)
    y_dat = torch.tensor(data.iloc[:, -1], dtype=torch.float32)
    x = make_seqs(seq_len, x_dat)
    y = make_seqs(seq_len, y_dat)[:, -1]
    dataset = TensorDataset(x, y)
    return dataset


def split_train_dataset(dataset: TensorDataset, split_index: int) -> TensorDataset:
    train_dataset = dataset[:split_index]
    x, y = train_dataset
    channels = x.size(1)
    seq_len = x.size(2)
    input_dim = x.size(3)
    dat = x.view(x.size(0), -1)
    dat = torch.cat([dat, y], dim=1).numpy()
    y1 = y[:, 0].numpy()
    ros = RandomOverSampler()
    dat, y1 = ros.fit_resample(dat, y1)
    y2 = dat[:, -1]
    dat, y2 = ros.fit_resample(dat, y2)
    x = dat[:, :-2]
    y = dat[:, -2:]
    x = torch.tensor(x, dtype=torch.float32).view(-1, channels, seq_len, input_dim)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 2)
    train_dataset = TensorDataset(x, y)
    return train_dataset


def split_test_dataset(dataset: TensorDataset, split_index: int) -> TensorDataset:
    test_dataset = dataset[split_index:]
    test_dataset = TensorDataset(*test_dataset)
    return test_dataset


def make_dataloader(
    dataset: TensorDataset, batch_size: int, shuffle: bool = True
) -> DataLoader:
    dataloader = DataLoader(dataset, batch_size, shuffle)
    return dataloader


def price_dataset(data: pd.DataFrame, seq_len: int) -> TensorDataset:
    pri_dat = get_price_dat(data)
    pri_ind = add_indi(pri_dat, indicators)
    zscore_pri_ind = data_to_zscore(pri_ind, ["RSI"])
    mid_price = (pri_ind["ask1_price"] + pri_ind["bid1_price"]) / 2
    tag_price = get_target(mid_price[1:])
    price_target = pd.DataFrame({"price_target": tag_price})
    marked_price = pd.concat(
        [zscore_pri_ind.iloc[: len(tag_price)], price_target],
        axis=1,
        join="outer",
    )
    pri_dataset = make_seq_dataset(marked_price, seq_len)
    return pri_dataset


def volume_dataset(data: pd.DataFrame, seq_len: int) -> TensorDataset:
    vol_dat = get_volume_dat(data)
    vol_ind = add_indi(vol_dat, indicators)
    zscore_vol_ind = data_to_zscore(vol_ind)
    mid_volume = (vol_ind["ask1_volume"] + vol_ind["bid1_volume"]) / 2
    tag_volume = get_target(mid_volume[1:])
    volume_target = pd.DataFrame({"volume_target": tag_volume})
    marked_volume = pd.concat(
        [zscore_vol_ind.iloc[: len(tag_volume)], volume_target],
        axis=1,
        join="outer",
    )
    vol_dataset = make_seq_dataset(marked_volume, seq_len)
    return vol_dataset


def get_dataset(data: pd.DataFrame, seq_len: int) -> TensorDataset:
    pri_dataset = price_dataset(data, seq_len)
    vol_dataset = volume_dataset(data, seq_len)
    price_x, price_y = pri_dataset.tensors
    volume_x, volume_y = vol_dataset.tensors
    data_x = torch.stack([price_x, volume_x], dim=1)
    data_y = torch.stack([price_y, volume_y], dim=1)
    dataset = TensorDataset(data_x, data_y)
    return dataset


def train_loader(
    data: pd.DataFrame,
    seq_len: int,
    split_index: int,
    batch_size: int,
    shuffle: bool = True,
):
    dataset = get_dataset(data, seq_len)
    split_dataset = split_train_dataset(dataset, split_index)
    return make_dataloader(split_dataset, batch_size, shuffle)
