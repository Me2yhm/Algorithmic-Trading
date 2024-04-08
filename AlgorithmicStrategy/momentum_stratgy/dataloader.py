import os
from typing import Iterable, Literal
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE

import sys

import talib as TA


def cal_zscore(pcg: list):
    pcg = np.where(np.isinf(pcg), 0, pcg)
    zscores = [
        (pcg[i] - np.mean(pcg[1:])) / np.std(pcg[1:]) for i in range(1, len(pcg))
    ]
    zscores.insert(0, 0)
    return zscores


def add_indi(data: pd.DataFrame, inids: list[str]) -> pd.DataFrame:
    for i in range(1, len(data.columns)):
        col = data.columns[i]
        for ind in inids:
            ind = ind.upper()
            new_col = col + ind
            if ind[:2] == "MA":
                per = int(ind[2:]) if len(ind) > 2 else None
                data[new_col] = getattr(TA, ind[:2])(
                    data.iloc[:, i].values, periods=per
                )
                continue
            data[new_col] = getattr(TA, ind)(data.iloc[:, i].values)
    data.dropna(inplace=True)
    return data


data_path = r"E:\workspace\Algorithmic-Trading\AlgorithmicStrategy\datas\000001.SZ\snapshot\gtja\2023-07-03.csv"
data = pd.read_csv(data_path)
data = data[data["time"] >= 93000000].reset_index(drop=True)
data.iloc[:, 1:] = data.iloc[:, 1:].astype(float)
ind_data = add_indi(data, ["sma", "rsi"])
print(ind_data.head(5))
