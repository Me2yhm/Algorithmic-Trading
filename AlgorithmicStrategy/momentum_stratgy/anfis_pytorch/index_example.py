#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, DataLoader

from .membership import BellMembFunc, make_bell_mfs, make_anfis
from .experimental import train_anfis, test_anfis


def make_train_data(
    file_path: str,
    batch_size=1024,
) -> DataLoader:
    data = pd.read_csv(file_path, header=None)
    *x, y = [list(data.values[:, i]) for i in range(3)]
    x = torch.tensor(x).transpose(-1, 0).double()
    y = torch.tensor(y).transpose(-1, 0).double()
    td = TensorDataset(x, y)
    return DataLoader(td, batch_size=batch_size, shuffle=True)


def make_model(data, num_mfs, num_out):
    model = make_anfis(data, num_mfs, num_out)
    return model


if __name__ == "__main__":
    # file_path = r"AlgorithmicStrategy\momentum_stratgy\anfis-pytorch\index_marked.csv"
    # file_path = r"E:\workspace\Algorithmic-Trading\AlgorithmicStrategy\momentum_stratgy\factor_result.csv"
    # data = make_train_data(file_path=file_path)
    # x, _ = data.dataset.tensors
    x = torch.stack([torch.linspace(67, 67, 80), torch.linspace(23, 37, 80)], dim=1)
    y = torch.rand(80)
    td = TensorDataset(x, y)
    data = DataLoader(td, batch_size=1024, shuffle=True)
    model = make_model(x, num_mfs=2, num_out=1)
    train_anfis(model=model, data=data)
    test_anfis(model=model, data=data, show_plots=True)
