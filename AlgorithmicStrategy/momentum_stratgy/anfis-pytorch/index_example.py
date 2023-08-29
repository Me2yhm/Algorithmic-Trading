#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, DataLoader

import anfis
from membership import BellMembFunc, make_bell_mfs, make_anfis
from experimental import train_anfis, test_anfis


def make_train_data(
    file_path: str,
    batch_size=1024,
):
    data = pd.read_csv(file_path, header=None)
    *x, y = [list(data.values[:, i] for i in (0, 1, 2))]
    x = torch.tensor(x)
    y = torch.tensor(y)
    td = TensorDataset(x, y)
    return DataLoader(td, batch_size=batch_size, shuffle=True)


def make_model(data, num_mfs, num_out):
    model = make_anfis(data, num_mfs, num_out)
    return model


if __name__ == "__mian__":
    file_path = r"AlgorithmicStrategy\momentum_stratgy\anfis-pytorch\index_example.py"
    data = make_train_data(file_path=file_path)
    x, _ = data.dataset.tensors
    model = make_model(x, num_mfs=2, num_out=1)
    train_anfis(model=model, data=data)
    test_anfis(model=model, data=data, show_plots=True)
