import datetime
from functools import cached_property
from typing import Union

import pandas as pd
import torch as t
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import NamedTuple
from datetime import datetime


class FileFea(NamedTuple):
    code: str
    rollback: int
    date: datetime


# def filename_parser(filename: str):
#     file_fea = filename.split('_')
#     code = file_fea[0]
#     rollback = int(file_fea[1].strip('s'))
#     date = datetime.today().year * 100_00 + int(file_fea[2])
#     date = datetime.strptime(str(date), '%Y%m%d')
#     return FileFea(code=code, rollback=rollback, date=date)

def filename_parser(filename: str):
    return datetime.strptime(filename, '%Y-%m-%d')
def get_all_files(datafolder: Path):
    datafiles = list(datafolder.glob("*.csv"))
    datafiles = sorted(datafiles, key=lambda x: filename_parser(x.stem))
    return datafiles


class JoyeLOB():
    def __init__(self, window: int):

        self.datas: dict[Path, np.ndarray] = {}
        self.data: np.ndarray = None
        self.window: int = window

    def push(self, file:Path):
        self.datas[file] = np.loadtxt(file, delimiter=',', skiprows=1)

    def __getitem__(self, index):
        return self.datas.get(index, None)



if __name__ == '__main__':
    pass