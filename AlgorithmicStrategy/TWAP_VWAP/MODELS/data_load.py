import datetime
from functools import cached_property
from typing import Union

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


def filename_parser(filename: str):
    file_fea = filename.split('_')
    code = file_fea[0]
    rollback = int(file_fea[1].strip('s'))
    date = datetime.today().year * 100_00 + int(file_fea[2])
    date = datetime.strptime(str(date), '%Y%m%d')
    return FileFea(code=code, rollback=rollback, date=date)


def get_all_files(datafolder: Path):
    datafiles = list(datafolder.glob("*.csv"))
    datafiles = sorted(datafiles, key=lambda x: filename_parser(x.stem).date)
    return datafiles


class JoyeLOB(Dataset):
    def __init__(self, filepath: Union[str, Path], window: int):
        self.raw = np.loadtxt(filepath, skiprows=1, delimiter=",")
        self.window = window

    @cached_property
    def length(self):
        return len(self.raw)

    @cached_property
    def max_index(self):
        return self.length - self.window

    def __len__(self):
        return self.length - self.window + 1

    def __getitem__(self, index):
        lob_data = self.raw[index: index + self.window]
        return lob_data


datafolder = Path(__file__).parents[1] / "DATA" / "ML"
datafiles = get_all_files(datafolder)
lob = JoyeLOB(datafiles[1], window=100)
print(lob[lob.max_index].shape)
