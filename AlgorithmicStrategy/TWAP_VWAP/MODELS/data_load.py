import datetime
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Literal

import numpy as np


class FileFea(NamedTuple):
    code: str
    rollback: int
    date: datetime


def filename_parser(filename: str):
    return datetime.strptime(filename, "%Y-%m-%d")


def get_all_files(datafolder: Path):
    datafiles = list(datafolder.glob("*.csv"))
    datafiles = sorted(datafiles, key=lambda x: filename_parser(x.stem))
    return datafiles


class JoyeLOB:
    def __init__(self, window: int):
        self.datas: dict[Path, dict[str, np.ndarray]] = {}
        self.times: dict[int, int] = {}
        self.window: int = window

    def push(self, file: Path):
        if file not in self.datas:
            data: np.ndarray = np.loadtxt(file, delimiter=",", skiprows=1)
            self.datas[file] = {
                "X": data[:, 1:-2],
                "volume_hist": data[:, -2],
                "volume_today": data[:, -1],
                "timestamp": data[:, 0],
            }

    def pop(self, file: Path):
        del self.datas[file]

    def batch(self, file: Path, timestamp: int):
        idx = self.search_timestamp_idx(file, timestamp)
        if idx < (self.window - 1):
            return None, None, None, None
        else:
            return (
                int(self.datas[file]["timestamp"][idx]),
                self.datas[file]["X"][idx - self.window + 1 : idx + 1, :][
                    np.newaxis, np.newaxis, :, :
                ],
                self.datas[file]["volume_hist"][idx],
                self.datas[file]["volume_today"][idx],
            )

    def search_timestamp_idx(self, file: Path, timestamp: int):
        timestamp_arr = self.datas[file]["timestamp"].copy()
        timestamp_arr[timestamp_arr >= timestamp] = np.inf
        idx = np.argmin(np.abs(timestamp - timestamp_arr))
        # target = int(timestamp_arr[idx])
        return idx

    def __getitem__(self, index):
        return self.datas.get(index, None)

    def __contains__(self, item):
        return item in self.datas


class LittleOB:
    def __init__(self, direction: Literal["ASK", "BUY"]):
        self.direction = direction
        self.datas: dict[Path, dict[str, np.ndarray]] = {}
        self.times: dict[int, int] = {}

    def push(self, file: Path):
        if file not in self.datas:
            data: np.ndarray = np.loadtxt(file, delimiter=",", skiprows=1)
            self.datas[file] = {
                "ASK": data[:, 1],
                "BUY": data[:, 2],
                "VWAP": data[-1, 3],
                "timestamp": data[:, 0],
            }

    def pop(self, file: Path):
        del self.datas[file]

    def batch(self, file: Path, timestamp: int):
        idx = self.search_timestamp_idx(file, timestamp)
        return (
            int(self.datas[file]["timestamp"][idx]),
            self.datas[file][self.direction][idx],
        )

    def get_VWAP(self, file: Path):
        return self.datas[file]["VWAP"]

    def search_timestamp_idx(self, file: Path, timestamp: int):
        timestamp_arr = self.datas[file]["timestamp"].copy()
        timestamp_arr[timestamp_arr >= timestamp] = np.inf
        idx = np.argmin(np.abs(timestamp - timestamp_arr))
        # target = int(timestamp_arr[idx])
        return idx

    def __getitem__(self, index):
        return self.datas.get(index, None)

    def __contains__(self, item):
        return item in self.datas


if __name__ == "__main__":
    train_folder = Path().cwd().parent / "DATA/ML/000157/train"
    train_files = train_folder.glob("*.csv")
    joye_data = JoyeLOB(window=100)
    for file in train_files:
        joye_data.push(file)
        timestamp, X, y = joye_data.batch(file, timestamp=20230704095103000)
        print(timestamp)
        break
