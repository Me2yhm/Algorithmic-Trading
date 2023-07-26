from datetime import datetime, date
from collections import OrderedDict
from pathlib import Path
from os import PathLike
import csv
import numpy as np
import pandas as pd
from typing import TypedDict, Union

TimeType = Union[datetime, date, pd.Timestamp]
PathType = Union[str, Path, PathLike]


class DataStream:
    __slots__ = "current_file", "current_reader", "delimiter", "data_folder", "data_files"

    def __init__(self, data_folder: PathType, delimiter: str = ","):
        self.current_file = None
        self.current_reader = None
        self.delimiter: str = delimiter
        self.data_folder: Path = Path(data_folder)
        self._refresh_data_files()

    def _open_next_file(self):
        if self.current_file is not None:
            self.current_file.close()

        if self.data_files:
            file_path = self.data_files.pop(0)
            self.current_file = open(file_path, 'r', newline='')
            self.current_reader = csv.reader(self.current_file, delimiter=self.delimiter)
        else:
            self.current_file = None
            self.current_reader = None

    def _refresh_data_files(self):
        self.data_files: list[Path] = list(self.data_folder.glob("*.csv"))

    def __next__(self):
        if self.data_files and self.current_file is None:
            self._open_next_file()
        while self.current_reader is not None:
            try:
                row = next(self.current_reader)
                return row
            except StopIteration:
                self._open_next_file()

        raise StopIteration

    def __iter__(self):
        return self

    def close(self):
        if self.current_file is not None:
            self.current_file.close()
            self.current_file = None
            self.current_reader = None

    def __del__(self):
        self.close()


class TickStream(DataStream):
    def __init__(self, data_folder: PathType, delimiter: str = ","):
        super().__init__(data_folder, delimiter)


class SnapShot(TypedDict):
    timestamp: TimeType
    A: OrderedDict[float, tuple[float, float]]
    S: OrderedDict[float, tuple[float, float]]


if __name__ == "__main__":
    tick = TickStream("./DATA/TICK_DATA")
    print(tick.data_files)
    for tk_dt in tick:
        print(tk_dt)
        break
    ss = SnapShot(timestamp=datetime.now(), A=OrderedDict(), S=OrderedDict())
