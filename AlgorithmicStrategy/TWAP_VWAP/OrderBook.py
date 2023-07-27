from datetime import datetime, date
from collections import OrderedDict
from pathlib import Path
from os import PathLike
import csv
import numpy as np
import pandas as pd
from typing import TypedDict, Union, TextIO, Iterator

TimeType = Union[datetime, date, pd.Timestamp]
PathType = Union[str, Path, PathLike]


class DataStream:
    __slots__ = "current_file", "current_reader", "delimiter", "data_folder", \
        "data_files", "columns", "rest_data_files", "date_columns", "ticker_column"

    # noinspection PyTypeChecker
    def __init__(self, data_folder: PathType, delimiter: str = ",", **kwargs):
        self.ticker_column: str = kwargs.get("ticker_columns", "SecurityID")
        self.date_columns: tuple = ("TransactTime", "TradeTime")
        self.current_file: TextIO = None
        self.current_reader: Iterator[list[str]] = None
        self.delimiter: str = delimiter
        self.data_folder: Path = Path(data_folder)
        self.data_files: list[Path] = list(self.data_folder.glob("*.csv"))
        self.rest_data_files: list[Path] = self.data_files.copy()
        self.columns: list[str] = None
        self._open_next_file()

    def _open_next_file(self):
        if self.current_file is not None:
            self.current_file.close()

        if self.rest_data_files:
            file_path = self.rest_data_files.pop(0)
            self.current_file = open(file_path, 'r', newline='')
            self.current_reader = csv.reader(
                self.current_file, delimiter=self.delimiter)
            if self.columns is None:
                self.columns = next(self.current_reader)
            else:
                assert self.columns == next(self.current_reader)
        else:
            self.current_file = None
            self.current_reader = None

    def _refresh_data_files(self):
        self.rest_data_files = self.data_files.copy()

    @staticmethod
    def isfloat(txt):
        s = txt.split('.')
        if len(s) > 2:
            return False
        else:
            for si in s:
                if not si.isdigit():
                    return False
            return True

    def _format(self, data: list[str]) -> dict[str, object]:
        assert len(data) == len(self.columns)
        res = []
        for i, j in zip(self.columns, data):
            if i in self.date_columns:
                res.append(datetime.strptime(j, "%Y%m%d%H%M%S%f"))
            elif i == self.ticker_column:
                res.append(j)
            elif j.isdigit():
                res.append(int(j))
            elif self.isfloat(j):
                res.append(round(float(j), 3))
            else:
                res.append(j)
        return dict(zip(self.columns, res))

    def __next__(self):
        if self.rest_data_files and self.current_file is None:
            self._open_next_file()
        while self.current_reader is not None:
            try:
                row = next(self.current_reader)
                return self._format(row)
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


class SnapShot(TypedDict):
    timestamp: TimeType
    A: OrderedDict[float, tuple[float, float]]
    S: OrderedDict[float, tuple[float, float]]


class OrderBook:

    def __init__(self, tick_api: DataStream, order_api: DataStream):
        self.snapshots: OrderedDict[TimeType, SnapShot] = OrderedDict()
        self.last_snapshot = None
        self.tick: DataStream = tick_api
        self.order: DataStream = order_api


if __name__ == "__main__":
    tick = DataStream("./DATA/TICK_DATA")
    print(next(tick))
    order = DataStream("./DATA/ORDER_DATA")
    print(next(order))
