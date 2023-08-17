import csv
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TextIO, Iterator
from .Schema import TimeType, PathType, OrderTick


class DataBase:
    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        pass

    @staticmethod
    def isfloat(txt):
        s = txt.split(".")
        if len(s) > 2:
            return False
        else:
            for si in s:
                if not si.isdigit():
                    return False
            return True

    def fresh(self, num: int = 1) -> list[dict]:
        res = []
        for i in range(num):
            res.append(next(self))
        return res


class DataStream(DataBase):
    __slots__ = (
        "current_file",
        "current_reader",
        "delimiter",
        "data_folder",
        "data_files",
        "columns",
        "rest_data_files",
        "date_column",
        "ticker",
        "file_date",
        "file_date_num",
        "data_cache"
    )

    # noinspection PyTypeChecker
    def __init__(
            self, data_folder: PathType, ticker: str, delimiter: str = ",", **kwargs
    ):
        assert ticker.endswith(".SH") or ticker.endswith(
            ".SZ"
        ), f"ticker {ticker} is not right"
        self.ticker: str = ticker
        self.date_column: str = kwargs.get("date_column", "time")
        self.file_date: TimeType = None
        self.file_date_num: int = None
        self.current_file: TextIO = None
        self.current_reader: Iterator[list[str]] = None
        self.delimiter: str = delimiter
        self.data_folder: Path = Path(data_folder)
        self.data_files: list[Path] = list(self.data_folder.glob("*.csv"))
        self.rest_data_files: list[Path] = self.data_files.copy()
        self.columns: list[str] = None
        self.data_cache: list[OrderTick] = []
        self._open_next_file()

    def isCALL(self, timestamp: int):
        return (
                timestamp < self.file_date_num + 93000000
                or self.file_date_num + 145700000 < timestamp
        )

    def isTrade(self, timestamp: int):
        return (
            self.file_date_num + 9_15_00_000 <= timestamp <= self.file_date_num + 11_30_00_000
        ) or (self.file_date_num + 13_00_00_000 <= timestamp <= self.file_date + 15_00_00_000)

    def _open_next_file(self):
        if self.current_file is not None:
            self.current_file.close()

        if self.rest_data_files:
            file_path = self.rest_data_files.pop(0)
            self.file_date = datetime.strptime(file_path.stem, "%Y-%m-%d")
            self.file_date_num = int(self.file_date.strftime("%Y%m%d%H%M%S")) * 1000

            self.current_file = open(file_path, "r", newline="")
            self.current_reader = csv.reader(
                self.current_file, delimiter=self.delimiter
            )
            if self.columns is None:
                self.columns = next(self.current_reader)
            else:
                assert self.columns == next(self.current_reader)
        else:
            self.current_file = None
            self.current_reader = None

    def _refresh_data_files(self):
        self.rest_data_files = self.data_files.copy()

    def _format(self, data: list[str]) -> OrderTick:
        assert len(data) == len(self.columns)
        res = OrderTick()
        for i, j in zip(self.columns, data):
            if j.isdigit():
                if i == self.date_column:
                    j = j.ljust(8, "0")
                    tmp = int(j) + self.file_date_num
                    res["iscall"] = True if self.isCALL(tmp) else False
                else:
                    tmp = int(j)
                res[i] = tmp
            elif self.isfloat(j):
                res[i] = round(float(j), 3)
            else:
                res[i] = j
        return res

    def __next__(self) -> OrderTick:
        if self.rest_data_files and self.current_file is None:
            self._open_next_file()
        while self.current_reader is not None:
            try:
                row = next(self.current_reader)
                return self._format(row)
            except StopIteration:
                self._open_next_file()

        raise StopIteration

    def close(self):
        if self.current_file is not None:
            self.current_file.close()
            self.current_file = None
            self.current_reader = None

    def __del__(self):
        self.close()

    def next_batch(self, until: int = None) -> list[OrderTick]:
        if self.data_cache:
            # 如果缓存有数据，先处理缓存的数据
            data = self.data_cache.pop(0)
        else:
            data = self.fresh()[0]
        timestamp_this = data[self.date_column]

        if until is not None and timestamp_this > until:
            self.data_cache.append(data)
            return []

        res = [data]
        while True:
            try:
                next_data = self.fresh()
            except StopIteration:
                break
            timestamp_next = next_data[0][self.date_column]

            if timestamp_next != timestamp_this:
                self.data_cache.extend(next_data)
                break
            else:
                res.extend(next_data)

        return res


class DataSet(DataBase):
    __slots__ = (
        "current_file",
        "current_reader",
        "delimiter",
        "columns",
        "file_path",
        "date_column",
        "ticker",
        "file_date",
        "file_date_num",
        "data_cache"
    )

    # noinspection PyTypeChecker
    def __init__(
            self, data_path: PathType, ticker: str, delimiter: str = ",", **kwargs
    ):
        self.file_path: PathType = Path(data_path)
        assert ticker.endswith("SH") or ticker.endswith(
            "SZ"
        ), f"ticker {ticker} is not right"
        self.ticker: str = ticker
        self.date_column: str = kwargs.get("date_column", "time")
        self.file_date = datetime.strptime(self.file_path.stem, "%Y-%m-%d")
        self.file_date_num = int(self.file_date.strftime("%Y%m%d%H%M%S")) * 1000

        self.current_file: TextIO = None
        self.current_reader: Iterator[list[str]] = None
        self.delimiter: str = delimiter
        self.columns: list[str] = None
        self.data_cache: list[OrderTick] = []
        self._open_next_file()

    def isCALL(self, timestamp: int):
        return (
                timestamp < self.file_date_num + 93000000
                or self.file_date_num + 145700000 < timestamp
        )

    def isTrade(self, timestamp: int):
        return (
            self.file_date_num + 9_15_00_000 <= timestamp <= self.file_date_num + 11_30_00_000
        ) or (self.file_date_num + 13_00_00_000 <= timestamp <= self.file_date_num + 15_00_00_000)


    def _open_next_file(self):
        if self.current_file is not None:
            self.current_file.close()

        self.current_file = open(self.file_path, "r", newline="")
        self.current_reader = csv.reader(self.current_file, delimiter=self.delimiter)
        if self.columns is None:
            self.columns = next(self.current_reader)
        else:
            assert self.columns == next(self.current_reader)

    def _format(self, data: list[str]) -> OrderTick:
        assert len(data) == len(self.columns)
        res = OrderTick()
        for i, j in zip(self.columns, data):
            if j.isdigit():
                if i == self.date_column:
                    j = j.ljust(8, "0")
                    tmp = int(j) + self.file_date_num
                    res["iscall"] = True if self.isCALL(tmp) else False
                else:
                    tmp = int(j)
                res[i] = tmp
            elif self.isfloat(j):
                res[i] = round(float(j), 3)
            else:
                res[i] = j
        return res

    def __next__(self) -> OrderTick:
        if self.file_path and self.current_file is None:
            self._open_next_file()
        while self.current_reader is not None:
            try:
                row = next(self.current_reader)
                return self._format(row)
            except StopIteration:
                break
        raise StopIteration

    def close(self):
        if self.current_file is not None:
            self.current_file.close()
            self.current_file = None
            self.current_reader = None

    def __del__(self):
        self.close()

    def next_batch(self, until: int = None) -> list[OrderTick]:
        if self.data_cache:
            # 如果缓存有数据，先处理缓存的数据
            data = self.data_cache.pop(0)
        else:
            data = self.fresh()[0]
        timestamp_this = data[self.date_column]

        if until is not None and timestamp_this > until:
            self.data_cache.append(data)
            return []

        res = [data]
        while True:
            try:
                next_data = self.fresh()
            except StopIteration:
                break
            timestamp_next = next_data[0][self.date_column]

            if timestamp_next != timestamp_this:
                self.data_cache.extend(next_data)
                break
            else:
                res.extend(next_data)

        return res