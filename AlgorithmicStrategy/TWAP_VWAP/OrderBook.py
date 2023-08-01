import csv
import json
from collections import OrderedDict
from datetime import datetime, date
from os import PathLike
from pathlib import Path
from typing import TypedDict, Union, TextIO, Iterator, Literal

import numpy as np
import pandas as pd

TimeType = Union[datetime, date, pd.Timestamp, str, int]
PathType = Union[str, Path, PathLike]


class OT(TypedDict, total=False):
    time: int
    oid: int
    oidb: int
    oids: int
    price: float
    volume: int
    flag: int
    ptype: int


class LifeTime(TypedDict, total=False):
    oid: int
    price: float
    volume: int
    rest: int
    birth: int
    death: int
    life: int


class DataStream:
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
    )

    # noinspection PyTypeChecker
    def __init__(self, data_folder: PathType, delimiter: str = ",", **kwargs):
        self.ticker: str = data_folder.parents[1].name
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
        self._open_next_file()

    def isCALL(self, timestamp: int):
        return (
            timestamp < self.file_date_num + 93000000
            or self.file_date_num + 145700000 < timestamp
        )

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

    def _format(self, data: list[str]) -> OT:
        assert len(data) == len(self.columns)
        res = OT()
        for i, j in zip(self.columns, data):
            if j.isdigit():
                tmp = int(j) + self.file_date_num if i == self.date_column else int(j)
                res[i] = tmp
            elif self.isfloat(j):
                res[i] = round(float(j), 3)
            else:
                res[i] = j
        return res

    def __next__(self) -> OT:
        if self.rest_data_files and self.current_file is None:
            self._open_next_file()
        while self.current_reader is not None:
            try:
                row = next(self.current_reader)
                return self._format(row)
            except StopIteration:
                self._open_next_file()

        raise StopIteration

    def fresh(self, num: int = 1) -> Union[list[dict], dict]:
        if num == 1:
            return next(self)
        else:
            res = []
            for i in range(num):
                res.append(next(self))
            return res

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
    ask: OrderedDict[float, float]
    bid: OrderedDict[float, float]


class OrderBook:
    __slots__ = "snapshots", "last_snapshot", "data_api", "oid_map"

    # noinspection PyTypeChecker
    def __init__(self, data_api: DataStream):
        self.snapshots: OrderedDict[TimeType, SnapShot] = OrderedDict()
        self.last_snapshot: SnapShot = None
        self.data_api: DataStream = data_api
        self.oid_map: dict[int, LifeTime] = dict()

    def single_update(self):
        data = self.data_api.fresh()
        if self.last_snapshot is None:
            self.last_snapshot = SnapShot(
                timestamp=data[self.data_api.date_column],
                ask=OrderedDict(),
                bid=OrderedDict(),
            )
        if data["oid"] != 0 or data["ptype"] != 0:
            self._update_from_order(data)
            # if data['flag'] == 1:
            #     self._validate(data, key="oidb")
            #     self._update_from_order(data)
            # elif data['flag'] == 2:
            #     self._validate(data, key="oids")
            #     self._update_from_order(data)
        else:
            self._update_from_tick(data)

        return data["time"]

    def update(self, until: int = None):
        while True:
            timestamp_this = self.single_update()
            if until is not None and timestamp_this > until:
                break

    @staticmethod
    def _order_change(
        snap: SnapShot, AS: Literal["ask", "bid"], direction: Literal[1, -1], data: OT
    ):
        snap[AS][data["price"]] = (
            snap[AS].get(data["price"], 0) + direction * data["volume"]
        )
        if snap[AS][data["price"]] == 0:
            del snap[AS][data["price"]]

        snap[AS] = OrderedDict(
            sorted(snap[AS].items(), key=lambda x: x[0], reverse=(AS == "bid"))
        )

    def _tick_change(self, snap: SnapShot, data: OT):
        # if not self.data_api.isCALL(data["time"]):
        #     self._order_change(snap, "ask", -1, data)
        #     self._order_change(snap, "bid", -1, data)
        # else:
        TradePrice = data["price"]
        TradeQty = data["volume"]
        A_p_v = list(snap["ask"].items())
        B_p_v = list(snap["bid"].items())

        rest = TradeQty
        for p, v in B_p_v:
            if p >= TradePrice:
                if v >= rest:
                    v -= rest
                    if v == 0:
                        del snap["bid"][p]
                    else:
                        snap["bid"][p] = v
                else:
                    rest -= v
                    del snap["bid"][p]
            else:
                break

        rest = TradeQty
        for p, v in A_p_v:
            if p <= TradePrice:
                if v >= rest:
                    v -= rest
                    if v == 0:
                        del snap["ask"][p]
                    else:
                        snap["ask"][p] = v
                else:
                    rest -= v
                    del snap["ask"][p]
            else:
                break

    def _update_from_order(self, data: OT):
        snap: SnapShot = self.last_snapshot.copy()
        snap["timestamp"]: int = data[self.data_api.date_column]

        if self.data_api.ticker.endswith("SZ"):
            if data["oid"] in self.oid_map and self.oid_map[data["oid"]]["price"] == 0:
                death = self.oid_map[data["oid"]]["death"]
                birth = self.oid_map[data["oid"]].get("birth", death)
                if data["time"] <= birth:
                    birth = data["time"]
                self.oid_map[data["oid"]]["life"] = death - birth
                self.oid_map[data["oid"]]["price"] = data["price"]
                return

            assert data["price"] != 0.0, data

            AS: str = "bid" if data["flag"] == 1 else "ask"
            self.oid_map[data["oid"]] = LifeTime(
                oid=data["oid"],
                price=data["price"],
                volume=data["volume"],
                birth=data["time"],
            )
            self._order_change(snap, AS, 1, data)
        elif self.data_api.ticker.endswith("SH"):
            pass
        else:
            raise NotImplementedError

        self.last_snapshot = snap
        self.snapshots[data[self.data_api.date_column]] = self.last_snapshot.copy()

    def _validate(self, data: OT, key: str):
        try:
            data["price"] = self.oid_map[data[key]]["price"]
            return True
        except KeyError:
            self.oid_map[data[key]] = LifeTime(
                oid=data[key],
                death=data["time"],
                volume=data["volume"],
                price=data["price"],
                rest=0,
            )
            return False

    def _update_from_tick(self, data: dict):
        snap: SnapShot = self.last_snapshot.copy()
        snap["timestamp"]: int = data[self.data_api.date_column]
        if self.data_api.ticker.endswith("SZ"):
            if data["flag"] == 1:
                if not self._validate(data, key="oidb"):
                    return
                self._tick_change(snap, data)
            elif data["flag"] == 2:
                if not self._validate(data, key="oids"):
                    return
                self._tick_change(snap, data)
            elif data["flag"] == 3:
                if not self._validate(data, key="oidb"):
                    return
                self._order_change(snap, "bid", -1, data)
            elif data["flag"] == 4:
                if not self._validate(data, key="oids"):
                    return
                self._order_change(snap, "ask", -1, data)

        elif self.data_api.ticker.endswith("SH"):
            pass
        else:
            raise NotImplementedError

        self.last_snapshot = snap
        self.snapshots[data[self.data_api.date_column]] = self.last_snapshot.copy()

    def search_closet_time(self, query_stamp: int):
        logged_timestamp: np.ndarray = np.array(list(self.snapshots.keys()))
        search_timestamp = logged_timestamp[logged_timestamp <= query_stamp]
        time_difference = np.abs(search_timestamp - query_stamp)
        closest_index = np.argmin(time_difference)
        closest_time = search_timestamp[closest_index]
        return closest_time

    @staticmethod
    def print_json(dict_like: dict):
        print(json.dumps(dict_like, indent=2))

    def search_snapshot(self, query_stamp: int):
        closest_time = self.search_closet_time(query_stamp)
        return self.snapshots[closest_time]


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    data_dir = Path(__file__).parent / "../datas/000001.SZ/tick/gta"
    tick = DataStream(data_dir, date_column="time")
    print(tick.rest_data_files)
    # print(tick.columns)
    # print(tick.fresh())
    # print(tick.ticker)
    # print(tick.file_date)
    ob = OrderBook(data_api=tick)
    timestamp = 20230508093100000
    ob.update(until=timestamp)
    near = ob.search_snapshot(timestamp)
    print(near["timestamp"])
    print(ob.print_json(near["bid"]))

    # lt = LifeTime(oid=122)
    # print(lt["oid"])
