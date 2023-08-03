from enum import Enum
from typing import TypedDict, Union
from collections import OrderedDict
import pandas as pd
from datetime import datetime, date
from os import PathLike
from pathlib import Path

TimeType = Union[datetime, date, pd.Timestamp, str, int]
PathType = Union[str, Path, PathLike]


class OrderTick(TypedDict, total=False):
    time: int
    oid: int
    oidb: int
    oids: int
    price: float
    volume: int
    flag: int
    ptype: int


class OrderFlag(int, Enum):
    NEUTRAL = 0
    BUY = 1
    SELL = 2
    CANCEL_BUY = 3
    CANCEL_SELL = 4


class LifeTime(TypedDict, total=False):
    oid: int
    price: float
    volume: int
    rest: int
    birth: int
    death: int
    life: int
    AS: str


class SnapShot(TypedDict):
    timestamp: TimeType
    ask: OrderedDict[float, float]
    bid: OrderedDict[float, float]
