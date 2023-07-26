from __future__ import annotations
from abc import ABCMeta, abstractmethod
import sys
import os

sys.path.insert(0, '../')
from base import AlgorithmicStrategy  # type: ignore


class DataStream(metaclass=ABCMeta):
    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __iter__(self):
        return self


class TickStream(DataStream):
    def __init__(self, path: os.PathLike | str):
        self.path: os.PathLike = path
        self.current: int = 0
        self.limit: int = 10

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.limit:
            raise StopIteration
        result = self.current
        self.current = self.current + 1
        return result


class TradeStrategy(AlgorithmicStrategy):
    def __init__(self):
        pass

    def TWAP_stratgy(self):
        pass

    def VWAP_stratgy(self):
        pass

    def momentum_stratgy(self):
        pass

    def snapshot_stratgy(self):
        pass


if __name__ == "__main__":
    trader = TradeStrategy()
    tick = TickStream("./")
    for i in tick:
        print(i)
