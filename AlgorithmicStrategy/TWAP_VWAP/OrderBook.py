from abc import ABCMeta, abstractmethod
from pathlib import Path
import pandas as pd



class DataStream(metaclass=ABCMeta):
    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __iter__(self):
        return self


class TickStream(DataStream):
    def __init__(self, data_folder: str):
        self.data_folder: Path = Path(data_folder)
        self.current: int = 0

    def __iter__(self):
        return self

    def __next__(self):
        pass




if __name__ == "__main__":
    tick = TickStream("./DATA/TICK_DATA")

