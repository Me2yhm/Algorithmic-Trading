import pandas as pd
from collections import OrderedDict
from pathlib import Path


class Normalized_reader:
    def __init__(self, file_folder: Path):
        self.file_folder: Path = file_folder
        self.filenames: list[Path] = list(self.file_folder.glob("*.csv"))
        self.filenames.sort(key=lambda x: int(x.stem.split("_")[-1]))
        self.dfs: dict[str, pd.DataFrame] = {
            k.stem.split("_")[-1]: pd.read_csv(k) for k in self.filenames
        }
        self.inputs: OrderedDict = OrderedDict()
        self.trade_record: OrderedDict = OrderedDict()
        self.timestamp_list: list[int] = []

    def generate_inputs(self, filename):
        self.trade_record = OrderedDict()
        self.inputs = OrderedDict()
        df = self.dfs[filename]
        input_df = df.drop(["trade_price", "timestamp"], axis=1)
        for limit in range(99, len(self.dfs[filename])):
            trade_price = df.loc[limit, "trade_price"]
            trade_time = df.loc[limit, "timestamp"]
            self.trade_record[trade_time] = {
                "trade_volume": None,
                "trade_price": trade_price,
            }
            self.inputs[trade_time] = input_df.loc[limit - 99 : limit, :]

        self.timestamp_list = list(self.inputs.keys())
        return self.inputs, self.trade_record
