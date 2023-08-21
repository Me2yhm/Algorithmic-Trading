import json
from collections import deque
from pathlib import Path
from typing import cast
from tqdm import tqdm
import pandas as pd
import csv

from .OrderBook import OrderBook
from .Writer import Writer



class LimitedQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = deque()

    @property
    def size(self):
        return len(self.queue)

    def push(self, item):
        if self.size >= self.max_size:
            self.queue.popleft()  # 移除最老的元素
        self.queue.append(item)

    @property
    def items(self):
        return list(self.queue)


class Normalizer:
    def __init__(self, file_folder: Path, is_train: bool = True, **kwargs):
        self.file_folder: Path = file_folder
        self.vars = json.loads(open(Path(__file__).parent / "vars.json", "r").read())
        self.filenames: list[Path] = list(self.file_folder.glob("*.csv"))
        self.filenames.sort(key=lambda x: int(x.stem.split("_")[-1]))
        self.df_map = {file:pd.read_csv(file) for file in self.filenames}
        self.df_origin: pd.DataFrame = pd.concat(self.df_map.values(), axis=0, ignore_index=True)
        self.df_normalized: pd.DataFrame = pd.DataFrame()

        self.is_train: bool = is_train
        self.const_var: list[str] = self.vars.get("const_var")
        self.continuous_var: list[str] = self.vars.get("continuous_var")
        self.discrete_var: list[str] = self.vars.get("discrete_var")
        self.total_columns: list[str] = self.vars.get("total_columns")
        self.dis_norm: list[int] = kwargs.get("dis_norm", [30, 120, 480])
        self.cont_norm = None

        self.df_list: list[pd.DataFrame] = list()
        self.hist_feature: pd.DataFrame = pd.DataFrame()
        self.df_total: pd.DataFrame = pd.DataFrame()
        self.index: int = 0

    def generate_hist_feature(self, filenames: list[Path]):
        hist_feature = cast(pd.DataFrame, 0)
        for file in filenames:
            df = self.df_map[file]
            hist_feature += df.loc[:, ["VWAP", "volume_range"]]
        
        volume_sum = hist_feature['volume_range'].sum()
        df_volume_range_new = hist_feature['volume_range'].copy(deep=True)
        df_volume_percentage = df_volume_range_new/volume_sum
        df_volume_percentage = df_volume_percentage.rename('volume_percentage')
        hist_feature = hist_feature.rename(
            columns={"VWAP": "VWAP_hist", "volume_range": "volume_range_hist"}
        )
        hist_feature /= len(self.filenames)
        df_VWAP_original = hist_feature["VWAP_hist"].copy(deep=True)
        df_VWAP_original = df_VWAP_original.rename("VWAP_hist_original")
        hist_feature = (
                               hist_feature - hist_feature.mean()
                       ) / hist_feature.std()
        self.hist_feature = pd.concat([hist_feature, df_VWAP_original,df_volume_percentage], axis=1)

    def get_past_files(self, file: Path, limit: int = 5):
        idx = self.filenames.index(file)
        if limit is not None:
            if idx < limit:
                return self.filenames[:idx]
            else:
                return self.filenames[idx - limit: idx]
        else:
            return self.filenames[:idx]

    def get_continuous_params(self):
        param_mean = self.df_origin[self.continuous_var].mean()
        param_std = self.df_origin[self.continuous_var].std()
        self.cont_norm = [param_mean, param_std]

    def to_continuous_normalize(self, df):
        return (df[self.continuous_var] - self.cont_norm[0]) / self.cont_norm[1]

    def get_quantile(self, quantile: float):
        result = (
                         self.df_origin["ask_order_stale_0"].quantile(quantile)
                         + self.df_origin["ask_order_stale_1"].quantile(quantile)
                         + self.df_origin["ask_order_stale_2"].quantile(quantile)
                         + self.df_origin["bid_order_stale_0"].quantile(quantile)
                         + self.df_origin["bid_order_stale_1"].quantile(quantile)
                         + self.df_origin["bid_order_stale_2"].quantile(quantile)
                 ) / 6
        return result

    def get_dis_norm(self):
        self.dis_norm = [
            self.get_quantile(0.25),
            self.get_quantile(0.50),
            self.get_quantile(0.75),
        ]

    def to_discrete_stale(self, df: pd.DataFrame):
        l1, l2, l3 = self.dis_norm
        map_dict = {-1: "no_value", 0: "min", 1: "lower", 2: "upper", 3: "max"}
        df[(df < 0)] = -1
        df[(df >= 0) & (df < l1)] = 0
        df[(df < l2) & (df >= l1)] = 1
        df[(df < l3) & (df >= l2)] = 2
        df[(df >= l3)] = 3
        df = df.replace(map_dict)
        return df

    def to_discrete_normalize(self, df: pd.DataFrame):
        df_dis = self.to_discrete_stale(df)
        dummy_df = pd.get_dummies(df_dis, columns=df_dis.columns, dtype=int)
        return dummy_df

    def insert_columns(self, df: pd.DataFrame):
        columns = df.columns
        for index in range(0, len(self.total_columns)):
            if self.total_columns[index] not in columns:
                df.insert(index, self.total_columns[index], 0)

        return df

    def to_normalize(self, df: pd.DataFrame):
        df_consts = df[self.const_var]
        df_cont = self.to_continuous_normalize(df[self.continuous_var])
        df_dis = self.to_discrete_normalize(df[self.discrete_var])
        self.df_normalized = pd.concat(
            [df_consts, df_cont, df_dis, self.hist_feature], axis=1
        )
        return self.insert_columns(self.df_normalized)

    def initialize_output(
            self, is_train: bool = True, output_path: Path = None, **kwargs
    ):
        self.is_train = is_train
        if self.is_train:
            self.get_continuous_params()
            self.get_dis_norm()

        if output_path is not None:
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)
            for file in tqdm(self.filenames):
                filenames = self.get_past_files(file, limit=5)
                if len(filenames) == 0:
                    continue
                self.generate_hist_feature(filenames)
                self.to_normalize(pd.read_csv(file)).to_csv(
                    output_path / ("norm_" + file.name), index=False
                )

    def to_normalize_by_row(self, df: pd.DataFrame):
        df_cont = self.to_continuous_normalize(df[self.continuous_var])
        df_dis = self.to_discrete_normalize(df[self.discrete_var])
        df_hist = pd.DataFrame([self.hist_feature.iloc[self.index]])
        df_hist = df_hist.reset_index(drop=True)
        df_normalized_row = pd.concat([df_cont, df_dis, df_hist], axis=1)
        return self.insert_columns(df_normalized_row)

    def normalize_by_time(
            self, ob: OrderBook, update_time, rollback: int, write_filename
    ):
        writer = Writer(write_filename, None, rollback=rollback, bid_ask_num=10)
        columns = writer.columns
        df_len = len(self.df_total)
        ob.update(update_time)
        row_list = writer.collect_data_by_timestamp(
            ob, update_time, update_time - rollback
        )
        row_df = pd.DataFrame([row_list], columns=columns)
        row_df_normalized = self.to_normalize_by_row(row_df)
        if df_len == 0:
            self.df_total = row_df_normalized
        elif df_len < 100:
            self.df_total = pd.concat(
                [self.df_total, row_df_normalized], ignore_index=True
            )
        elif df_len == 100:
            self.df_total = pd.concat(
                [self.df_total, row_df_normalized], ignore_index=True
            )
            self.df_total = self.df_total.drop(0).reset_index(drop=True)
        self.index += 1
        return self.df_total

    
