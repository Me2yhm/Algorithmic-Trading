import os
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import cast
import pandas as pd

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
        self.filenames: list[Path] = list(self.file_folder.glob("*.csv"))
        self.df_origin: pd.DataFrame = pd.DataFrame()
        self.df_normalized: pd.DataFrame = pd.DataFrame()
        self.is_train: bool = is_train
        self.continuous_var: list[str] = kwargs.get(
            "continuous_var",
            [
                "preclose_range",
                "open_range",
                "high_range",
                "low_range",
                "close_range",
                "ask_price_0",
                "ask_price_1",
                "ask_price_2",
                "ask_price_3",
                "ask_price_4",
                "ask_price_5",
                "ask_price_6",
                "ask_price_7",
                "ask_price_8",
                "ask_price_9",
                "ask_volume_0",
                "ask_volume_1",
                "ask_volume_2",
                "ask_volume_3",
                "ask_volume_4",
                "ask_volume_5",
                "ask_volume_6",
                "ask_volume_7",
                "ask_volume_8",
                "ask_volume_9",
                "bid_price_0",
                "bid_price_1",
                "bid_price_2",
                "bid_price_3",
                "bid_price_4",
                "bid_price_5",
                "bid_price_6",
                "bid_price_7",
                "bid_price_8",
                "bid_price_9",
                "bid_volume_0",
                "bid_volume_1",
                "bid_volume_2",
                "bid_volume_3",
                "bid_volume_4",
                "bid_volume_5",
                "bid_volume_6",
                "bid_volume_7",
                "bid_volume_8",
                "bid_volume_9",
                "ask_order_num_0",
                "ask_order_num_1",
                "ask_order_num_2",
                "ask_order_num_3",
                "ask_order_num_4",
                "ask_order_num_5",
                "ask_order_num_6",
                "ask_order_num_7",
                "ask_order_num_8",
                "ask_order_num_9",
                "bid_order_num_0",
                "bid_order_num_1",
                "bid_order_num_2",
                "bid_order_num_3",
                "bid_order_num_4",
                "bid_order_num_5",
                "bid_order_num_6",
                "bid_order_num_7",
                "bid_order_num_8",
                "bid_order_num_9",
                "order_num_range",
                "VWAP_range",
                "volume_range",
                "passive_num_range",
                "depth",
            ],
        )
        self.discrete_var: list[str] = kwargs.get(
            "discrete_var",
            [
                "ask_order_stale_0",
                "ask_order_stale_1",
                "ask_order_stale_2",
                "ask_order_stale_3",
                "ask_order_stale_4",
                "ask_order_stale_5",
                "ask_order_stale_6",
                "ask_order_stale_7",
                "ask_order_stale_8",
                "ask_order_stale_9",
                "bid_order_stale_0",
                "bid_order_stale_1",
                "bid_order_stale_2",
                "bid_order_stale_3",
                "bid_order_stale_4",
                "bid_order_stale_5",
                "bid_order_stale_6",
                "bid_order_stale_7",
                "bid_order_stale_8",
                "bid_order_stale_9",
                "passive_stale_avg_range",
            ],
        )
        self.dis_norm: list[int] = kwargs.get("dis_norm", [30, 120, 480])
        self.cont_norm = None

        self.df_list: list[pd.DataFrame] = list()
        self.hist_feature: pd.DataFrame = pd.DataFrame()
        self.df_total: pd.DataFrame = pd.DataFrame()
        self.index: int = 0
        self.total_columns: list[str] = kwargs.get(
            "total_columns",
            [
                "timestamp",
                "trade_price",
                "preclose_range",
                "open_range",
                "high_range",
                "low_range",
                "close_range",
                "ask_price_0",
                "ask_price_1",
                "ask_price_2",
                "ask_price_3",
                "ask_price_4",
                "ask_price_5",
                "ask_price_6",
                "ask_price_7",
                "ask_price_8",
                "ask_price_9",
                "ask_volume_0",
                "ask_volume_1",
                "ask_volume_2",
                "ask_volume_3",
                "ask_volume_4",
                "ask_volume_5",
                "ask_volume_6",
                "ask_volume_7",
                "ask_volume_8",
                "ask_volume_9",
                "bid_price_0",
                "bid_price_1",
                "bid_price_2",
                "bid_price_3",
                "bid_price_4",
                "bid_price_5",
                "bid_price_6",
                "bid_price_7",
                "bid_price_8",
                "bid_price_9",
                "bid_volume_0",
                "bid_volume_1",
                "bid_volume_2",
                "bid_volume_3",
                "bid_volume_4",
                "bid_volume_5",
                "bid_volume_6",
                "bid_volume_7",
                "bid_volume_8",
                "bid_volume_9",
                "ask_order_num_0",
                "ask_order_num_1",
                "ask_order_num_2",
                "ask_order_num_3",
                "ask_order_num_4",
                "ask_order_num_5",
                "ask_order_num_6",
                "ask_order_num_7",
                "ask_order_num_8",
                "ask_order_num_9",
                "bid_order_num_0",
                "bid_order_num_1",
                "bid_order_num_2",
                "bid_order_num_3",
                "bid_order_num_4",
                "bid_order_num_5",
                "bid_order_num_6",
                "bid_order_num_7",
                "bid_order_num_8",
                "bid_order_num_9",
                "order_num_range",
                "VWAP_range",
                "volume_range",
                "passive_num_range",
                "depth",
                "ask_order_stale_0_lower",
                "ask_order_stale_0_max",
                "ask_order_stale_0_min",
                "ask_order_stale_0_no_value",
                "ask_order_stale_0_upper",
                "ask_order_stale_1_lower",
                "ask_order_stale_1_max",
                "ask_order_stale_1_min",
                "ask_order_stale_1_no_value",
                "ask_order_stale_1_upper",
                "ask_order_stale_2_lower",
                "ask_order_stale_2_max",
                "ask_order_stale_2_min",
                "ask_order_stale_2_no_value",
                "ask_order_stale_2_upper",
                "ask_order_stale_3_lower",
                "ask_order_stale_3_max",
                "ask_order_stale_3_min",
                "ask_order_stale_3_no_value",
                "ask_order_stale_3_upper",
                "ask_order_stale_4_lower",
                "ask_order_stale_4_max",
                "ask_order_stale_4_min",
                "ask_order_stale_4_no_value",
                "ask_order_stale_4_upper",
                "ask_order_stale_5_lower",
                "ask_order_stale_5_max",
                "ask_order_stale_5_min",
                "ask_order_stale_5_no_value",
                "ask_order_stale_5_upper",
                "ask_order_stale_6_lower",
                "ask_order_stale_6_max",
                "ask_order_stale_6_min",
                "ask_order_stale_6_no_value",
                "ask_order_stale_6_upper",
                "ask_order_stale_7_lower",
                "ask_order_stale_7_max",
                "ask_order_stale_7_min",
                "ask_order_stale_7_no_value",
                "ask_order_stale_7_upper",
                "ask_order_stale_8_lower",
                "ask_order_stale_8_max",
                "ask_order_stale_8_min",
                "ask_order_stale_8_no_value",
                "ask_order_stale_8_upper",
                "ask_order_stale_9_lower",
                "ask_order_stale_9_max",
                "ask_order_stale_9_min",
                "ask_order_stale_9_no_value",
                "ask_order_stale_9_upper",
                "bid_order_stale_0_lower",
                "bid_order_stale_0_max",
                "bid_order_stale_0_min",
                "bid_order_stale_0_no_value",
                "bid_order_stale_0_upper",
                "bid_order_stale_1_lower",
                "bid_order_stale_1_max",
                "bid_order_stale_1_min",
                "bid_order_stale_1_no_value",
                "bid_order_stale_1_upper",
                "bid_order_stale_2_lower",
                "bid_order_stale_2_max",
                "bid_order_stale_2_min",
                "bid_order_stale_2_no_value",
                "bid_order_stale_2_upper",
                "bid_order_stale_3_lower",
                "bid_order_stale_3_max",
                "bid_order_stale_3_min",
                "bid_order_stale_3_no_value",
                "bid_order_stale_3_upper",
                "bid_order_stale_4_lower",
                "bid_order_stale_4_max",
                "bid_order_stale_4_min",
                "bid_order_stale_4_no_value",
                "bid_order_stale_4_upper",
                "bid_order_stale_5_lower",
                "bid_order_stale_5_max",
                "bid_order_stale_5_min",
                "bid_order_stale_5_no_value",
                "bid_order_stale_5_upper",
                "bid_order_stale_6_lower",
                "bid_order_stale_6_max",
                "bid_order_stale_6_min",
                "bid_order_stale_6_no_value",
                "bid_order_stale_6_upper",
                "bid_order_stale_7_lower",
                "bid_order_stale_7_max",
                "bid_order_stale_7_min",
                "bid_order_stale_7_no_value",
                "bid_order_stale_7_upper",
                "bid_order_stale_8_lower",
                "bid_order_stale_8_max",
                "bid_order_stale_8_min",
                "bid_order_stale_8_no_value",
                "bid_order_stale_8_upper",
                "bid_order_stale_9_lower",
                "bid_order_stale_9_max",
                "bid_order_stale_9_min",
                "bid_order_stale_9_no_value",
                "bid_order_stale_9_upper",
                "passive_stale_avg_range_lower",
                "passive_stale_avg_range_max",
                "passive_stale_avg_range_min",
                "passive_stale_avg_range_no_value",
                "passive_stale_avg_range_upper",
            ],
        )

    def generate_hist_feature(self):
        if self.is_train:
            res = pd.DataFrame()
            hist_feature = cast(pd.DataFrame, 0)
            for file in self.filenames:
                df = pd.read_csv(file)
                hist_feature += df.loc[:, ["VWAP", "volume_range"]]
                res = pd.concat([res, df], axis=0, ignore_index=True)

            hist_feature = hist_feature.rename(
                columns={"VWAP": "VWAP_hist", "volume_range": "volume_range_hist"}
            )
            hist_feature /= len(self.filenames)
            df_VWAP_original = hist_feature["VWAP_hist"].copy(deep=True)
            df_VWAP_original = df_VWAP_original.rename("VWAP_hist_original")
            hist_feature = (
                hist_feature - self.hist_feature.mean()
            ) / hist_feature.std()
            self.hist_feature = pd.concat([hist_feature, df_VWAP_original], axis=1)
            self.df_origin = res

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
        map_dict = {
            -1:"no_value",
            0:"min",
            1:"lower",
            2:"upper",
            3:"max"
        }
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
        df_cont = self.to_continuous_normalize(df[self.continuous_var])
        df_dis = self.to_discrete_normalize(df[self.discrete_var])
        self.df_normalized = pd.concat([df_cont, df_dis, self.hist_feature], axis=1)
        return self.insert_columns(self.df_normalized)

    def initialize_output(
        self, is_train: bool = True, output_path: Path = None, **kwargs
    ):
        self.is_train = is_train
        self.generate_hist_feature()
        if self.is_train:
            self.get_continuous_params()
            self.get_dis_norm()

        if output_path is not None:
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)
            for file in self.filenames:
                self.to_normalize(pd.read_csv(file)).to_csv(
                    output_path / ("norm_" + file.name), index=False
                )

    def to_normalize_by_row(self, df):
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
