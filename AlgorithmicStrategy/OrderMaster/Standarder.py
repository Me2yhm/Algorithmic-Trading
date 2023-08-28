import json
from pathlib import Path
from typing import cast, Union
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm


class Standarder:
    def __init__(self, file_folder: Path, train: bool, limits: int = 5, **kwargs):
        self.df_volume_percentage = None
        self.hist_df = None
        self.hist_feature = None
        self.dis_param = None
        self.con_param_std = None
        self.con_param_mean = None
        self.vars = json.loads(open(Path(__file__).parent / "vars.json", "r").read())
        self.const_var: list[str] = self.vars.get("const_var")
        self.continuous_var: list[str] = self.vars.get("continuous_var")
        self.discrete_var: list[str] = self.vars.get("discrete_var")
        self.total_columns: list[str] = self.vars.get("total_columns")

        self.file_folder: Path = file_folder
        self.limits: int = limits

        self.filenames: list[Path] = list()
        self.dfs: dict[Path, pd.DataFrame] = dict()
        self.all_data: pd.DataFrame = None

        self.train: bool = train

    def fresh_files(self):
        self.filenames: list[Path] = list(self.file_folder.glob("*.csv"))
        self.filenames.sort(key=lambda x: datetime.strptime(x.stem, "%Y-%m-%d"))

    def read_files(self):
        self.dfs: dict[Path, pd.DataFrame] = {k: pd.read_csv(k) for k in self.filenames}

    def get_past_files(self, file: Path):
        idx = self.filenames.index(file)
        if self.limits is not None:
            if idx < self.limits:
                return self.filenames[:idx]
            else:
                return self.filenames[idx - self.limits : idx]
        else:
            return self.filenames[:idx]

    def generate_hist_feature(self, filenames: list[Path]):
        hist_feature = cast(pd.DataFrame, 0)
        for file in filenames:
            df = self.dfs[file]
            hist_feature += df.loc[:, ["VWAP", "volume_range"]]

        df_volume = hist_feature.copy()
        df_volume["volume_percentage"] = (
            df_volume["volume_range"] / df_volume["volume_range"].sum()
        )
        df_volume["timestamp"] = df["timestamp"]
        df_volume_percentage = df_volume[['timestamp', 'volume_percentage']]

        hist_feature = hist_feature.rename(
            columns={"VWAP": "VWAP_hist", "volume_range": "volume_range_hist"}
        )
        hist_feature /= len(self.filenames)

        hist_feature = (hist_feature - hist_feature.mean()) / hist_feature.std()
        hist_feature["volume_percentage"] = df_volume_percentage["volume_percentage"]
        self.hist_feature = hist_feature
        self.df_volume_percentage = df_volume_percentage
        self.hist_df = pd.concat(
            [self.dfs[s] for s in filenames], axis=0, ignore_index=True
        )

    def fit(self, data: pd.DataFrame):
        con_data = data[self.continuous_var].values
        self.con_param_mean = con_data.mean()
        self.con_param_std = con_data.std()
        self.dis_param = self.get_quantile(data, quantile=[0.25, 0.5, 0.75])

    def get_quantile(self, data: pd.DataFrame, quantile: Union[float, list[float]]):
        tmp = data[
            [
                "ask_order_stale_0",
                "ask_order_stale_1",
                "ask_order_stale_2",
                "bid_order_stale_0",
                "bid_order_stale_1",
                "bid_order_stale_2",
            ]
        ].values

        if isinstance(quantile, float):
            quantile = [quantile]

        res = []
        for q in quantile:
            res.append(float(np.quantile(tmp, q, axis=0).mean()))

        return res

    def transform(self, data: pd.DataFrame):

        df_volume = data[['volume_range']].copy()
        df_volume['volume_range'] /= df_volume['volume_range'].sum()
        df_volume = df_volume.rename(columns={"volume_range": "volume_percent_today"})
        df_consts = data[self.const_var]
        df_cont = (data[self.continuous_var] - self.con_param_mean) / self.con_param_std

        df_dis = data[self.discrete_var].copy()
        l1, l2, l3 = self.dis_param
        map_dict = {-1: "no_value", 0: "min", 1: "lower", 2: "upper", 3: "max"}
        df_dis[(df_dis < 0)] = -1
        df_dis[(df_dis >= 0) & (df_dis < l1)] = 0
        df_dis[(df_dis < l2) & (df_dis >= l1)] = 1
        df_dis[(df_dis < l3) & (df_dis >= l2)] = 2
        df_dis[(df_dis >= l3)] = 3
        df_dis = df_dis.replace(map_dict)
        df_dis = pd.get_dummies(df_dis, columns=df_dis.columns, dtype=int)
        df_normalized = pd.concat(
            [df_consts, df_cont, df_dis, self.hist_feature, df_volume], axis=1
        )
        return self.insert_columns(df_normalized)

    def insert_columns(self, df: pd.DataFrame):
        columns = df.columns
        for index in range(0, len(self.total_columns)):
            if self.total_columns[index] not in columns:
                df.insert(index, self.total_columns[index], 0)
        return df

    def fit_transform(
        self,
        datas: Union[Path, list[Path], pd.DataFrame] = None,
        output: Path = None,
        label: Path = None,
    ):
        if datas is None:
            datas = self.filenames
        else:
            if isinstance(datas, Path):
                datas = [datas]
        if output is not None and not output.exists():
            output.mkdir(parents=True, exist_ok=True)

        if label is not None and not label.exists():
            label.mkdir(parents=True, exist_ok=True)

        if isinstance(datas, list):
            for file in tqdm(datas):
                past_filenames = self.get_past_files(file)
                if len(past_filenames) == 0:
                    continue
                self.generate_hist_feature(past_filenames)
                self.fit(self.hist_df)
                df_normalized = self.transform(self.dfs[file])
                if output is not None:
                    df_normalized.to_csv(output / file.name, index=False)

                if label is not None:
                    self.df_volume_percentage.to_csv(label / file.name, index=False)
        else:
            df_normalized = self.transform(datas)
            return df_normalized

    def add_new_files(self, path: Path):
        self.filenames.append(path)
        self.dfs[path] = pd.read_csv(path)
