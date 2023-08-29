import math
import numpy as np
from typing import Dict, Union

import pandas as pd

from .modelType import modelType


class Tick:
    """tick数据流式传入一tick的信息"""

    def __init__(self, one_tick, delta=500):
        """初始化
        one_tick流式输入一tick
        delta 一周期的时间 毫秒级别
        """
        self.Time = pd.to_datetime(
            str(one_tick["time"]), format="%Y%m%d%H%M%S%f"
        )  # 这里可能必须转换成pd时间
        self.Price = float(one_tick["price"])
        self.Qty = int(one_tick["volume"])
        self.delta = int(delta)  # 时间切刀
        self.T: int
        self.is_time_node = False  # 是否这单tick的时间为正好处于周期节点


class Snapshot:
    def __init__(self, one_snapshot, delta=500):
        # one_snapshot 指orderbook.snapshots['某时间']
        time_int = list(one_snapshot.values())[0]

        self.ask1, self.ask_vol = max(one_snapshot["ask"].items())
        if one_snapshot["bid"] != {}:  # 集合竞价时为空
            self.bid1, self.bid_vol = min(one_snapshot["bid"].items())
        else:
            self.bid1 = self.ask1
        self.Time = pd.to_datetime(
            str(time_int), format="%Y%m%d%H%M%S%f"
        )  # 转换成pd.Timestamp
        self.T: int
        self.delta = int(delta)  # 时间切刀


class T:  # 周期
    def __init__(self, one_stream: Union[Tick, Snapshot]):
        self.stream = one_stream
        self.start: pd.Timestamp
        self.end: pd.Timestamp
        self.period: int

    def cal_T(self):
        time = self.stream.Time
        opentime = (
            str(time.year)
            + str(time.month).zfill(2)
            + str(time.day).zfill(2)
            + "092500000"
        )
        openqu = pd.to_datetime(opentime, format="%Y%m%d%H%M%S%f")
        Delta = pd.Timedelta(time - openqu)
        period = pd.Timedelta(self.stream.delta, unit="microseconds")
        Delta_float = int(Delta.total_seconds()) + float(Delta.microseconds) / (10**6)
        period_float = int(period.total_seconds())
        cont = Delta_float // period_float
        self.period = int(cont)
        self.stream.T = int(cont)  # tick/snapshot的周期
        self.start = openqu + cont * period  # pd.TimeStamp
        self.end = openqu + (cont + 1) * period  # pd.TimeStamp


class StreamDict:
    """dic: dict[int,Union[list, Snapshot]]"""

    def __init__(self):
        self.dic = {0: []}
        self.max_T_dict = 0  # 字典的最大周期
        self.old_T_dict = 0  # 字典的老周期

    def append(self, one_stream: Union[Tick, Snapshot]):
        """逻辑说明:
        第一周期的1进来, old = 0, max = 1
        第一周期的2进来, old = 1, max = 1
        第一周期的3进来, old = 1, max = 1
        第二周期的1进来, old = 1, max = 2
        以此类推, 我们只有当dict的old != max的tick进来, 即一个新的周期刚被创建时, 才计算之前老周期所对应的一些指标

        Args:
            one_tick (Tick): _description_

        """
        if isinstance(one_stream, Tick):  # StreamDict为存放ticks的字典
            if one_stream.T not in self.dic:  # 如果是这个键是空的,添加列表
                if (one_stream.T - self.max_T_dict) > 1:  # 如果出现周期跳跃
                    for i in range(self.max_T_dict + 1, one_stream.T):
                        self.dic[i] = []
                    self.dic[one_stream.T] = [one_stream]
                    self.old_T_dict = self.max_T_dict
                    self.max_T_dict = one_stream.T
                elif (one_stream.T - self.max_T_dict) <= 1:
                    self.dic[one_stream.T] = [one_stream]
                    self.old_T_dict = self.max_T_dict
                    self.max_T_dict = one_stream.T
            else:
                self.dic[one_stream.T].append(one_stream)
                self.old_T_dict = self.max_T_dict

        elif isinstance(one_stream, Snapshot):  # StreamDict为存放snapshot的字典
            """snapshot比tick的字典多一行！dict存放的是周期开始的price，收盘时的price对应一个新的周期"""
            if one_stream.T not in self.dic:
                if (one_stream.T - self.max_T_dict) > 1:  # 出现周期跳跃
                    for i in range(self.max_T_dict + 1, one_stream.T):
                        self.dic[i] = []
                self.dic[one_stream.T] = [one_stream]
                self.old_T_dict = self.max_T_dict
                self.max_T_dict = one_stream.T
            else:
                self.dic[one_stream.T].append(one_stream)
                self.old_T_dict = self.max_T_dict
        else:
            print("流式输入数据类型有误")


class TurnOver:
    """秒级换手率

    Args:
        是RiskType的子类
    cal_turnover: 计算当前周期的换手率,并记录当前周期的成交量
    """

    def __init__(self, Dic: StreamDict):
        """
        Args:
            tick 全部的tick数据类实例化对象
        """
        self.stream = Dic
        self.dict = dict()
        self.volume_dict = dict()

    def cal_turnover(self, whole, i):  # whole是流通股, i是所计算的to的周期
        j: Tick  # 类型注释
        qty = 0
        for j in list(self.stream.dic[i]):
            qty += j.Qty
        self.volume_dict[i] = qty
        self.dict[i] = qty / whole  # 换手率储存在dict里


class Ret:
    """收益率

    cal_ret：计算当前周期的收益率
    """

    def __init__(self, Dic: StreamDict):
        """初始化

        Args:
            Dic (StreamDict): 全部的snapshot实例化对象
        """
        self.stream = Dic
        self.price_dict = dict()
        self.dict = dict()  # ret_dict

    def cal_ret(self, i):  # i为计算return的周期
        # 计算收益率
        if (
            i >= 0
            and i in self.price_dict
            and i + 1 in self.price_dict
            and self.price_dict[i] != None
            and self.price_dict[i] != 0
        ):
            self.dict[i] = self.price_dict[i + 1] / self.price_dict[i]
        else:
            self.dict[i] = None


class Info:
    def __init__(self, turnover: TurnOver):
        self.dict = dict()
        self.stream = turnover.volume_dict
        self.mean_old = None  # 记录上一次计算的mean
        self.mean_new = None
        self.std = None  # 记录上一次计算的std
        self.first_volume = None  # 上一次计算的最初周期的volume
        self.last_volume = None  # 本次计算的最后周期的volume
        self.var = None

    def cal_info(self, i, period=10):
        # i为计算info的周期,period为计算std的周期
        fix = period / (period - 1)
        if i in self.stream:  # 已有第i期ret
            if (i - period + 1) in self.stream:  # 可回看perioed期数据计算std
                if (i - period) not in self.stream:  # 第一个可算std的周期
                    self.first_volume = self.stream[
                        i - period + 1
                    ]  # 存入第一个volume 用于增量计算
                    data_values = list(self.stream.values())  # 将字典的值转换为列表
                    self.std = np.std(data_values)
                    self.var = self.std**2  # 初始化：第一个方差
                    self.mean_new = self.mean_old = np.mean(data_values)
                    if self.mean_new != 0:
                        self.dict[i] = self.std / self.mean_new
                    else:
                        self.dict[i] = 0
                else:
                    # 增量法计算标准差
                    self.last_volume = self.stream[i]
                    self.mean_new = self.mean_old + (
                        (self.last_volume - self.first_volume) / period
                    )  # 无偏估计
                    self.var = self.var + fix * (
                        self.mean_old**2
                        - self.mean_new**2
                        + 1 / period * (self.last_volume**2 - self.first_volume**2)
                    )
                    self.std = math.sqrt(self.var)
                    self.dict[i] = self.std / self.mean_new
                    self.mean_old = self.mean_new
                    # 更改first_volume为本次第一个周期的volume
                    self.first_volume = self.stream[i - period + 1]
            else:
                self.dict[i] = None
        else:
            print("error:要计算当期info，请先更新ret字典")


class Hurst:
    def __init__(self, k=6):
        self.k = k  # 将ret_series分成k组来回归
        self.hurst_dict = dict()
        self.stream = None

    def cal_hurst(self, i, ret_dict, period=128):
        # period最好取2的幂次
        if (i - period + 1) < 0:  # 周期不满
            self.hurst_dict[i] = None
        else:
            start_time = i - period + 1
            self.stream = {
                x: ret_dict[x] for x in ret_dict if start_time <= x <= i
            }  # ret_dict为ret的字典{int：float},截取周期长度为period
            ret_series = pd.Series(self.stream, index=list(self.stream.keys()))
            # 计算第i期的hurst指数
            RS = [0.0] * self.k  # initialize R/S-index
            size_list = [0] * self.k
            for j in range(self.k):
                # 第j种划分，即划分为2**j组
                # TODO
                size_list[j] = period / (2**j)  # 第j种划分方式下，一组的周期数目
                subseries_index_list = np.array_split(ret_series.index, 2**j)
                count = 0
                for s in range(2**j):
                    series = pd.Series(
                        ret_series[subseries_index_list[s]],
                        index=ret_series.index[: len(subseries_index_list[s])],
                    )
                    std = series.std()
                    mean = series.mean()
                    if np.isnan(std) or np.isnan(mean):
                        continue
                    else:
                        series_delta = series.apply(lambda x: x - mean)
                        R = series_delta.max() - series_delta.min()
                        # breakpoint()
                        if std != 0:
                            RS[j] += R / std
                            count += 1
                    if count != 0:
                        RS[j] = RS[j] / count
                    else:
                        RS[j] = 0
            # 去掉RS的0值，对R/S值和k回归，取系数为hurst
            RS_new = size_list_new = []
            for j in range(len(RS)):
                if RS[j] != 0:
                    RS_new.append(RS[j])
                    size_list_new.append(len(ret_series) / (2**j))
            RS_new = np.array(RS_new)
            size_list_new = np.array(size_list_new)
            if len(RS_new) != 0:
                self.hurst_dict[i] = np.polyfit(
                    np.log(size_list_new), np.log(RS_new), 1
                )[0]
            else:
                self.hurst_dict[i] = None


class Factor1:
    """
    计算基于换手率排序的反转因子
    """

    def __init__(self):
        self.factor1_dict = dict()

    def cal_factor1(self, i: int, ret: Ret, turnover: TurnOver, delta=5):
        """i为当前周期,delta为turnover排序的周期"""
        # 基于换手率对ret排序
        # 先获取排序后的周期list
        if i - delta < 0:  # 周期不足 #要算ret需要下一个周期的price delta=5的时候需要0~4期（且第5期已到，即i>=5）的数据
            self.factor1_dict[i] = None
        else:
            start_period = i - delta
            # breakpoint()
            sorted_turnover_periods = sorted(
                [x for x in range(start_period, i) if turnover.dict[x] is not None],
                key=lambda x: turnover.dict[x],
            )
            try:
                self.factor1_dict[i - 1] = (
                    ret.dict[sorted_turnover_periods[-1]]
                    - ret.dict[sorted_turnover_periods[0]]
                )
            except Exception as e:
                self.factor1_dict[i - 1] = None

        # 根据这个排序生成相应的factor


class Factor2:
    """计算基于信息分布排序的反转因子"""

    def __init__(self):
        self.factor2_dict = dict()

    def cal_factor2(self, i: int, ret: Ret, info: Info, delta=5):
        """i为当前周期,delta为info排序的周期"""
        # 基于info对ret排序
        # 先获取排序后的周期list
        if i - delta < 0:  # 周期不足
            self.factor2_dict[i] = None
        else:
            start_period = i - delta
            sorted_info_periods = sorted(
                [x for x in range(start_period, i) if info.dict[x] is not None],
                key=lambda x: info.dict[x],
            )
            try:
                self.factor2_dict[i - 1] = (
                    ret.dict[sorted_info_periods[-1]] - ret.dict[sorted_info_periods[0]]
                )
            except Exception as e:
                self.factor2_dict[i - 1] = None
        # 根据这个排序生成相应的factor


class Model_reverse(modelType):
    def __init__(
        self,
        delta_stream=30000000,
        std_period=10,
        hurst_period=128,
        k=6,
        whole=294e10,
        delta_factor1=5,
        delta_factor2=5,
    ):
        # 一开始初始化的时候先把这些实例化，然后每次update就更新这里的属性
        self.TD = StreamDict()  # tickdict
        self.SD = StreamDict()  # snapshotdict
        self.TO = TurnOver(self.TD)  # 计算换手率，储存换手率
        self.ret = Ret(self.SD)  # 计算股票收益率，储存
        self.info = Info(self.TO)
        self.k = k  # hurst计算时的分组数目
        self.hurst = Hurst(self.k)
        self.factor1 = Factor1()
        self.factor2 = Factor2()
        self.delta1 = delta_factor1
        self.delta2 = delta_factor2
        self.whole = whole  # 总流通股本
        self.delta = delta_stream  # 周期间隔 以微秒为单位
        self.info_period = std_period  # 计算波动率的周期数
        self.hurst_period = hurst_period  # 计算hurst的周期数
        self.period_now: int
        self.period_start: pd.Timestamp
        self.period_end: pd.Timestamp
        self.total_period = int(23700000000 / self.delta) + 1  # 总周期数
        self.period_list = [0] * self.total_period  # 周期列表

    def _error(self):
        # 报错并跳过
        print("This stream data can't calculate all indexes!")

    def _tick_store(self, one_tick):
        # 每次进来一条tick，就调用这个函数,将tick的信息进行预处理，并储存在self中的属性中
        one_Tick = Tick(one_tick, delta=self.delta)  # 初始化tick对象
        T_tick = T(one_Tick)  # 计算tick的周期
        T_tick.cal_T()
        # print("tick周期:", one_Tick.T)
        self.period_now = one_Tick.T  # 用最新一单tick的周期作为当前周期
        self.TD.append(one_Tick)  # 将这单tick添加到TD中
        self.period_start = T_tick.start
        self.period_end = T_tick.end

    def _snapshot_store(self, one_snapshot):
        # 每次进来一条snapshot，就调用这个函数,将snapshot的信息进行预处理，并储存在self中的属性中
        one_Snapshot = Snapshot(one_snapshot, delta=self.delta)
        one_Snapshot.T = self.period_now
        self.SD.append(one_Snapshot)  # 将价格添加到SD中

    def _cal_turnover(self, i):
        self.TO.cal_turnover(self.whole, i)  # 计算这个周期的turnover

    def _cal_ret(self, i):
        self.ret.cal_ret(i)  # 计算这个周期的ret

    def _cal_info(self, i):
        self.info.cal_info(i, period=self.info_period)

    def _cal_hurst(self, i):
        self.hurst.cal_hurst(i, self.ret.dict, self.hurst_period)

    def _cal_factor1(self, i):
        self.factor1.cal_factor1(i, self.ret, self.TO, self.delta1)

    def _cal_factor2(self, i):
        self.factor2.cal_factor2(i, self.ret, self.info, self.delta2)

    def _get_closest_time(self, query_stamp, timelist: list):
        logged_timestamp: np.ndarray = np.array(timelist)
        search_timestamp = logged_timestamp[logged_timestamp <= query_stamp]
        return search_timestamp[-1]

    def _cal_pricedict(self, price_dict_all, date_today, openqu):
        price_list = [0.0] * self.total_period
        for i in range(self.TD.old_T_dict, self.TD.max_T_dict + 1):
            timestamp = (
                int(list(self.TD.dic.keys())[i]) * pd.Timedelta(microseconds=self.delta)
                + openqu
            )  # type:ignore
            self.period_list[i] = int(timestamp.strftime("%Y%m%d%H%M%S%f")[:-3])
            price_dict1 = price_dict_all[date_today]
            price_dict = {
                time_int: price_dict1[time_int]
                for time_int in price_dict1.keys()
                if int(str(time_int)[8:]) >= 92400000
            }
            closest_time = self._get_closest_time(
                self.period_list[i], list(price_dict.keys())
            )  # 获取此周期最近的盘口时间：int
            price_list[i] = price_dict[closest_time]
            self.ret.price_dict[i] = price_list[i]

    # strategy的self.timestamp作为当前时间输入
    def model_update(
        self, ticks, price_dict_all: Dict[str, Dict[int, float]], timestamp: int
    ):
        # 先储存
        # timestamp 17位int
        date_today = (
            str(timestamp)[:4] + "-" + str(timestamp)[4:6] + "-" + str(timestamp)[6:8]
        )
        if int(str(timestamp)[8:]) >= 92500000:  # 跳过集合竞价期间的数据
            time_now = pd.to_datetime(str(timestamp), format="%Y%m%d%H%M%S%f")
            tickdict = ticks[date_today]  # 到目前为止所有时刻的ticks
            tickdict_now = tickdict[timestamp]
            # 0. 存入one_tick和one_snapshot
            for one_tick in tickdict_now:
                self._tick_store(one_tick)  # 把一个时间戳的ticks一条条变为Tick对象存入TD
            period_start_int = int(self.period_start.strftime("%Y%m%d%H%M%S%f")[:-3])
            opentime = (
                str(time_now.year)
                + str(time_now.month).zfill(2)
                + str(time_now.day).zfill(2)
                + "092500000"
            )
            openqu = pd.to_datetime(opentime, format="%Y%m%d%H%M%S%f")
            try:
                self._cal_pricedict(price_dict_all, date_today, openqu)
                if self.TD.max_T_dict == self.TD.old_T_dict:
                    return
                for i in range(self.TD.old_T_dict, self.TD.max_T_dict):
                    self._cal_turnover(i)
                    self._cal_ret(i)
                    self._cal_info(i)
                    # 计算反转因子和hurst
                    self._cal_factor1(i)
                    self._cal_factor2(i)
                    self._cal_hurst(i)
            except KeyError:
                print("there were no price yet")
        return {
            "factor1": self.factor1.factor1_dict,
            "factor2": self.factor2.factor2_dict,
            "hurst": self.hurst.hurst_dict,
        }
