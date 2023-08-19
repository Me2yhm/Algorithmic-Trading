import pandas as pd
from typing import Dict, List, TypedDict
import numpy as np
import time as t

class Tick():
    """tick数据流式传入一tick的信息
    """
    
    def __init__(self, one_tick):
        """初始化
            one_tick流式输入一tick
            delta 一周期的时间
        """
        self.time = pd.to_datetime(one_tick[1], format='%Y%m%d%H%M%S%f')#这里可能必须转换成pd时间
        self.price = float(one_tick[2])
        self.volume = int(one_tick[3])
        

class Hurst():
    """
    计算hurst指数，输入收益率时间序列ret_series,返回hurst值
    """
    def __init__(self,ret_series:pd.Series,k=6):
        self.k = k #将ret_series分成k组来回归
        self.ret_series = ret_series
    def calculate(self):
        RS = [0]*self.k
        size_list = [0]*self.k
        for i in range(self.k):
            size_list[i]= len(self.ret_series)/(2**i)
            subseries_list = np.array_split(self.ret_series.index, 2**i)
            #计算每组的R/S值
            for s in range(2**i):
                series = pd.Series(self.ret_series[subseries_list[s]], index=self.ret_series.index[:len(subseries_list[s])])
                std = series.std()
                mean = series.mean()
                # breakpoint()
                if np.isnan(std) or np.isnan(mean):
                    continue
                else:
                    series_delta = series.apply(lambda x: x-mean)
                    R = series_delta.max() - series_delta.min()
                    breakpoint()
                    RS[i] += R/std
                RS[i] = RS[i]/2**i
        #去掉RS的0值，对R/S值和k回归，取系数为hurst
        RS_new = size_list_new = []
        for i in range(len(RS)):
            if RS[i] !=0:
                RS_new.append(RS[i])
                size_list_new.append(len(self.ret_series)/(2**i))
        RS_new = np.array(RS_new)
        size_list_new = np.array(size_list_new)
        if len(RS_new)!=0:
            hurst = np.polyfit(np.log(size_list_new), np.log(RS_new), 1)[0]
        else:
            hurst = None
        return hurst
    
class Factor_turnover_based():
    """
    计算基于换手率排序的因子
    """
    def __init__(self, ret_series, turnover_series):
        #self.risk  #风险
        self.ret_series = ret_series #收益率因子
        self.turnover_series = turnover_series 
    def calculate(self):
        #基于换手率排序，计算因子值
        df_turnover_ret = pd.concat([self.ret_series, self.turnover_series],axis=1)
        df_turnover_ret.columns = ['ret','turnover']
        df_sorted = df_turnover_ret.sort_values(by='turnover')
        index_min= df_sorted['ret'].head(10).mean()
        index_max = df_sorted['ret'].tail(10).mean()
        factor_1 = index_max - index_min
        return factor_1

class Factor_information_based():
    """
    计算基于信息分布排序的因子
    """
    def __init__(self, ret_series,info_series):
        #self.risk  #风险
        self.ret_series = ret_series #收益率因子
        self.info_series = info_series #信息分布
    def calculate(self):
        #基于信息分布排序，计算因子值
        df_info_ret = pd.concat([self.ret_series, self.info_series],axis=1)
        df_info_ret.columns = ['ret','info']
        df_sorted = df_info_ret.sort_values(by='info')
        index_min= df_sorted['ret'].head(10).mean()
        index_max = df_sorted['ret'].tail(10).mean()
        factor_2 = index_max - index_min
        return factor_2
    
class Ret():
    """
    计算收益率和收益率序列
    """
    def __init__(self, time_now: pd.Timestamp , pricelist: pd.Series, m = pd.Timedelta(seconds=10) ):
        self.m = m  # 算ret的时间间隔
        self.time_now = time_now
        self.pricelist = pricelist
    def calculate(self, time):
        #计算收益率
        time_begin = time - self.m
        try:
            ret = self.pricelist[time] / self.pricelist[time_begin]
        except Exception as e:
            ret = None  
        return ret
    def form_series(self, interval= pd.Timedelta(seconds = 20), l = 60):
        #返回收益率序列，时间间隔为interval
        time_list = sorted([self.time_now - i * interval for i in range(l)])
        ret_list =  [self.calculate(t) for t in time_list]
        ret_series = pd.Series(ret_list, index = time_list)
        return ret_series
        
           
class Turnover():
    """
    计算换手率和换手率序列
    """
    def __init__(self, time_now: pd.Timestamp , tickdict:Dict, interval = pd.Timedelta(seconds=20)):
        self.interval = interval
        self.time_now = time_now
        #把tick字典的键str->pd.Timestamp
        self.tickdict = tickdict
        
    def form_series(self, l = 61, total_shares = 293.52):
        #total_shares 为总发行股数*（10**8）
        #返回换手率序列
        time_list = sorted([self.time_now - i * self.interval for i in range(l)])
        volume = turnover_series = {}
        for i in range(len(time_list)-1):
            volume_temp = 0
            start_time = time_list[i]
            end_time = time_list[i+1]
            #取出特定时间段的tick，计算volume
            date_today = self.time_now.strftime('%Y-%m-%d')
            # breakpoint()
            ticklist_today = self.tickdict[date_today]
            for tick in ticklist_today:
                if  start_time < pd.to_datetime(tick['time'], format='%Y%m%d%H%M%S%f') <= end_time:
                    if isinstance(tick['volume'], (int, float)):
                            volume_temp += tick['volume']
                    else:
                        print("Tick volume is not an int:", tick['volume'])
                    volume[end_time] = volume_temp
            if end_time in volume.keys():
                turnover_series[end_time] = volume[end_time]/total_shares
            else:
                turnover_series[end_time] = None
            turnover_series = pd.Series(turnover_series)
        return turnover_series

class Information():
    """
    计算信息分布和信息分布序列
    """
    def __init__(self, time_now: pd.Timestamp , tickdict: Dict, m = pd.Timedelta(milliseconds=500)):
        self.m = m  # 算信息分布的时间长度
        self.time_now = time_now
        self.tickdict = tickdict
    def calculate(self, time, s = 120):
        #计算某一时刻信息分布
        time_list = sorted([time - i * self.m for i in range(s)])
        volume = {}
        for i in range(len(time_list)-1):
            volume_temp = 0
            start_time = time_list[i]
            end_time = time_list[i+1]
            #取出特定时间段的tick，计算volume
            date_today = self.time_now.strftime('%Y-%m-%d')
            ticklist_today = self.tickdict[date_today]
            for tick in ticklist_today:
                if  start_time < pd.to_datetime(tick['time'], format='%Y%m%d%H%M%S%f') <= end_time:
                    if isinstance(tick['volume'], int):
                            volume_temp += tick['volume']
                    else:
                        print("Tick volume is not an int:", tick['volume'])
                    if end_time in volume.keys():
                        volume[end_time] = volume_temp
                    else:
                        volume[end_time] = None
        volume_array = np.array(list(volume.values()))

        try:
            volume_std = np.std(volume_array)
            volume_mean = np.mean(volume_array) 
        except Exception as e:
            volume_std = volume_mean = None
        try:
            info = volume_std/volume_mean
        except Exception as e:
            info = None

        return info
    def form_series(self, interval= pd.Timedelta(seconds = 20), l = 60):
        #计算信息分布时间序列
        time_list = sorted([self.time_now - i * interval for i in range(l)])
        info_list = [(time_list[i], self.calculate(time = time_list[i])) for i in range(len(time_list))]
        info_series = pd.Series(info_list, index = time_list)
        return info_series
    
############################################# 

#直接用tick文件  形成tickdict  
#tickdict = {}
#with open("D:\\玖奕\\拼盘口1\\拼盘口\\tick.csv", 'r') as file:
    # reader = csv.reader(file)
    # next(reader)
    # for row in reader:
    #     one_tick = Tick(row)
    #     if one_tick.time in tickdict.keys():
    #         tickdict[one_tick.time].append(one_tick)
    #     else:
    #         tickdict[one_tick.time] = [one_tick]

       
