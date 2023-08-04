import pandas as pd
import numpy as np
import csv

class Hurst():
    def __init__(self,ret_series:pd.Series,k=6):
        self.k = k #取几组来回归
        self.ret_series = ret_series
    def calculate(self):
        RS = [0]*self.k
        for i in range(self.k):
            subseries_list = np.array_split(self.ret_series.values, 2**i)
            RS[i] = 0
            for s in range(2**i):
                series = pd.Series(subseries_list[s], index=self.ret_series.index[:len(subseries_list[s])])
                std = series.std()
                mean = series.mean()
                series_delta = series.apply(lambda x: x-mean)
                R = series_delta.max() - series_delta.min()
                RS += R/std
            RS[i] = RS[i]/2**i
        hurst = np.polyfit(np.log(len(RS)), np.log(RS), 1)[0]
        return hurst
    
class Factor_turnover_based():
    def __init__(self, ret_series, turnover_series):
        #self.risk  #风险
        self.ret_series = ret_series #收益率因子
        self.turnover_series = turnover_series #信息分布
    def calculate(self):
        df_turnover_ret = pd.concat([self.ret_series, self.turnover_series],axis=1)
        df_turnover_ret.columns = ['ret','turnover']
        df_sorted = df_turnover_ret.sort_values(by='turnover')
        index_min= df_sorted['ret'].head(10).mean()
        index_max = df_sorted['ret'].tail(10).mean()
        factor_1 = index_max - index_min
        return factor_1

class Factor_information_based():
    def __init__(self, ret_series,info_series):
        #self.risk  #风险
        self.ret_series = ret_series #收益率因子
        self.info_series = info_series #信息分布
    def calculate(self):
        df_info_ret = pd.concat([self.ret_series, self.info_series],axis=1)
        df_info_ret.columns = ['ret','info']
        df_sorted = df_info_ret.sort_values(by='info')
        index_min= df_sorted['ret'].head(10).mean()
        index_max = df_sorted['ret'].tail(10).mean()
        factor_2 = index_max - index_min
        return factor_2
    
class Ret():
    def __init__(self, time_now: pd.Timestamp , pricelist: pd.Series, m = pd.Timedelta(seconds=5) ):
        self.m = m  # 算ret的时间间隔
        self.time_now = time_now
        self.pricelist = pricelist
    def calculate(self, time):
        time_begin = time - self.m
        try:
            ret = self.pricelist[time] / self.pricelist[time_begin]
        except KeyError:
            ret = None  
        return ret
    def form_series(self, interval= pd.Timedelta(seconds = 1), l = 60):
        time_list = sorted([self.time_now - i * interval for i in range(l)])
        ret_list =  [self.calculate(t) for t in time_list]
        ret_series = pd.Series(ret_list, index = time_list)
        return ret_series
        
           
class Turnover():
    def __init__(self, time_now: pd.Timestamp , tick: pd.Series, interval = pd.Timedelta(seconds=1)):
        self.interval = interval
        self.time_now = time_now
        self.tick = tick
    def form_series(self, l = 61, total_shares = None):
        time_list = sorted([self.time_now - i * self.interval for i in range(l)])
        volume = pd.Series([0]*(l-1),index = time_list[1:])
        for i in range(len(time_list)-1):
            start_time = time_list[i]
            end_time = time_list[i+1]
            start_idx = self.tick.index.searchsorted(start_time)
            end_idx = self.tick.index.searchsorted(end_time)
            subset= self.tick.iloc[start_idx:end_idx]
            volume.loc[time_list[i]] = subset.apply(lambda d: sum(d)).sum()
        turnover_series = volume.apply(lambda x: x/total_shares)
        return turnover_series

class Information():
    def __init__(self, time_now: pd.Timestamp , tick: pd.Series, m = pd.Timedelta(milliseconds=500)):
        self.m = m  # 算信息分布的时间长度
        self.time_now = time_now
        self.tick = tick
    def calculate(self, time, s = 120):
        time_list = sorted([time - i * self.m for i in range(s)])
        volume = pd.Series([0]*(s-1),index = time_list[1:]) #初始化
        for i in range(len(time_list)-1):
            start_time = time_list[i]
            end_time = time_list[i+1]
            start_idx = self.tick.index.searchsorted(start_time)
            end_idx = self.tick.index.searchsorted(end_time)
            subset= self.tick.iloc[start_idx:end_idx]
            volume.loc[time_list[i]] = subset.apply(lambda d: sum(d)).sum()
        volume_std = volume.std()
        volume_mean = volume.mean() 
        try:
            info = volume_std/volume_mean
        except KeyError:
            info = None
        return info
    def form_series(self, interval= pd.Timedelta(seconds = 1), l = 60):
        time_list = sorted([self.time_now - i * interval for i in range(l)])
        info_list = [(time_list[i], self.calculate(time = time_list[i])) for i in range(len(time_list))]
        info_series = pd.Series(info_list, index = time_list)
        return info_series
    
############################################# 
class modelType(ABC):
    """
    模型的抽象基类，用于实现指标更新或者模型增量训练
    """
    @abstractmethod
    def model_update():
        """
        数据增量地传进来，每传进来一条数据，模型计算一次指标或者训练一次模型
        """
class model_reverse(modelType):
    def model_update(self,tick,orderbook):
        """
        基于更新的数据计算新的反转因子
        """
        #修改tick数据的格式 
        time_now = tick.index[-1]
        
        #从盘口信息得到price_list #卖一和买一的加权平均作为price
        time_list = list(orderbook.snapshots.keys())
        price_list = [0]*len(time_list)
        for i in range(len(time_list)):
            ask_1,volume_ask = next(iter(orderbook.snapshots.ask.items()))
            bid_1,volume_bid = next(iter(orderbook.snapshots.bid.items()))
            price_list[i] = ask_1*volume_ask/(volume_ask+volume_bid)+bid_1*volume_bid/(volume_bid+volume_ask)
        price_list = pd.Series(price_list,index=time_list)
        
        ret_series = Ret(time_now,price_list).form_series()
        turnover_series = Turnover(time_now,tick).form_series()
        hurst = Hurst(ret_series).calculate() 
        info_series = Information(time_now,tick).form_series()
        factor_1 = Factor_turnover_based(ret_series, turnover_series).calculate() 
        factor_2 = Factor_information_based(ret_series,info_series).calculate()
        index_dict = {'hurst':hurst,'factor_turnover':factor_1,'factor_info':factor_2}
        return index_dict        


with open("D:\\玖奕\\拼盘口1\\拼盘口\\tick.csv", 'r') as file:
    reader = csv.reader(file)
    next(reader)
    cols = ['code','time','price','volume','Amount','buyNo','sellNo','index','channel','flag','bizindex']
    tick_df = pd.DataFrame(columns=cols)
    for row in reader:
        one_tick = pd.DataFrame([row],columns=cols)
        one_tick['time'][0] = pd.to_datetime(one_tick['time'][0][:-1],format='%Y%m%d%H%M%S%f')
        tick_df = pd.concat([tick_df,one_tick],ignore_index = True)
    tick = tick_df.groupby('time')['volume'].apply(list)
       
