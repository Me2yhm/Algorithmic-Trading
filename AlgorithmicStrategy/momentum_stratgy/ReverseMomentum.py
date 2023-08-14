import pandas as pd
from abc import abstractmethod, ABC
from typing import Dict, List, TypedDict,Union
import numpy as np
from .modelType import modelType
from ..OrderMaster.OrderBook import OrderBook
import math

class Tick():
    """tick数据流式传入一tick的信息
    """
    
    def __init__(self, one_tick, delta=500):
        """初始化
            one_tick流式输入一tick
            delta 一周期的时间 毫秒级别
        """
        self.Time = pd.to_datetime(str(one_tick['time']), format='%Y%m%d%H%M%S%f')#这里可能必须转换成pd时间
        self.Price = float(one_tick['price'])
        self.Qty = int(one_tick['volume'])
        self.delta = int(delta)#时间切刀
        self.T : int
        self.is_time_node = False #是否这单tick的时间为正好处于周期节点
        #pd.Timedelta((self.Time - pd.to_datetime(20230228092500000, format='%Y%m%d%H%M%S%f')), unit='microseconds')
        #这里要把时间全部转换成毫秒再相除
        #所属周期，这里必须需要转换成pd时间

class Snapshot():
    def __init__(self, one_snapshot, delta=500):
        # one_snapshot 指orderbook.snapshots['某时间']
        time_int = list(one_snapshot.keys())[-1]
        self.ask1, self.ask_vol = max(one_snapshot[time_int]["ask"].keys())
        self.bid1, self.bid_vol = min(one_snapshot[time_int]["bid"].keys())
        self.Time = pd.to_datetime(str(time_int),format='%Y%m%d%H%M%S%f') #转换成pd.Timestamp
        self.T : int
        self.delta = int(delta)#时间切刀
        self.is_time_node = False #是否盘口的时间为正好处于周期节点
        
class T():#周期
    def __init__(self, one_stream:Union[Tick,Snapshot]) -> None:
        self.stream = one_stream
        self.start : pd.Timestamp
        self.end : pd.Timestamp
        self.period : int

    def cal_T(self):
        time = self.stream.Time
        opentime = str(time.year)+str(time.month).zfill(2)+str(time.day).zfill(2)+str(092500000)
        openqu = pd.to_datetime(opentime, format='%Y%m%d%H%M%S%f')
        Delta = pd.Timedelta(time - openqu)
        period = pd.Timedelta(self.stream.delta, unit='microseconds')
        Delta_float = int(Delta.total_seconds()) + float(Delta.microseconds)
        period_float = int(period.total_seconds())
        cont = Delta_float // period_float
        self.period = int(cont)
        self.stream.T = int(cont) #指针 tick/snapshot的周期
        self.start = openqu + cont * period
        self.end = openqu + (cont + 1) * period
        tolerance = 1e-9  # 容差值
        if abs(Delta_float % period_float) < tolerance: #位于周期节点
            self.stream.is_time_node = True
        
            
class StreamDict(): 
    """dic: dict[int,Union[list, Snapshot]]"""
    def __init__(self):
        self.dic = dict()
        self.max_T_dict = 0#字典的最大周期
        self.old_T_dict = 0#字典的老周期
    
    def append(self, one_stream:Union[Tick,Snapshot]):
        """逻辑说明:
        第一周期的1进来, old = 0, max = 1
        第一周期的2进来, old = 1, max = 1 
        第一周期的3进来, old = 1, max = 1
        第二周期的1进来, old = 1, max = 2
        以此类推, 我们只有当dict的old != max的tick进来, 即一个新的周期刚被创建时, 才计算之前老周期所对应的一些指标

        Args:
            one_tick (Tick): _description_
            
        """
        if isinstance(one_stream,Tick): #StreamDict为存放ticks的字典
            if one_stream.T not in self.dic:#如果是这个键是空的,添加列表
                if (one_stream.T - self.max_T_dict) > 1: #如果出现周期跳跃
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
        elif isinstance(one_stream, Snapshot):    #StreamDict为存放snapshot的字典
            if one_stream.is_time_node == True:
                self.dic[one_stream.T] = one_stream
        else:
            print("流式输入数据类型有误")

class TurnOver():
    """秒级换手率

    Args:
        是RiskType的子类
    cal_turnover: 计算当前周期的换手率,并记录当前周期的成交量
    """
    def __init__(self, Dic:StreamDict):
        """
        Args:
            tick 全部的tick数据类实例化对象
        """
        self.stream = Dic
        self.dict = dict()
        self.volume_dict = dict()
        
    def cal_turnover(self, whole, i):#whole是流通股, i是所计算的to的周期
        j:Tick#类型注释
        qty = 0
        for j in list(self.stream.dic[i]):
            qty += j.Qty
        self.volume_dict[i] = qty
        self.dict[i] = qty / whole#换手率储存在dict里

class Ret():
    """收益率
    
    cal_ret：计算当前周期的收益率
    """
    def __init__(self, Dic:StreamDict):
        """初始化

        Args:
            Dic (StreamDict): 全部的snapshot实例化对象
        """
        self.stream = Dic
        self.price_dict = dict()
        self.dict = dict()
    def _cal_price(self,i):
        self.price_dict[i] = self.stream.dic[i].ask1 + self.stream.dic[i].bid1
    def cal_ret(self, i): #i为计算return的周期
        #计算收益率
        if i > 0 and i in self.price_dict and i - 1 in self.price_dict and self.price_dict[i-1] != 0:
            self.dict[i] = self.price_dict[i] / self.price_dict[i-1]
        else:
            self.dict[i] = None
            
class Info():
    def __init__(self, turnover:TurnOver):
        self.dict = dict()
        self.stream = turnover.volume_dict
        self.mean = None #记录上一次计算的mean
        self.std = None #记录上一次计算的std
        self.first_volume = None #上一次计算的最初周期的volume
        self.last_volume = None  #本次计算的最后周期的volume
        self.var = None
        
    def cal_info(self, i, period = 10):
        #i为计算info的周期,period为计算std的周期
        if i in self.stream: #已有第i期ret
            if (i - period + 1) in self.stream: #可回看perioed期数据计算std
                if (i - period) not in self.stream: #第一个可算std的周期
                    self.first_volume = self.stream[i - period + 1] #存入第一个volume 用于增量计算
                    data_values = list(self.stream.values())  # 将字典的值转换为列表
                    self.std = np.std(data_values) 
                    self.var = self.std**2 #初始化：第一个方差
                    self.mean = np.mean(data_values)
                    self.dict[i] = self.std / self.mean                 
                else:
                    #增量法计算标准差
                    self.last_volume = self.stream[i]
                    self.mean = self.mean + ((self.last_volume - self.first_volume) / (period - 1)) #无偏估计
                    self.var = self.var - (self.first_volume-self.mean)**2/(period - 1) + (self.last_volume-self.mean)**2/(period - 1)
                    self.std = math.sqrt(self.var)
                    self.dict[i] = self.std / self.mean  
                    #更改first_volume为本次第一个周期的volume
                    self.first_volume = self.stream[i - period + 1]
            else:
                self.dict[i] = None
        else:
            print("error:要计算当期info，请先更新ret字典")
               
        
class Hurst():
    def __init__(self):
        self.hurst : int
        #TODO
        
        
        
        
class factor1():
    def __init__(self):       
        self.factor1_dict = dict()
    def cal_factor1(self, i:int, ret:Ret, turnover:TurnOver, delta=5):
        """i为当前周期,delta为turnover排序的周期"""
        #基于换手率对ret排序
        #先获取排序后的周期list
        start_period =  max(i - delta + 1, 1)
        sorted_turnover_periods = sorted(range(start_period, i + 1), key=lambda x: turnover.dict[x] if x in turnover.dict else -1)
        if len(sorted_turnover_periods) < delta:
            self.factor1_dict[i] = None
        else:
            self.factor1_dict[i] = ret.dict[sorted_turnover_periods[-1]] -  ret.dict[sorted_turnover_periods[0]]
        #根据这个排序生成相应的factor
        
class factor2():
    def __init__(self):       
        self.factor2_dict = dict()
    def cal_factor1(self, i:int, ret:Ret, info:Info, delta=5):
        """i为当前周期,delta为info排序的周期"""
        #基于info对ret排序
        #先获取排序后的周期list
        start_period =  max(i - delta + 1, 1)
        sorted_info_periods = sorted(range(start_period, i + 1), key=lambda x: info.dict[x] if x in info.dict else -1)
        if len(sorted_info_periods) < delta:#周期数不够
            self.factor2_dict[i] = None
        else:
            self.factor2_dict[i] = ret.dict[sorted_info_periods[-1]] -  ret.dict[sorted_info_periods[0]]
        #根据这个排序生成相应的factor
                        
class Model_reverse(modelType):
   
    def __init__(self, Risk_d = 5, Umr_m = 20, H = 10, delta_stream = 40000000, whole = 294e10) -> None:
        #一开始初始化的时候先把这些实例化，然后每次update就更新这里的属性
        self.TD = StreamDict()#tickdict
        self.RD = StreamDict()#pricedict
        self.TO = TurnOver(self.TD)#计算换手率，储存换手率
        self.stockret = Ret(self.TD)#计算股票收益率，储存
        self.whole = whole#总流通股本

        self.delta = delta_stream
        self.H = H
        self.Umr_m = Umr_m
    
    def _error(self):
        #报错并跳过
        print("This stream data can't calculate all indexes!")
  
    def _tick_store(self,one_tick):
        #每次进来一条tick，就调用这个函数,将tick的信息进行预处理，并储存在self中的属性中
        one_tick = Tick(one_tick, delta = self.delta)#初始化tick对象
        T_tick = T(one_tick)#计算tick的周期
        T_tick.cal_T()
        self.TD.append(one_tick)#将这单tick添加到TD中
        
    def _snapshot_store(self,one_snapshot):
        #每次进来一条snapshot，就调用这个函数,将snapshot的信息进行预处理，并储存在self中的属性中
        T_ret= T(one_snapshot)#计算snapshot的周期
        T_ret.cal_T()
        self.RD.append(one_snapshot)#将价格添加到RD中
        #这个函数只做一件事，那就是更新TD和RD
    
    def _cal_turnover(self, i):
        self.TO.cal_turnover(self.whole, i)#计算这个周期的turnover
    
    #TODO 把strategy的self.timestamp作为当前时间输入
    def model_update(self, ticks, one_mktstream,timestamp:int):
        #先储存
        #timestamp 17位int
        date_today = str(timestamp)[:4]+'-'+str(timestamp)[4:6]+'-'+str(timestamp)[6:8]
        time_now = pd.to_datetime(str(timestamp), format='%Y%m%d%H%M%S%f')
        tickdict = ticks[date_today] #到目前为止所有时刻的ticks
        tickdict_now = tickdict[int(time_now.strftime('%H:%M:%S:%f')[:-3])]
        for one_tick in tickdict_now:
            self._tick_store(one_tick)
            #1. 先要判断进来的stream是不是填满了这个周期，如果没填满就不开始计算
            if self.TD.max_T_dict == self.TD.old_T_dict :
                #此时这个周期的stream未必都来了
                return 
            for i in range(self.TD.old_T_dict + 1, self.TD.max_T_dict + 1):
                #计算turnover不需要跳过任何周期，最先计算
                self._cal_turnover(i)
                #2. 判断是否满足计算收益率的周期数
                if self.TD.max_T_dict > 1:
                    break
                #计算收益率
                self._cal_mktret(i)
                self._cal_stockret(i)
                #计算umr
                self._cal_umr(i)
        