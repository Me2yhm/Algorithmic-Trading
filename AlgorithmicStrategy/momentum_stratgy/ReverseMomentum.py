import pandas as pd
from abc import abstractmethod, ABC
from typing import Dict, List, TypedDict,Union
from collections import OrderedDict, deque
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
    def __init__(self, one_stream:Union[Tick,Snapshot]):
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
            """snapshot比tick的字典多一行！dict存放的是周期开始的price，收盘时的price对应一个新的周期"""
            
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
        self.dict = dict() #ret_dict
    def _cal_price(self,i):
        self.price_dict[i] = self.stream.dic[i].ask1 + self.stream.dic[i].bid1
    def cal_ret(self, i): #i为计算return的周期
        #计算收益率
        if i >= 0 and i in self.price_dict and i + 1 in self.price_dict and self.price_dict[i] != 0:
            self.dict[i] = self.price_dict[i+1] / self.price_dict[i]
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
    def __init__(self, k=6):
        self.k = k #将ret_series分成k组来回归
        self.hurstdict = dict()
        self.stream = None 
    def cal_hurst(self, i, ret_dict, period = 128):
        #period最好取2的幂次
        if (i-period+1) < 0: #周期不满
            self.hurstdict[i] = None
        else:
            start_time = i - period + 1
            self.stream = {x: ret_dict[x] for x in ret_dict if start_time <= x <= i} #ret_dict为ret的字典{int：float},截取周期长度为period
            ret_series = pd.Series(self.stream, index = list(self.stream.keys()))
            #计算第i期的hurst指数
            RS = [0]*self.k #initialize R/S-index
            size_list = [0]*self.k
            for j in range(self.k):            
                #第j种划分，即划分为2**j组
                #TODO
                size_list[j] = period / (2**j) #第j种划分方式下，一组的周期数目
                #计算每组的R/S值
                # for number in range(2**j):   #字典好像没有pd.Series方便
                    #number为j划分的第几组
                    # temp_end_period = list(ret_dict.keys())[-1] - number*size_list[j]   #第number组的起始周期数
                    # temp_start_period = temp_end_period - size_list[j]                  #第number组的结束周期数
                    # selected_items = {key: value for key, value in self.stream.items() if temp_start_period <= key <= temp_end_period}
                subseries_index_list = np.array_split(ret_series.index, 2**j)  
                count = 0  
                for s in range(2**j):
                    series = pd.Series(ret_series[subseries_index_list[s]], index=ret_series.index[:len(subseries_index_list[s])])
                    std = series.std()
                    mean = series.mean()
                    if np.isnan(std) or np.isnan(mean):
                        continue
                    else:
                        series_delta = series.apply(lambda x: x-mean)
                        R = series_delta.max() - series_delta.min()
                        # breakpoint()
                        RS[j] += R/std
                        count +=1
                    RS[j] = RS[j]/2**count
                    #去掉RS的0值，对R/S值和k回归，取系数为hurst
            RS_new = size_list_new = []
            for j in range(len(RS)):
                if RS[j] !=0:
                    RS_new.append(RS[j])
                    size_list_new.append(len(ret_series)/(2**j))
            RS_new = np.array(RS_new)
            size_list_new = np.array(size_list_new)
            if len(RS_new)!=0:
                self.hurstdict[i] = np.polyfit(np.log(size_list_new), np.log(RS_new), 1)[0]
            else:
                self.hurstdict[i] = None
            
        
class Factor1():
    """
    计算基于换手率排序的反转因子
    """
    def __init__(self):       
        self.factor1_dict = dict()
    def cal_factor1(self, i:int, ret:Ret, turnover:TurnOver, delta=5):
        """i为当前周期,delta为turnover排序的周期"""
        #基于换手率对ret排序
        #先获取排序后的周期list
        if i - delta + 1 < 0: #周期不足
            self.factor1_dict[i] = None
        else:
            start_period =  i - delta + 1
            sorted_turnover_periods = sorted(range(start_period, i + 1), key=lambda x: turnover.dict[x])
            self.factor1_dict[i] = ret.dict[sorted_turnover_periods[-1]] -  ret.dict[sorted_turnover_periods[0]]
        #根据这个排序生成相应的factor
        
class Factor2():
    """计算基于信息分布排序的反转因子
    """
    def __init__(self):       
        self.factor2_dict = dict()
    def cal_factor2(self, i:int, ret:Ret, info:Info, delta=5):
        """i为当前周期,delta为info排序的周期"""
        #基于info对ret排序
        #先获取排序后的周期list
        if i - delta + 1 < 0: #周期不足            
            self.factor2_dict[i] = None
        else:
            start_period =  i - delta + 1
            sorted_info_periods = sorted(range(start_period, i + 1), key=lambda x: info.dict[x])
            self.factor2_dict[i] = ret.dict[sorted_info_periods[-1]] -  ret.dict[sorted_info_periods[0]]
        #根据这个排序生成相应的factor
                        
class Model_reverse(modelType):
   
    def __init__(self, delta_stream=40000000,std_period=10, hurst_period=128, k=6, whole=294e10,delta_factor1=5,delta_factor2=5):
        #一开始初始化的时候先把这些实例化，然后每次update就更新这里的属性
        self.TD = StreamDict()#tickdict
        self.SD = StreamDict()#snapshotdict
        self.TO = TurnOver(self.TD)#计算换手率，储存换手率
        self.ret = Ret(self.TD)#计算股票收益率，储存
        self.info = Info(self.TO)
        self.k = k #hurst计算时的分组数目 
        self.hurst = Hurst(self.k)
        self.factor1 = Factor1()
        self.factor2 = Factor2()
        self.delta1 = delta_factor1
        self.delta2 = delta_factor2
        self.whole = whole#总流通股本
        self.delta = delta_stream #周期间隔 以微秒为单位
        self.info_period = std_period #计算波动率的周期数
        self.hurst_period = hurst_period #计算hurst的周期数
    
    def _error(self):
        #报错并跳过
        print("This stream data can't calculate all indexes!")
  
    def _tick_store(self,one_tick):
        #每次进来一条tick，就调用这个函数,将tick的信息进行预处理，并储存在self中的属性中
        one_Tick = Tick(one_tick, delta = self.delta)#初始化tick对象
        T_tick = T(one_Tick)#计算tick的周期
        T_tick.cal_T()
        self.TD.append(one_tick)#将这单tick添加到TD中
        
    def _snapshot_store(self,one_snapshot):
        #每次进来一条snapshot，就调用这个函数,将snapshot的信息进行预处理，并储存在self中的属性中
        one_Snapshot = Snapshot(one_snapshot, delta = self.delta)
        T_snapshot= T(one_Snapshot)#计算snapshot的周期
        T_snapshot.cal_T()
        self.SD.append(one_snapshot)#将价格添加到SD中
        #这个函数只做一件事，那就是更新TD和SD
    
    def _cal_turnover(self, i):
        self.TO.cal_turnover(self.whole, i)#计算这个周期的turnover
    
    def _cal_ret(self,i):
        if i == 0: #计算第0个周期开始和结束的价格
            self.ret._cal_price(0)
            self.ret._cal_price(1)
        else:
            self.ret._cal_price(i+1) #计算这个周期结束（=下个周期开始）的价格
        self.ret.cal_ret(i) #计算这个周期的ret
        
    def _cal_info(self,i):
        self.info.cal_info(i, period = self.info_period)
        
    def _cal_hurst(self,i):
        self.hurst.cal_hurst(i,self.ret.dict,self.hurst_period)
        
    def _cal_factor1(self,i):
        self.factor1.cal_factor1(i, self.ret, self.TO, self.delta1)
        
    def _cal_factor2(self,i):
        self.factor2.cal_factor2(i,self.ret,self.info,self.delta2)
        
    #TODO 把strategy的self.timestamp作为当前时间输入
    def model_update(self, ticks, orderbook:OrderBook,timestamp:int):
        #先储存
        #timestamp 17位int
        date_today = str(timestamp)[:4]+'-'+str(timestamp)[4:6]+'-'+str(timestamp)[6:8]
        time_now = pd.to_datetime(str(timestamp), format='%Y%m%d%H%M%S%f')
        tickdict = ticks[date_today] #到目前为止所有时刻的ticks
        tickdict_now = tickdict[int(time_now.strftime('%H:%M:%S:%f')[:-3])]
        
        #0. 存入one_tick和one_snapshot
        deque_dict = deque(orderbook.snapshots.items())
        last_element = deque_dict.pop() #返回一个元组：（time，snapshot）
        one_snapshot = last_element[1]
        self._snapshot_store(one_snapshot) 
        for one_tick in tickdict_now:
            self._tick_store(one_tick) #把一个时间戳的ticks一条条变为Tick对象存入TD
            
        #1. 先要判断进来的stream是不是填满了这个周期，如果没填满就不开始计算    
        if self.TD.max_T_dict == self.TD.old_T_dict :
            #此时这个周期的stream未必都来了
            return {"factor1":self.factor1.factor1_dict,"factor2":self.factor2.factor2_dict,"hurst":self.hurst.hurstdict}
        for i in range(self.TD.old_T_dict, self.TD.max_T_dict): 
            #计算各指标
            self._cal_turnover(i)
            self._cal_ret(i)
            self._cal_info(i)
            #计算反转因子和hurst
            self._cal_factor1(i)
            self._cal_factor2(i)
            self._cal_hurst(i)
            return {"factor1":self.factor1.factor1_dict,"factor2":self.factor2.factor2_dict,"hurst":self.hurst.hurstdict}
