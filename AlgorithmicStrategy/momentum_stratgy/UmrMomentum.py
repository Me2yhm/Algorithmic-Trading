import pandas as pd
from .modelType import modelType

class AbstractStream():
    #抽象流式数据基类
    def __init__(self, stream) -> None:
        self.Price : float
        self.Qty : int
        self.Amo : float
        self.Time : pd.Timestamp
        self.delta : int
        self.T : int
    
class Tick(AbstractStream):
    """tick数据流式传入一tick的信息
    """
    
    def __init__(self, one_tick, delta=500):
        """初始化
            one_tick流式输入一tick
            delta 一周期的时间
        """
        self.ID = one_tick[0]
        self.Time = pd.to_datetime(one_tick[1], format='%Y%m%d%H%M%S%f')#这里可能必须转换成pd时间
        self.Price = float(one_tick[2])
        self.Qty = int(one_tick[3])
        self.Amo = float(one_tick[4])
        self.BuyNO = one_tick[5]
        self.SellNo = one_tick[6]
        self.Index = one_tick[7]
        self.ChannelNo = one_tick[8]
        self.BSFlag = one_tick[9]
        self.BizIndex = one_tick[10]
        self.delta = int(delta)#时间切刀
        self.T : int
        #pd.Timedelta((self.Time - pd.to_datetime(20230228092500000, format='%Y%m%d%H%M%S%f')), unit='microseconds')
        #这里要把时间全部转换成毫秒再相除
        #所属周期，这里必须需要转换成pd时间
    
class MktStream(AbstractStream):##接收公司的沪深300指数行情
    def __init__(self, mkt:dict, delta=50000) -> None:
        self.Price = mkt['f45']
        self.Time = mkt['f86']
        self.delta : int
        self.T : int

class T():#周期
    def __init__(self, one_stream:AbstractStream) -> None:
        self.stream = one_stream
        self.start : pd.Timestamp
        self.end : pd.Timestamp
        self.period : int
    
    def cal_T(self):
        time = pd.to_datetime(self.stream.Time, format='%Y%m%d%H%M%S%f')
        openqu = pd.to_datetime(20230228092500000, format='%Y%m%d%H%M%S%f')
        Delta = pd.Timedelta(time - openqu)
        period = pd.Timedelta(self.stream.delta, unit='microseconds')
        cont = -1
        while{True}:
            
            Delta -= period
            cont += 1
            if int(Delta.days) < 0:
                break
        self.period = cont
        self.stream.T = cont
        self.start = openqu + cont * period
        self.end = openqu + (cont + 1) * period
      
#先初始化这个Dict类，然后不断往里面append一单stream
class StreamDict():
    
    dic: dict[int,list]
    def __init__(self):
        self.dic = dict()
        self.max_T_dict : int#字典的最大周期
        self.old_T_dict  : int#字典的老周期
    
    def append(self, one_stream:AbstractStream):
        """逻辑说明:
        第一周期的1进来, old = 0, max = 1
        第一周期的2进来, old = 1, max = 1 
        第一周期的3进来, old = 1, max = 1
        第二周期的1进来, old = 1, max = 2
        以此类推, 我们只有当dict的old != max的tick进来, 即一个新的周期刚被创建时, 才计算之前老周期所对应的一些指标

        Args:
            one_tick (Tick): _description_
        """
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

class RiskType():
    """
    抽象类，用于区分不同的风险变量
    """
    #下面是所有风险变量必须有的结构
    def __init__(self, Dic:StreamDict) -> None:
        """初始化

        Args:
            tick 全部的tick数据类实例化对象
        """
        self.tick = Dic#tick数据的dict
        self.dict = dict()#用于储存时间周期对应的风险变量值

#todo 集合竞价集中成交的一批会体现在第一个周期里，有没有好的解释？？？
class TurnOver(RiskType):
    """秒级换手率

    Args:
        是RiskType的子类
    cal_turnover: 计算当前周期的换手率
    """
    def __init__(self, Dic:StreamDict) -> None:
        """初始化

        Args:
            tick 全部的tick数据类实例化对象
        """
        self.stream = Dic
        self.dict = dict()
    
    def cal_turnover(self, whole, i):#whole是流通股, i是所计算的to的周期
        j:Tick#类型注释
        qty = 0
        for j in list(self.stream.dic[i]):
            qty += j.Qty
        self.dict[i] = qty / whole#换手率储存在dict里

class Risk():
    def __init__(self, d:int, type:RiskType):
        """初始化风险系数

        Args:
            d 风险系数回看时间
            type (RiskType): 风险变量的类型, 是一个RiskType实例
            tick 数据类实例
        """
        self.d = d
        self.risk_variable = type 
        self.dic = dict()
        self.new_risk : float
    
    def cal_risk(self, i):
        """计算当前时间周期的risk, 并存入dic中

        Args:
            i (_type_): 当前时间周期
        """
        if i == 50:
            sum = 0
            for j in range(i - self.d + 1, i + 1):
                sum += self.risk_variable.dict[j]
            self.new_risk = (sum / self.d - self.risk_variable.dict[i]) * 10e8
            self.dic[i] = self.new_risk
        else:#增量形式
             self.new_risk = self.new_risk + self.risk_variable.dict[i] - self.risk_variable.dict[i + 1] - 1 / self.d * (self.risk_variable.dict[i - self.d] - self.risk_variable.dict[i + 1])
             self.dic[i] = self.new_risk
        
class Ret():
    #统一计算股票和指数的收益率
    def __init__(self, dic:StreamDict) -> None:
        self.dict = dic
        self.reSdict = dict()
        self.weireSdict = dict()
        self.lnreSdict = dict()
    
    #这个不怎么用
    def cal_ret(self, i):#i表示这一周期的股票收益率
        #使用周期内最后一笔成交单作为股票当前周期的股价(模拟日线的收盘价)
        #这里要处理一下如果是该时期一单tick也没有的情形
        stream_i:AbstractStream
        stream_i_1:AbstractStream
        if self.dict.dic[i] == [] or self.dict.dic[i - 1] == []:
            #只要有一个是空列表，这个时间段的收益率都设置成0
            self.reSdict[i] = 0
        elif self.dict.dic[i] is not [] and self.dict.dic[i - 1] is not []:
            stream_i = self.dict.dic[i][-1]
            stream_i_1 = self.dict.dic[i][-1]
            self.reSdict[i] = stream_i.Price / stream_i_1.Price - 1

    def cal_weighted_ret(self, i):
        streams_i:list[Tick]
        streams_i_1:list[Tick]
        #if i == 188:
        #    h = 0
        if self.dict.dic[i] == [] or self.dict.dic[i - 1] == []:
            #只要有一个是空列表，这个时间段的收益率都设置成0
            self.weireSdict[i] = 0
        elif self.dict.dic[i] is not [] and self.dict.dic[i - 1] is not []:
            streams_i = self.dict.dic[i]
            streams_i_1 = self.dict.dic[i - 1]
            sum_i = 0
            sum_i_1 = 0
            for tick in streams_i:
                sum_i += tick.Qty
            for tick in streams_i_1:
                sum_i_1 += tick.Qty
            price_i = 0
            price_i_1 = 0
            for stream in streams_i:
                price_i += stream.Amo
            price_i = price_i / sum_i
            for stream in streams_i_1:
                price_i_1 += stream.Amo
            price_i_1 = price_i_1 / sum_i_1
            self.weireSdict[i] = price_i / price_i_1 - 1
                
    def cal_ln_ret(self):
        pass

class TimeWeight():
    #时间权重，每次计算生成一个对象
    def __init__(self, m, H) -> None:
        self.m = m
        self.H = H
        self.time_weigt_dict =dict()
        self.sum = self._cal_sum()
        
    def _cal_sum(self):
        sum = 0
        for j in range(1, self.m + 1):
            sum += 2 ** (- j / self.H)
        return sum 
    
    def cal_time_weight(self, i):
        for j in range(1, self.m + 1):
            self.time_weigt_dict[j + i - self.m] = 2 ** (-(self.m - j + 1) / self.H) / self.sum
            
class UMR():
    def __init__(self) -> None:
        self.umrdict = dict()#这个是不变的
        self.new_umr : float
        self.firt : float
    """
    def _cal_time_weight(self, i):
        sum = 0
        if i == 188:
            h = 0
        for j in range(1, self.m + 1):
            sum += 2 ** (- j / self.H)
        for j in range(i - self.m + 1, i + 1):
            self.time_weight[j] = (2 ** (- (self.m - j + 1) / self.H) / sum)
    """
    def cal_UMR(self,i, risk:Risk, stockret:Ret, mktret:Ret, time_weight:TimeWeight):
        """
            i是当前计算umr的周期
        """
        if i == 80:
            
            umr = 0
            for j in range(i - time_weight.m + 1, i + 1):
                umr += risk.dic[j] * time_weight.time_weigt_dict[j] * (stockret.weireSdict[j] - mktret.weireSdict[j])
            self.new_umr = umr
            self.umrdict[i] = self.new_umr
        else:
            self.new_umr = self.new_umr - risk.dic[i - time_weight.m] * time_weight.time_weigt_dict[i - time_weight.m] * (stockret.weireSdict[i - time_weight.m] - mktret.weireSdict[i - time_weight.m]) + risk.dic[i] * time_weight.time_weigt_dict[i] * (stockret.weireSdict[i] - mktret.weireSdict[i])
            self.umrdict[i] = self.new_umr
                                        
   
class UMRMonmentum(modelType):
    
    def __init__(self, Risk_d = 5, Umr_m = 20, H = 20, delta_stream = 40000000, whole = 294e10) -> None:
        #一开始初始化的时候先把这些实例化，然后每次update就更新这里的属性
        self.TD = StreamDict()#tickdict
        self.MktD = StreamDict()#mktdict
        self.TO = TurnOver(self.TD)#计算换手率，储存换手率
        self.risk_turnover = Risk(Risk_d, self.TO)#计算换手率风险系数
        self.stockret = Ret(self.TD)#计算股票收益率，储存
        self.mktret = Ret(self.MktD)#计算mkt收益率，储存
        self.whole = whole#总流通股本
        self.umr = UMR()#初始化UMR对象
        self.delta = delta_stream
        self.H = H
        self.Umr_m = Umr_m
    
    def _error(self):
        #报错并跳过
        print("This stream data can't calculate all indexes!")
        
        
    
    def _stream_store(self, one_tick, one_mktstream):
        #每次进来一条tick或者mktstream，就调用这个函数,将tick和mktstream的信息进行预处理，并储存在self中的属性中
        one_tick = Tick(one_tick, delta = self.delta)#初始化tick对象
        one_mktstream = MktStream(one_mktstream, delta = self.delta)#初始化mktstream对象
        T_tick = T(one_tick)#计算tick的周期
        T_tick.cal_T()
        T_mktstr = T(one_mktstream)#计算mktstream的周期
        T_mktstr.cal_T()
        self.TD.append(one_tick)#将这单tick添加到TD中
        self.MktD.append(one_mktstream)#将这单mktstream添加到MktD中
        #这个函数只做一件事，那就是1. 更新TD和MktD
    
    def _cal_turnover(self, i):
        self.TO.cal_turnover(self.whole, i)#计算这个周期的turnover
    
    def _cal_risk(self, i):
        self.risk_turnover.cal_risk(i)#计算这个周期的风险系数
    
    def _cal_stockret(self, i):
        self.stockret.cal_weighted_ret(i)#计算这个时期的收益率
    
    def _cal_mktret(self, i):
        self.mktret.cal_weighted_ret(i)#计算这个时期的收益率
        pass
    
    def _cal_umr(self, i):
        time = TimeWeight(self.Umr_m, self.H)
        time.cal_time_weight(i)
        self.umr.cal_UMR(i, self.risk_turnover, self.stockret, self.mktret, time)
        
    
    def model_update(self, one_tick, one_mktstream):
        #先储存
        self._stream_store(one_tick, one_mktstream)
        #1. 先要判断进来的stream是不是填满了这个周期，如果没填满就不开始计算
        if self.TD.max_T_dict == self.TD.old_T_dict or self.MktD.max_T_dict == self.MktD.old_T_dict:
            #此时这个周期的stream未必都来了
            self._error()
            return 
        for i in range(self.TD.old_T_dict + 1, self.TD.max_T_dict + 1):
            #计算turnover不需要跳过任何周期，最先计算
            self._cal_turnover(i)
            #2. 判断是否满足计算收益率的周期数
            if self.TD.max_T_dict > 1:
                self._error()
                break
            #计算收益率
            self._cal_mktret(i) 
            self._cal_stockret(i)
            #3. 判断是否满足计算风险系数的周期数
            if self.TD.max_T_dict < 50:
                self._error()
                break
            #计算风险系数
            self._cal_risk(i)
            #4. 判断是否满足计算时间权重的周期数
            if self.TD.max_T_dict < 80:
                self._error()
                break
            #计算umr
            self._cal_umr(i)
        