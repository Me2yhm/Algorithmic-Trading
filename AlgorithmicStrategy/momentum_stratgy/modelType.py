from abc import abstractmethod, ABC
from .reverse_factor import *
from ..OrderMaster.OrderBook import OrderBook
import time

class modelType(ABC):
    """
    模型的抽象基类，用于实现指标更新或者模型增量训练

    """

    @abstractmethod
    def model_update(tick) -> dict[str, float]:
        """
        模型接受一个tick参数(目前还不确定是什么数据类型)，根据tick数据返回模型计算结果
        """
        pass 

        
class Model_reverse(modelType):
    """
   反转因子模型
    """
    def model_update(self,tickdict,orderbook:OrderBook,timestamp):
        """
        基于更新的数据计算新的反转因子，返回一个因子字典
        """

        #tick.time修改为pd.Timestamp格式
        date_today = max(tickdict.keys()) #str
        tick_list_today = tickdict[date_today]
        tick_now = tick_list_today[-1]
        time_str = str(tick_now['time'])
        #time_str 年/月/日/小时/分钟/秒/微秒
        time_now = pd.Timestamp(pd.to_datetime(time_str, format='%Y%m%d%H%M%S%f'))
        
        # 生成时间列表
        backtrack_minutes = pd.Timedelta(minutes=30)
        interval = pd.Timedelta(seconds=3)
        backtrack_time = time_now - backtrack_minutes
        time_list = sorted(pd.date_range(start=backtrack_time, end=time_now, freq=interval).tolist())
        #从盘口信息得到price,作为计算ret的输入
        #TODO：tick时间和snapshot不同步
        price_list = [0]*len(time_list)
        for i in range(len(time_list)):
            time_int = int(time_list[i].strftime('%Y%m%d%H%M%S%f')[-3])
            if time_int in orderbook.snapshots:
                ask_1,volume_ask = orderbook.snapshots[time_int]['ask']
                bid_1,volume_bid = orderbook.snapshots[time_int]['bid']
                price_list[i] = ask_1*volume_ask/(volume_ask+volume_bid)+bid_1*volume_bid/(volume_bid+volume_ask)
            else:
                price_list[i] = None
        price_list = pd.Series(price_list,index=time_list)
   
        #更新各个指标
       
        ret_series = Ret(time_now,price_list,m = pd.Timedelta(seconds=10)).form_series()
        turnover_series = Turnover(time_now,tickdict).form_series()
        hurst = Hurst(ret_series).calculate() 
        info_series = Information(time_now,tickdict).form_series()
        factor_1 = Factor_turnover_based(ret_series, turnover_series).calculate() 
        factor_2 = Factor_information_based(ret_series,info_series).calculate()
        index_dict = {'time':time_now,'hurst':hurst,'factor_turnover':factor_1,'factor_info':factor_2}
        return index_dict        

