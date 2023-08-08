from abc import abstractmethod, ABC
from reverse_factor import *

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
    def model_update(self,tickdict,orderbook):
        """
        基于更新的数据计算新的反转因子，返回一个因子字典
        """
        #tick的key是str，需要改成pd.Timestamp
        new_tickdict = {pd.Timestamp(pd.to_datetime(k, format='%Y%m%d%H%M%S%f')): v for k, v in tickdict.items()}
        time_now = max(new_tickdict.keys())
        time_now_int = int(list(tickdict.keys())[-1])
        # 生成时间列表
        backtrack_minutes = pd.Timedelta(minutes=30)
        interval = pd.Timedelta(seconds=3)
        backtrack_time = time_now - backtrack_minutes
        time_list = sorted(pd.date_range(start=backtrack_time, end=time_now, freq=interval).tolist())
        
        #从盘口信息得到price,作为计算ret的输入
        #TODO：tick时间和snapshot不同步
        price_list = [0]*len(time_list)
        for i in range(len(time_list)):
            ask_1,volume_ask = list(orderbook.snapshots.values())[-i].ask.popitem()
            bid_1,volume_bid = list(orderbook.snapshots.values())[-i].bid.popitem()
            price_list[i] = ask_1*volume_ask/(volume_ask+volume_bid)+bid_1*volume_bid/(volume_bid+volume_ask)
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

