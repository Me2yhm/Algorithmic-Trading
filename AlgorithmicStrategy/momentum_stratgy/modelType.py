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
        
        
class model_reverse(modelType):
    def model_update(self,tickdict,orderbook):
        """
        基于更新的数据计算新的反转因子，返回一个因子字典
        """
        time_now = max(tickdict.keys())
        
        
        #TODO:从盘口信息得到price,作为计算ret的输入
        # time_list = list(tickdict.keys())
        # price_list = [0]*len(time_list)
        # for i in range(len(time_list)):
        #     ask_1,volume_ask = next(iter(orderbook.snapshots.ask.items()))
        #     bid_1,volume_bid = next(iter(orderbook.snapshots.bid.items()))
        #     price_list[i] = ask_1*volume_ask/(volume_ask+volume_bid)+bid_1*volume_bid/(volume_bid+volume_ask)
        # price_list = pd.Series(price_list,index=time_list)
        
        
        ret_series = Ret(time_now,price_list).form_series()
        turnover_series = Turnover(time_now,tickdict).form_series()
        hurst = Hurst(ret_series).calculate() 
        info_series = Information(time_now,tickdict).form_series()
        factor_1 = Factor_turnover_based(ret_series, turnover_series).calculate() 
        factor_2 = Factor_information_based(ret_series,info_series).calculate()
        index_dict = {'hurst':hurst,'factor_turnover':factor_1,'factor_info':factor_2}
        return index_dict        

