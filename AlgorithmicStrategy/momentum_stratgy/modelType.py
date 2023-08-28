from abc import abstractmethod, ABC
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

        
