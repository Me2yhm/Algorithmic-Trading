from abc import ABC, abstractmethod
from AlgorithmicStrategy.base import AlgorithmicStrategy


class modelType(ABC):
    """
    模型的抽象基类，用于实现指标更新或者模型增量训练
    """

    @abstractmethod
    def model_update():
        """
        数据增量地传进来，每传进来一条数据，模型计算一次指标或者训练一次模型
        """


class momentumStratgy(AlgorithmicStrategy):
    """
    动量算法类, 有如下属性
    orderbook: orderbook类, 可以撮合盘口, 记录盘口状态
    timeStamp: 记录当前时间戳
    deal: 成交记录
    possession: 调仓记录
    """

    def update_oderbook(self) -> None:
        """
        流式读取数据，更新盘口
        """
        pass

    def model_update(model) -> None:
        """
        盘口更新过后，根据更新过的数据增量地更新指标或者训练模型
        """
        pass

    def signal_update(self) -> dict:
        """
        调用model_update函数，根据函数结果返回信号
        """
        pass

    def exam_update(self) -> dict:
        """
        根据返回的信号计算胜率、赔率、换手率等——可以流式？
        """
        pass
