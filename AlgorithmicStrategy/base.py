from abc import ABC, abstractmethod

from pandas import DataFrame
from typing import Dict, List


class OrderBook(ABC):
    pass


class AlgorithmicStrategy(ABC):
    """
    抽象基类, 定义算法交易类的接口, 有如下属性
    orderbook: orderbook类, 可以撮合盘口, 记录盘口状态
    timeStamp: 记录当前时间戳
    deal: 成交记录
    possession: 调仓记录
    """

    # orderbook类, 还没定义
    orderbook: OrderBook
    tick: List[Dict]
    signal: List[Dict]
    timeStamp: int
    deal: DataFrame
    possession: DataFrame

    @abstractmethod
    def update_orderbook(self) -> dict:
        """
        更新orderbook和tick数据

        """

    @abstractmethod
    def model_update(self) -> None:
        """
        更新模型，数据流式传入，对模型进行流式训练和更新
        """

    @abstractmethod
    def signal_update(self) -> Dict:
        """
        模型更新之后，根据模型训练结果更新signal，
        signal是一个元素为字典的列表，每个signal包含{股票代码, 买入还是卖出["B"/"S"], 价格, volume}
        """

    @abstractmethod
    def stratgy_update(self) -> Dict:
        """
        根据更新过后的signal，更新成交记录和持仓记录，更新策略的评价结果
        动量是胜率、赔率，VWAP是成交成本与实际VWAP的差
        """
