from abc import ABC, abstractmethod
from pandas import DataFrame


class algorithmicStratgy(ABC):
    """
    抽象基类, 定义算法交易类的接口, 有如下属性
    oderbook: oderbook类, 可以撮合盘口, 记录盘口状态
    timeStamp: 记录当前时间戳
    deal: 成交记录
    possession: 调仓记录
    """

    oderbook: "oderbook"  # oderbook类, 还没定义
    timeStamp: int
    deal: DataFrame
    possession: DataFrame

    @abstractmethod
    def TWAP_stratgy() -> dict:
        """
        TWAP策略,返回一个signal字典, 包含{股票代码, 买入还是卖出["B"/"S"], 价格, volume}
        传入参数: lines:tick/order的一条数据, volume: 需要下单的总量

        """

    @abstractmethod
    def VWAP_stratgy() -> dict:
        """
        TWAP策略,返回一个signal字典, 包含{股票代码, 买入还是卖出["B"/"S"], 价格, volume}
        传入参数: lines:tick/order的一条数据, volume: 需要下单的总量
        """

    @abstractmethod
    def momentum_stratgy() -> dict:
        """
        动量策略,返回胜率、赔率、换手率。其余返回参数待定
        传入参数:  lines:tick/order的一条数据
        """

    @abstractmethod
    def snapshot_stratgy() -> dict:
        """
        盘口策略, 返回策略结果, 需要l2snapshot数据
        """
