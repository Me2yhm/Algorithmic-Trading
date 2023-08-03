from abc import ABC, abstractmethod

from typing import Dict, List
from TWAP_VWAP.OrderMaster.OrderBook import OrderBook


class AlgorithmicStrategy(ABC):
    """
    抽象基类, 定义算法交易类的接口, 有如下属性
    orderbook: orderbook类, 可以撮合盘口, 记录盘口状态
    timeStamp: 记录当前时间戳和上一秒时间戳
    deals: \成交记录,一个由字典组成的列表,
          \字典的形式为: {timestamp:int,symbol:str,direction:str,price:float,volume:int}
    possession: \持仓记录,一个字典
                \字典形式为{code:str,volume:int,averagePrice:float,cost:float}
                \其中cost是总的金额, averagePrice是总金额除以总量
    signal: \交易信号, 由字典组成的列表
            \字典形式为{symbol:str,direction:str,price:float,volume:int}
            \symbol为股票代码, direction表示买入或者卖出: ["B"/"S"]
    tick: \逐笔数据, 由字典组成的列表
          \字典形式为: {time:int,oid:int,oidb:int,oids:int,price:float,volume:int,flag:int,ptype:int}
          \ time表示当前时间戳, oid等键名的含义可以查询: 语雀|技术团队|数据格式知识库|逐笔数据文档
    commission: 手续费, 券商收取, 默认按万分之1.5算
    stamp_duty: 印花税, 买入没有印花税, 卖出有, 为0.001
    transfer_fee: 过户费, 为0.00002, 买卖都有
    """

    orderbook: OrderBook
    tick: List[Dict]
    signal: List[Dict]
    timeStamp: int
    deal: List[Dict]
    possession: Dict
    commission: float
    stamp_duty: float
    transfer_fee: float
    ask_cost: float
    bid_cost: float

    def __init__(
        self,
        commision: float = 0.00015,
        stamp_duty: float = 0.001,
        transfer_fee: float = 0.00002,
    ) -> None:
        self.tick = []
        self.signal = []
        self.timeStamp = 0
        self.deals = []
        self.possession = {"code": "", "volume": 0, "averagePrice": 0.0, "cost": 0.0}
        self.commission = commision
        self.stamp_duty = stamp_duty
        self.transfer_fee = transfer_fee
        self.buy_cost = self.commission + self.transfer_fee
        self.sell_cost = self.commission + self.transfer_fee + self.stamp_duty

    def update_orderbook(self, lines: List[Dict]) -> None:
        """
        更新orderbook和tick数据
        """
        self.tick.extend(lines)
        try:
            assert all([line["timestamp"] for line in lines])
            self.timeStamp = lines[-1]["timestamp"]
            self.orderbook.single_update(lines)
        except AssertionError:
            raise ValueError("lines need the same timestamp")

    @abstractmethod
    def model_update(self) -> None:
        """
        更新模型, 数据流式传入, 对模型进行流式训练和更新
        """

    @abstractmethod
    def signal_update(self) -> None:
        """
        调用model_update函数, 根据模型训练结果更新signal,
        signal是一个元素为字典的列表, 每个signal包含{股票代码, 买入还是卖出["B"/"S"], 价格, volume}
        signal: {symbol:str,direction:str,price:float,volume:int}

        """

    @abstractmethod
    def stratgy_update(self) -> Dict:
        """
        根据更新过后的signal, 更新成交记录和持仓记录, 更新策略的评价结果
        动量是胜率、赔率, VWAP是成交成本与实际VWAP的差
        """
