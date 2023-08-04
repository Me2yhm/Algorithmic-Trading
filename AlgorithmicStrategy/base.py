from abc import ABC, abstractmethod

from typing import Dict, List, TypedDict
from .utils import get_date
from .OrderMaster.OrderBook import OrderBook, OT


class deal(TypedDict):
    timestamp: int
    symbol: str
    direction: str
    price: float
    volume: int


class signal(TypedDict):
    timestamp: int
    symbol: str
    direction: str
    price: float
    volume: int


class possession(TypedDict):
    code: str
    volume: int
    averagePrice: float
    cost: float


class AlgorithmicStrategy(ABC):
    """
    抽象基类, 定义算法交易类的接口, 有如下属性
    orderbook: orderbook类, 可以撮合盘口, 记录盘口状态
    timeStamp: 记录当前时间戳和上一秒时间戳
    deals: \记录每日成交记录,一个一个字典,键为日期,值为deal类组成的列表:{date:[deal]}
          \deal类似字典,与signal类似
    possession: \记录每日持仓记录,一个字典,键为日期,值为possession类:{date:possession}
                \possession类似字典
                \其中cost是总的交易成本,考虑手续费;
                \ averagePrice是股票总金额除以总量(不包含手续费)
    signal: \记录每日交易信号, 一个字典,键为日期,值为signal类组成的列表:{date:[signal]}
            \signal类是一次交易信号,组成的列表代表一天的交易信号
            \signal类似字典,键名含义如下
            \symbol为股票代码, direction表示买入或者卖出: ["B"/"S"],
    tick: \记录每日的逐笔数据, 一个字典,字典的键为日期,值为一个由OT类组成的列表:{date:[OT]}
          \OT可以看作一个字典,形式为: {time:int,oid:int,oidb:int,oids:int,price:float,volume:int,flag:int,ptype:int}
          \ time表示当前时间戳, oid等键名的含义可以查询: 语雀|技术团队|数据格式知识库|逐笔数据文档
    commission: 手续费, 券商收取, 默认按万分之1.5算
    stamp_duty: 印花税, 买入没有印花税, 卖出有, 为0.001
    transfer_fee: 过户费, 为0.00002, 买卖都有
    """

    orderbook: OrderBook
    ticks: Dict[str, List[OT]]
    signals: Dict[str, List[signal]]
    timeStamp: int
    _date: str
    newday: bool
    deals: Dict[str, List[deal]]
    possessions: Dict[str, possession]
    commission: float
    stamp_duty: float
    transfer_fee: float
    sell_cost: float
    buy_cost: float

    def __init__(
        self,
        commision: float = 0.00015,
        stamp_duty: float = 0.001,
        transfer_fee: float = 0.00002,
    ) -> None:
        self.ticks = {}
        self.signals = {}
        self.timeStamp = 0
        self.deals = {}
        self.possessions = {}
        self.commission = commision
        self.stamp_duty = stamp_duty
        self.transfer_fee = transfer_fee
        self.buy_cost = self.commission + self.transfer_fee
        self.sell_cost = self.commission + self.transfer_fee + self.stamp_duty
        self._date = ""
        self.newday = True

    @property
    def date(self):
        """
        获得当前日期
        """
        return self._date

    @date.setter
    def date(self, newdate):
        """
        允许对self.date赋值,且当日期更改之后,self.newday变为True
        """
        if newdate != self._date:
            self.newday = True
            self._date = newdate
        else:
            self.newday = False

    def update_orderbook(self, lines: List[OT]) -> None:
        """
        更新orderbook和tick数据
        """
        if self.newday:
            self.ticks[self.date] = lines
        else:
            self.ticks[self.date].extend(lines)
        try:
            assert all([line["time"] == lines[0]["time"] for line in lines])
            self.timeStamp = lines[-1]["time"]
            self.date = get_date(self.timeStamp)
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
    def stratgy_update(self):
        """
        根据更新过后的signal, 更新成交记录和持仓记录, 更新策略的评价结果
        动量是胜率、赔率, VWAP是成交成本与实际VWAP的差
        """
