from abc import ABC, abstractmethod
from typing import List, Dict
from AlgorithmicStrategy.base import AlgorithmicStrategy, possession, signal
from .modelType import modelType


class momentumStratgy(AlgorithmicStrategy, ABC):
    """
    动量算法类, 有如下属性
    orderbook: orderbook类, 可以撮合盘口, 记录盘口状态
    tick: 储存逐笔数据
    timeStamp: 记录当前时间戳
    deals: 成交记录
    possessionss: 调仓记录
    model_indicator: 指标计算结果
    signals:记录交易信号
    win_times: \一个字典, 记录每日每笔交易盈利与否, 盈利为1, 否则为0
               \键为日期,值为一个列表,记录每笔交易盈利情况{date:[盈利为1, 否则为0]}
    win_rate: 胜率,一个字典,键为日期,值为该日的胜率{date:win_rate}
    odds: 赔率,一个字典,键为日期,值为改日赔率{date:odds}
    """

    win_times: Dict[str, List[int]]
    model_indicator: List[Dict[str, float]]
    win_rate = Dict[str, float]
    odds: float

    def __init__(self) -> None:
        super().__init__()
        self.model_indicator = []
        self.win_times = {}
        self.win_rate = {}
        self.odds = 0

    @abstractmethod
    def model_update(self, model: modelType) -> None:
        """
        盘口更新过后, 根据更新过的数据增量地更新指标或者训练模型
        """
        pass

    @abstractmethod
    def signal_update(self) -> dict:
        """
        调用model_update函数, 根据函数结果返回信号
        """
        pass

    def update_deal(self) -> None:
        if self.newday:
            self.deals[self.date] = [self.signals[self.date][-1]]
        else:
            self.deals[self.date].append(self.signals[self.date][-1])

    def update_poccession(self) -> None:
        deal = self.deals[self.date][-1]
        money = deal["volume"] * deal["price"]
        buy_commission = self.buy_cost * money
        sell_commission = self.sell_cost * money

        if self.newday:
            single_possession: possession = {
                "code": deal["symbol"],
                "averagePrice": 0.0,
                "cost": 0.0,
                "volume": 0,
            }
            self.possessions[self.date] = single_possession
            total = 0
        else:
            total = (
                self.possessions[self.date]["volume"]
                * self.possessions[self.date]["averagePrice"]
            )

        if deal["direction"] == "B":
            self.possessions[self.date]["volume"] += deal["volume"]
            self.possessions[self.date]["cost"] += money + buy_commission
            self.possessions[self.date]["averagePrice"] = (
                total + money
            ) / self.possessions[self.date]["volume"]
        else:
            self.possessions[self.date]["volume"] -= deal["volume"]
            self.possessions[self.date]["cost"] -= money - sell_commission
            self.possessions[self.date]["averagePrice"] = (
                total - money
            ) / self.possessions[self.date]["volume"]

    def stratgy_update(self) -> float:
        """
        根据返回的信号计算胜率、赔率、换手率等——可以流式？
        """
        self.signal_update()
        self.update_deal()
        self.update_poccession()
        single_poccession = self.possessions[self.date]
        average_cost = single_poccession["cost"] / single_poccession["volume"]
        one_signal = self.signals[self.date][-1]
        buy_cost = one_signal["price"] + self.buy_cost
        sell_cost = one_signal["price"] - self.sell_cost
        if self.newday:
            self.win_times[self.date] = []
            self.win_rate[self.date] = 0
        if (one_signal["direction"] == "B") and (buy_cost < average_cost):
            self.win_times[self.date].append(1)
        elif (one_signal["direction"] == "S") and (sell_cost > average_cost):
            self.win_times[self.date].append(1)
        else:
            self.win_times[self.date].append(0)

        self.win_rate[self.date] = sum(self.win_times) / len(self.win_times)
        return self.win_rate[self.date]
