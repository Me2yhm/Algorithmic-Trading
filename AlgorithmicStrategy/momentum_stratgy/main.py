from AlgorithmicStrategy.base import AlgorithmicStrategy
from .modelType import modelType
from typing import List, Dict


class momentumStratgy(AlgorithmicStrategy):
    """
    动量算法类, 有如下属性
    orderbook: orderbook类, 可以撮合盘口, 记录盘口状态
    tick: 储存逐笔数据
    timeStamp: 记录当前时间戳
    deals: 成交记录
    possession: 调仓记录
    model_indicator: 指标计算结果
    signal:记录交易信号
    win_times: 一个列表, 记录每笔交易盈利与否, 盈利为1, 否则为0
    win_probability: 胜率
    odds: 赔率
    """

    win_times: List[int]
    model_indicator: List[Dict[str, float]]
    win_probability = float
    odds: float

    def __init__(self) -> None:
        super().__init__()
        self.model_indicator = []
        self.win_times = []
        self.win_probability = 0
        self.odds = 0

    def model_update(self, model: modelType) -> None:
        """
        盘口更新过后, 根据更新过的数据增量地更新指标或者训练模型
        """
        pass

    def signal_update(self) -> dict:
        """
        调用model_update函数, 根据函数结果返回信号
        """
        pass

    def update_deal(self) -> None:
        self.deals.append(self.signal[-1])

    def update_poccession(self) -> None:
        deal = self.deals[-1]
        money = deal["volume"] * deal["price"]
        buy_commission = self.buy_cost * money
        sell_commission = self.sell_cost * money
        total = self.possession["volume"] * self.possession["averagePrice"]

        if self.possession["code"] == "":
            self.possession["code"] = deal["symbol"]

        if deal["direction"] == "B":
            self.possession["volume"] += deal["volume"]
            self.possession["cost"] += money + buy_commission
            self.possession["averagePrice"] = (total + money) / self.possession[
                "volume"
            ]
            if deal["price"] < self.possession["averagePrice"]:
                self.win_times.append(1)
            else:
                self.win_times.append(0)
        else:
            self.possession["volume"] -= deal["volume"]
            self.possession["cost"] -= money + sell_commission
            self.possession["averagePrice"] = (total - money) / self.possession[
                "volume"
            ]
            if deal["price"] > self.possession["averagePrice"]:
                self.win_times.append(1)
            else:
                self.win_times.append(0)

    def stratgy_update(self) -> Dict:
        """
        根据返回的信号计算胜率、赔率、换手率等——可以流式？
        """
        self.signal_update()
        self.update_deal()
        self.update_poccession()
        self.win_probability = sum(self.win_times) / len(self.win_times)
