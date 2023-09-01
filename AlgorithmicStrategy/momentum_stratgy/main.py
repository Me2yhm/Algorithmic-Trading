from abc import ABC, abstractmethod
from typing import List, Dict, Union
from collections import deque
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

from .utils import data_mark
from .anfis_pytorch.membership import make_anfis
from .anfis_pytorch.experimental import train_anfis
from .anfis_pytorch.anfis import AnfisNet
from ..base import AlgorithmicStrategy, Possession, Signal, Deal
from .modelType import modelType
from .ReverseMomentum import Model_reverse
from ..OrderMaster.OrderBook import OrderBook


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
    has_signal: bool

    def __init__(
        self,
        orderbook: OrderBook,
        symbol: str,
        commission: float = 0.00015,
        stamp_duty: float = 0.001,
        transfer_fee: float = 0.00002,
        pre_close: float = 0.0,
    ) -> None:
        super().__init__(
            orderbook, symbol, commission, stamp_duty, transfer_fee, pre_close
        )
        self.model_indicator = []
        self.win_times = {}
        self.win_rate = {}
        self.odds = 0
        self.has_signal = False

    @abstractmethod
    def model_update(self, model: Union[type[modelType], modelType]) -> None:
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
            self.deals[self.date] = []
        if self.has_signal:
            deal: Deal = self.signals[self.date][-1]
            self.deals[self.date].append(deal)

    def update_poccession(self) -> None:
        deal = self.deals[self.date][-1]
        money = deal["volume"] * deal["price"]
        buy_commission = self.buy_cost * money
        sell_commission = self.sell_cost * money

        if self.newday:
            single_possession: Possession = {
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
        if self.has_signal:
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

    def strategy_update(self) -> float:
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
        if self.has_signal:
            if (one_signal["direction"] == "B") and (buy_cost < average_cost):
                self.win_times[self.date].append(1)
            elif (one_signal["direction"] == "S") and (sell_cost > average_cost):
                self.win_times[self.date].append(1)
            else:
                self.win_times[self.date].append(0)
            self.win_rate[self.date] = sum(self.win_times) / len(self.win_times)
        return self.win_rate[self.date]


class anfisModel(modelType):
    """
    用anfis模型输出交易信号, anfis输入数据二维, 输出一维, 支持流式

    """

    test_count: int
    train_data: list[deque]
    train_count: int
    model: AnfisNet
    num_mfs: int
    num_out: int
    buyed_volume: int

    def __init__(self, num_mfs: int = 2, num_out: int = 1) -> None:
        self.test_count = 0
        self.train_count = 0
        self.train_data = [deque(maxlen=800) for i in range(3)]
        self.num_mfs = num_mfs
        self.num_out = num_out
        self.buyed_volume = 0
        pass

    @property
    def has_model(self) -> bool:
        if self.train_count < 800:
            return -1
        elif self.train_count == 800:
            return 0
        else:
            return 1

    @property
    def can_train(self) -> bool:
        if self.test_count < 200:
            return False
        else:
            return True

    def make_data(self, batch_size=1024):
        *x, y = self.train_data
        y = data_mark(y)
        x = torch.tensor(x).transpose(-1, 0)
        y = torch.tensor(y).transpose(-1, 0)
        td = TensorDataset(x, y)
        return DataLoader(td, batch_size=batch_size, shuffle=True)

    def make_model(self, x):
        self.model = make_anfis(x, self.num_mfs, self.num_out)

    def train_model(self):
        data = self.make_data()
        train_anfis(model=self.model, data=data)
        self.test_count = 0

    def pred(self, x: int, total_volum: int = 100000):
        output = self.model(x)
        if self.buyed_volume < total_volum and output >= 0.2:
            max_volum = int(total_volum * 0.25) + 1
            buy_volum = int(output * max_volum)
            self.buyed_volume = min(self.buyed_volume + buy_volum, total_volum)
            return True, buy_volum
        else:
            return False, 0

    def model_update(self, lines):
        self.train_count += 1
        for i in range(3):
            self.train_data[i].append(lines[i])
        if self.has_model == 1:
            self.test_count += 1
            dataset = self.make_data()
            if self.can_train:
                *x, y = dataset.dataset.tensors
                self.make_model(x)
                self.train_model()
                return False, 0
            else:
                x = torch.tensor(lines[:2]).unsqueeze(0)
                self.pred(x)
        elif self.has_model == 0:
            dataset = self.make_data()
            *x, y = dataset.dataset.tensors
            self.make_model(x)
            self.train_model()
            return False, 0
        else:
            return False, 0


class reverse_strategy(momentumStratgy):
    """
    反转因子模型
    """

    def model_update(self, model: Model_reverse):
        index = model.model_update(self.ticks, self.price_list, self.timeStamp)
        if index is not None:
            self.model_indicator.append(index)

    def signal_update(self, anfis_model: anfisModel):
        line = list(self.model_indicator[-1].values()[:2]).append(self.current_price)
        is_buy, volume = anfis_model.model_update(line)
        if self.newday:
            self.signals[self.date] = []
        if is_buy:
            signal: Signal = {
                "timestamp": self.timeStamp,
                "symbol": self.symbol,
                "direction": "buy",
                "price": self.current_price,
                "volume": volume,
            }
            self.signals[self.date].append(signal)
            self.has_signal = True
        else:
            self.has_signal = False
