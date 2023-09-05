import logging
import numpy as np

from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Union
from collections import deque


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

parent_path = Path(__file__).parent
logging.basicConfig(
    filename=Path.joinpath(parent_path, "reverse.log"),
    level=logging.INFO,
    format="%(asctime)s-%(message)s",
)


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
            logging.info("generate a deal")

    def update_poccession(self) -> None:
        if self.newday:
            single_possession: Possession = {
                "code": self.symbol,
                "averagePrice": 0.0,
                "cost": 0.0,
                "volume": 0,
            }
            self.possessions[self.date] = single_possession
            total = 0
        if self.has_signal:
            deal = self.deals[self.date][-1]
            money = deal["volume"] * deal["price"]
            buy_commission = self.buy_cost * money
            sell_commission = self.sell_cost * money
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
            logging.info("update the poccession")

    def strategy_update(self) -> float:
        """
        根据返回的信号计算胜率、赔率、换手率等——可以流式？
        """
        self.signal_update()
        self.update_deal()
        self.update_poccession()

        if self.newday:
            self.win_times[self.date] = []
            self.win_rate[self.date] = 0
        if self.has_signal:
            single_poccession = self.possessions[self.date]
            average_cost = single_poccession["cost"] / single_poccession["volume"]
            one_signal = self.signals[self.date][-1]
            buy_cost = one_signal["price"] + self.buy_cost
            sell_cost = one_signal["price"] - self.sell_cost
            if (one_signal["direction"] == "B") and (buy_cost < average_cost):
                self.win_times[self.date].append(1)
            elif (one_signal["direction"] == "S") and (sell_cost > average_cost):
                self.win_times[self.date].append(1)
            else:
                self.win_times[self.date].append(0)
            self.win_rate[self.date] = sum(self.win_times[self.date]) / len(
                self.win_times[self.date]
            )
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
        self.train_window = 1600
        self.test_window = 400
        self.train_data = [deque(maxlen=self.train_window) for i in range(3)]
        self.num_mfs = num_mfs
        self.num_out = num_out
        self.buyed_volume = 0

        pass

    @property
    def has_model(self) -> bool:
        if self.train_count < self.train_window:
            return -1
        elif self.train_count == self.train_window:
            return 0
        else:
            return 1

    @property
    def can_train(self) -> bool:
        if self.test_count < self.test_window:
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
        if self.buyed_volume < total_volum and output >= 0.4:
            max_volum = int(total_volum * 0.25) + 1
            buy_volum = int(output * max_volum)
            self.buyed_volume = min(self.buyed_volume + buy_volum, total_volum)
            return True, buy_volum
        else:
            return False, 0

    def model_update(self, lines: list) -> (bool, int):
        if any(line is None for line in lines):
            return False, 0
        self.train_count += 1
        for i in range(3):
            self.train_data[i].append(lines[i])
        if self.has_model == 1:
            self.test_count += 1
            if all(
                [
                    self.train_data[2][i] == self.train_data[2][0]
                    for i in range(len(self.train_data[2]))
                ]
            ):
                self.test_count = 0
                return False, 0
            dataset = self.make_data()
            if self.can_train:
                x, y = dataset.dataset.tensors
                if any(torch.all(x == x[0], dim=0)):
                    self.test_count -= self.test_window / 2
                    self.train_count -= self.test_window
                    return False, 0
                self.make_model(x)
                logging.info(f"train model with input x: \n {x},{x.size()}")
                self.train_model()
                return False, 0
            else:
                x = torch.tensor(lines[:2]).unsqueeze(0)
                logging.info(f"predict with {x}")
                isbuy, volume = self.pred(x)
                return isbuy, volume
        elif self.has_model == 0:
            dataset = self.make_data()
            x, y = dataset.dataset.tensors
            if torch.all(x == 0):
                self.train_count = 0
                return False, 0
            self.make_model(x)
            logging.info(f"train model with input x: \n {x}")
            self.train_model()
            logging.info("model initiates successfully")
            return False, 0
        else:
            return False, 0


class reverse_strategy(momentumStratgy):
    """
    反转因子模型
    """

    anfis_model: anfisModel

    def model_update(self, model: Model_reverse):
        indicators = model.model_update(self.ticks, self.price_list, self.timeStamp)
        if indicators is not None:
            if all([indi is not None for indi in indicators.values()]):
                indicator_values = [
                    v * 100000 for v in indicators.values()
                ]  # 因子值太小，为了结果显著，放大1000
                indicators.update(zip(indicators.keys(), indicator_values))
                self.model_indicator.append(indicators)

    def signal_update(self):
        if self.newday:
            self.anfis_model = anfisModel()
            self.signals[self.date] = []
        if self.model_indicator and self.current_price > 0:
            line = list(self.model_indicator[-1].values())[:2]
            line.append(self.current_price)
            is_buy, volume = self.anfis_model.model_update(line)
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
                logging.info("generate a signal")
            else:
                self.has_signal = False
