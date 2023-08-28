import argparse
import sys
import warnings
from abc import ABC
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

import numpy as np
import torch as t
from torch import optim

from AlgorithmicStrategy import DataSet, OrderBook, TradeTime, AlgorithmicStrategy


class VWAP(AlgorithmicStrategy):
    def __init__(self, orderbook: OrderBook, tick: DataSet, trade_volume: float, symbol: str,
                 direction: Literal["BUY", "SELL"], commission: float = 0.00015, stamp_duty: float = 0.001,
                 transfer_fee: float = 0.00002, pre_close: float = 0, ):
        super().__init__(orderbook, commission, stamp_duty, transfer_fee, pre_close)
        self.direction = direction
        self.symbol = symbol
        self.pre_close = pre_close
        self.transfer_fee = transfer_fee
        self.stamp_duty = stamp_duty
        self.trade_volume = trade_volume
        self.tick = tick
        self.orderbook = orderbook
        self.commission = commission

def get_newest_model():
    pass

if __name__ == "__main__":
    model_save_path: Path = Path().cwd() / "MODEL_SAVE"

