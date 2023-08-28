from typing import Literal
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from AlgorithmicStrategy import AlgorithmicStrategy, TradeTime, DataSet, OrderBook


class TWAP(AlgorithmicStrategy):
    def __init__(
        self,
        orderbook: OrderBook,
        tick: DataSet,
        trade_volume: float,
        time_interval: float,
        time_limit: float,
        symbol: str,  # 股票代码
        direction: Literal["BUY", "SELL"],  # 买卖方向
        commision: float = 0.00015,
        stamp_duty: float = 0.001,
        transfer_fee: float = 0.00002,
        pre_close: float = 0,
    ) -> None:
        super().__init__(orderbook, commision, stamp_duty, transfer_fee, pre_close)
        self.time_dict = {}
        self.tick = tick
        self.ob = orderbook
        self.signal = {
            "symbol": symbol,
            "direction": direction,
            "price": None,
            "volume": None
        }
        self.ts_list = []
        self.trade: bool
        self.volume_traded: float = 0.0
        self.money_traded: float = 0.0
        self.vwap: float
        self.vwap_market: float
        self.delta_vwap: float
        self.trade_num: float
        self.trade_volume: float = trade_volume
        self.time_interval: float = time_interval
        self.time_limit: tuple = (-time_limit,time_limit)
        self.interval_dict : dict = {}

    def get_time_dict(self):
        self.interval_dict['trade_interval'] = self.time_interval
        self.interval_dict["trade_limits"] = self.time_limit
        tt = TradeTime(begin=93000000, end=145700000, tick=self.tick)
        time_dict = tt.generate_signals(**self.interval_dict)
        self.time_dict = time_dict

   
    def get_trade_times(self):
        trade_times_list = []
        for _,v in self.time_dict:
            if v['trade']:
                trade_times_list.append(v)
        self.trade_times = len(trade_times_list)
        self.signal['volume'] = self.trade_volume/self.trade_times


    def model_update(self) -> None:
        pass

    def signal_update(self):
        for ts, action in self.time_dict:
            if ts == self.timeStamp and action["trade"]:
                self.ts_list.append(ts)
                self.ob.update(ts)
                if self.signal["direction"] == "BUY":
                    price = list(self.ob.search_snapshot(ts)["ask"].keys())[0]
                else:
                    price = list(self.ob.search_snapshot(ts)["bid"].keys())[0]
                self.signal["price"] = price
                self.volume_traded += self.signal["volume"]
                self.money_traded += price * self.signal["volume"]
                self.vwap_market = list(
                    self.ob.search_snapshot(ts)["total_trade"].values()
                )[2]
                self.trade = True
                break
            else:
                self.trade = False

    def stratgy_update(self):
        self.vwap = self.money_traded / self.volume_traded
        self.delta_vwap = self.vwap - self.vwap_market
