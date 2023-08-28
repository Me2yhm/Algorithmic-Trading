from typing import Literal
from AlgorithmicStrategy import AlgorithmicStrategy, TradeTime, DataSet, OrderBook


class TWAP(AlgorithmicStrategy):
    def __init__(
        self,
        orderbook: OrderBook,
        tick: DataSet,
        trade_num: float,
        trade_volume: float,
        symbol: str,  # 股票代码
        direction: Literal["BUY", "SELL"],  # 买卖方向
        commission: float = 0.00015,
        stamp_duty: float = 0.001,
        transfer_fee: float = 0.00002,
        pre_close: float = 0,
    ) -> None:
        super().__init__(orderbook, commission, stamp_duty, transfer_fee, pre_close)
        self.time_dict = {}
        self.tick = tick
        self.ob = orderbook
        self.signal = {
            "symbol": symbol,
            "direction": direction,
            "price": None,
            "volume": trade_volume / trade_num,
        }
        self.ts_list = []
        self.trade: bool
        self.volume_traded: float = 0.0
        self.money_traded: float = 0.0
        self.vwap: float
        self.vwap_market: float
        self.delta_vwap: float

    def get_time_dict(self):
        tt = TradeTime(begin=93000000, end=145700000, tick=self.tick)
        time_dict = tt.generate_signals()
        self.time_dict = time_dict

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
