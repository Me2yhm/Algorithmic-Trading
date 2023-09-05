import matplotlib.pyplot as plt
import datetime
import json
from pylab import mpl
import csv
from typing import Dict
from urllib.request import urlopen, quote
import numpy as np
import pandas as pd
import seaborn as sns
from decimal import Decimal, InvalidOperation
import ast
import math
from scipy.stats import poisson
import random
from abc import ABC, abstractmethod
from ..base import possession,signal,deal

random.seed(100)


class TradingSimulator(ABC):
    """
    Our aim in this class is to implement the Avellaneda-Stoikov model.
    Notations:
    - s(t): fair price process of the underlying stock
    - x(t): cash process
    - q(t): stock inventory (how many stocks the market maker is holding)

    In this case, we assume that the price moves in the dynamics of a Brownian motion.
    Stock inventory moves in Poisson process.
    """

    def __init__(self, prices: list, buy: list, sell: list, T: list, sigma,
                 model: str, number_of_simulation, k):
        """

        :param S0: initial stock price
        :param q: stock inventory
        :param sigma: volatility of the underlying
        :param T: maturity of the simulation
        :param gamma: risk aversion coefficient
        :param k: liquidity value
        """

        # Variable models
        # self.data = pd.read_csv(data_file)
        self.prices = prices
        self.S0 = prices[0]
        self.T = T
        self.model = model
        self.sigma = sigma
        self.buy = buy
        self.sell = sell
        self.number = number_of_simulation

        # Fixed models
        self.A = 0.05
        self.k = k
        self.q_tilde = 100
        self.gamma = 0.001 / self.q_tilde
        self.n_steps = len(T)
        self.n_paths = 500
        self.h = 6 * 60
        self.adverse = False
        self.signals = []
        

        if model == 'AS':
            self.deltaBidZero = 0.5 * self.gamma * self.sigma**2 * (
                self.T[-1] - self.T[0]) + 1 / self.gamma * math.log(
                    1 + self.gamma / self.k) + self.gamma * self.sigma**2 * (
                        self.T[-1] - self.T[0]) * 0
            self.deltaAskZero = 0.5 * self.gamma * self.sigma**2 * (
                self.T[-1] - self.T[0]) + 1 / self.gamma * math.log(
                    1 + self.gamma / self.k) - self.gamma * self.sigma**2 * (
                        self.T[-1] - self.T[0]) * 0
        else:
            self.trivial_value = 0
            self.deltaAskZero = 0
            self.deltaBidZero = 0

        self.orders = None
        self.cashflow = None
        self.pnl = None
        self.bidPrice = None
        self.askPrice = None
        self.jump = 1

    def setTrivialDeltaValue(self, value):
        """
        Set value of the trivial model
        :return: None
        """
        self.trivial_value = value

        # Call the setTrivialDelta method to set the value
        self.setTrivialDelta(delta=value)

    def setTrivialDelta(self, delta):
        """
        Method that sets the initial values of trivial deltas using the trivial value attribute
        :return:
        """

        self.deltaBidZero = delta
        self.deltaAskZero = delta

    def setJump(self, jump=0.0):
        """
        Method that sets the jump for the Question 5. Introduced in the middle of the trading day.
        :param jump: float.
        :return: None
        """

        self.jump = 1 + jump

    def activateAdverseSelection(self, activate=False):
        """
        Method that activates adverse selection
        :return: None.
        """

        self.adverse = activate

    @staticmethod
    def sigmoid(x):
        """
        Method that returns the sigmoid value.
        :param x: float.
        :return: float. Sigmoid value of given x.
        """

        return 1 / (1 + math.exp(-x))
    


    def execute(self):

        def model_update(self):
            # time = self.data['time'].values
            prices = self.prices
            # bid_prices = self.data[['bid1_price', 'bid2_price', 'bid3_price', 'bid4_price', 'bid5_price']].values
            # bid_volumes = self.data[['bid1_volume', 'bid2_volume', 'bid3_volume', 'bid4_volume', 'bid5_volume']].values
            # ask_prices = self.data[['ask1_price', 'ask2_price', 'ask3_price', 'ask4_price', 'ask5_price']].values
            # ask_volumes = self.data[['ask1_volume', 'ask2_volume', 'ask3_volume', 'ask4_volume', 'ask5_volume']].values

            orders = np.zeros(self.n_steps)
            orders[0] = 0

            cashflow = np.zeros(self.n_steps)
            cashflow[0] = 0

            # Create the delta bid and asks and add the first value
            deltaBid = np.zeros(self.n_steps)
            deltaAsk = np.zeros(self.n_steps)
            deltaBid[0] = self.deltaBidZero
            deltaAsk[0] = self.deltaAskZero

            lambdaBid = np.zeros(self.n_steps)
            lambdaAsk = np.zeros(self.n_steps)
            lambdaBid[0] = self.A * math.exp(-self.k * self.deltaBidZero)
            lambdaAsk[0] = self.A * math.exp(-self.k * self.deltaAskZero)
            pnl_of_simulation = np.zeros((self.number))
            final_inventory = np.zeros((self.number))

            for t in range(1, self.n_steps - 3):

                # If the model we are using the AS model then we set the lambda and poisson process values
                if self.model == 'AS':
                    deltaBid[t] = round(
                        max(
                            0.5 * self.gamma * self.sigma**2 *
                            (self.T[-1] - self.T[0]) +
                            1 / self.gamma * math.log(1 + self.gamma / self.k) +
                            self.gamma * self.sigma**2 *
                            (self.T[-1] - self.T[0]) * orders[t - 1], 0), 2)
                    deltaAsk[t] = round(
                        max(
                            0.5 * self.gamma * self.sigma**2 *
                            (self.T[-1] - self.T[0]) +
                            1 / self.gamma * math.log(1 + self.gamma / self.k) -
                            self.gamma * self.sigma**2 *
                            (self.T[-1] - self.T[0]) * orders[t - 1], 0), 2)

                # Else we proceed with the initial delta values
                else:
                    deltaBid[t] = self.deltaBidZero
                    deltaAsk[t] = self.deltaAskZero

            for simulation in range(self.number):
                pnl = np.zeros((len(prices)))
                cashflow = np.zeros((len(prices)))
                orders = np.zeros((len(prices)))
                reserve_price = np.zeros((len(prices)))
                r_optimal_ask = np.zeros((len(prices)))
                r_optimal_bid = np.zeros((len(prices)))

                for step in range(len(prices) - 1):
                    reserve_price[step] = prices[
                        step] - orders[step] * self.gamma * (self.sigma**2)
                    reserve_spread = (2 / self.gamma) * np.log(1 +
                                                            self.gamma / self.k)

                    r_optimal_ask[step] = reserve_price[step] + reserve_spread / 2
                    r_optimal_bid[step] = reserve_price[step] - reserve_spread / 2
                    optimal_distance_ask = -self.gamma * orders[step] * (
                        self.sigma**
                        2) + (1 / self.gamma) * np.log(1 + (self.gamma / self.k))
                    optimal_distance_bid = self.gamma * orders[step] * (
                        self.sigma**
                        2) + (1 / self.gamma) * np.log(1 + (self.gamma / self.k))

                    lambda_ask = np.exp(-self.k * optimal_distance_ask)
                    lambda_bid = np.exp(-self.k * optimal_distance_bid)

                    ask_probability = 1 - math.exp(-lambda_ask)
                    bid_probability = 1 - math.exp(-lambda_bid)

                    ask_amount = 0
                    bid_amount = 0
                    if random.random() < ask_probability:
                        ask_amount = 1
                    if random.random() < bid_probability:
                        bid_amount = 1
                    
                    orders[step + 1] = orders[step] - ask_amount + bid_amount
                    cashflow[step + 1] = cashflow[step] + r_optimal_ask[
                        step] * ask_amount - r_optimal_bid[step] * bid_amount
                    pnl[step +
                        1] = cashflow[step + 1] + orders[step + 1] * prices[step]

                pnl_of_simulation[simulation] = pnl[-1]
                final_inventory[simulation] = orders[-1]

            self.orders = orders
            self.cashflow = cashflow
            self.pnl = pnl
            self.bidPrice = r_optimal_bid
            self.askPrice = r_optimal_ask
            for prediction in bid_probability:
                    # 根据预测结果生成信号
                    if random.random() < prediction:

                        signal = {
                            "symbol": "AAPL",  # 股票代码，需要读取变量，这边只是一个假设
                            "direction": "B",  # 买入 
                            "bidprice": r_optimal_bid,  # 买价
                            "volume": orders  # 交易量
                        }
                    else:
                        signal = {
                            "symbol": "AAPL",  # 同上
                            "direction": "S",  # 卖出 
                            "askprice": r_optimal_ask,  #卖价
                            "volume": orders  # 交易量
                        }
                    self.signals.append(signal)

            return self.signals
        
        def signal_update(deals: Dict[int, deal], portfolio: Dict[str, possession], 
                          timestamp: int, symbol: str, direction: str, price: float, volume: int):
            deals[timestamp] = deal(timestamp, symbol, direction, price, volume)

            # 更新持仓记录
            if symbol in portfolio:
                # 如果股票代码已存在于持仓记录中，更新相应的数量、平均价格和成本
                possession = portfolio[symbol]
                if direction == "B":
                    possession.volume += volume
                    possession.cost += price * volume
                    possession.averagePrice = possession.cost / possession.volume
                elif direction == "S":
                    possession.volume -= volume
                    possession.cost -= price * volume
                    if possession.volume == 0:
                        # 如果持仓数量为0，从持仓记录中移除该股票代码
                        del portfolio[symbol]
                else:
                    # 处理未知的交易方向
                    print("Unknown direction:", direction)
            else:
                # 如果股票代码不存在于持仓记录中，创建新的持仓记录
                if direction == "B":
                    cost = price * volume
                    averagePrice = price
                    possession = possession(symbol, volume, averagePrice, cost)
                    portfolio[symbol] = possession
                elif direction == "S":
                    # 处理卖出时没有对应的买入记录的情况
                    print("No corresponding buy record for sell transaction:", symbol)
                else:
                    # 处理未知的交易方向
                    print("Unknown direction:", direction)

        def strategy_update(deal: Dict[int, deal]):
            """
            根据更新过后的signal, 更新成交记录和持仓记录, 更新策略的评价结果
            评价指标是收益率
            """
            total_profit = {}
            # 遍历成交记录，计算每笔交易的收益率
            for timestamp, deal in deal.items():
                symbol = deal.symbol
                direction = deal.direction
                price = deal.price
                volume = deal.volume

                if direction == "B":
                    # 如果是买入交易，记录买入价格
                    total_profit[timestamp] = {
                        "symbol": symbol,
                        "buy_price": price,
                        "sell_price": None,
                        "returns": None
                    }
                elif direction == "S":
                    # 如果是卖出交易，计算收益率并更新字典
                    if timestamp in total_profit:
                        total_profit[timestamp]["sell_price"] = price
                        buy_price = total_profit[timestamp]["buy_price"]
                        sell_price = total_profit[timestamp]["sell_price"]
                        total_profit[timestamp]["returns"] = (sell_price - buy_price) / buy_price

            return total_profit
        # signal_result = signal_update(deal, possession, self.T, "AAPL", "B", self.prices, self.orders)#还需要改
        # self.print_signal_result(signal_result)
        strategy_result = strategy_update(deal)
        self.print_strategy_result(strategy_result)

    def print_signals(self):
        # 输出信号列表
        signals = self.signals
        for signal in signals:
            print(signal)

    def print_positions(portfolio: Dict[str, possession]):
        # 输出持仓记录
        for symbol, possession in portfolio.items():
            print(f"Stock Code: {symbol}")
            print(f"Volume: {possession.volume}")
            print(f"Average Price: {possession.averagePrice}")
            print(f"Cost: {possession.cost}")
            print()

    def print_transactions(deal: Dict[int, deal]):
        # 输出成交记录
        for timestamp, deal in deal.items():
            print(f"Timestamp: {timestamp}")
            print(f"Symbol: {deal['symbol']}")
            print(f"Direction: {deal['direction']}")
            print(f"Price: {deal['price']}")
            print(f"Volume: {deal['volume']}")
            print()
    
    def print_strategy_result(self, result: Dict):
        # 输出策略的评价结果
        print(result)

         


         

    # def getPrices(self):
    #     """
    #     Method that returns prices.
    #     :return: list. self.prices
    #     """

    #     return self.prices

    # def getOrders(self):
    #     """
    #     Method that returns orders.
    #     :return: list. self.orders
    #     """

    #     return self.orders

    # def getCashflow(self):
    #     """
    #     Method that returns cashflow.
    #     :return: list. self.cashflow
    #     """

    #     return self.cashflow

    # def getPnL(self):
    #     """
    #     Method that returns PnL.
    #     :return: list. self.pnl
    #     """

    #     return self.pnl

    # def getFinalPnL(self):
    #     """
    #     Method that returns PnL.
    #     :return: double. self.pnl
    #     """

    #     return self.pnl[-1]

    # def visualize(self):
    #     """
    #     Method to visualize market
    #     :return:
    #     """
    #     time_steps = list(range(self.n_steps))
    #     sns.color_palette("hls", 8)
    #     fig, ax = plt.subplots(4,
    #                            1,
    #                            gridspec_kw={'height_ratios': [3, 1, 1, 3]},
    #                            sharex=True,
    #                            figsize=(20, 10))

    #     ################### Plot values ###################
    #     g1_1 = sns.lineplot(x=time_steps,
    #                         y=self.prices,
    #                         ax=ax[0],
    #                         label='Fair Value')

    #     g1_bid = sns.lineplot(x=time_steps,
    #                           y=self.bidPrice,
    #                           ax=ax[0],
    #                           label='Bid Price',
    #                           alpha=0.6,
    #                           linestyle='--')

    #     g1_ask = sns.lineplot(x=time_steps,
    #                           y=self.askPrice,
    #                           ax=ax[0],
    #                           label='Ask Price',
    #                           alpha=0.6,
    #                           linestyle='--')

    #     g1_2 = sns.lineplot(x=time_steps,
    #                         y=self.orders,
    #                         ax=ax[1],
    #                         label='Inventory',
    #                         drawstyle='steps-pre')

    #     g1_3 = sns.lineplot(x=time_steps,
    #                         y=self.cashflow,
    #                         ax=ax[2],
    #                         label='Cash flow',
    #                         drawstyle='steps-pre')

    #     g1_4 = sns.lineplot(x=time_steps,
    #                         y=self.pnl,
    #                         ax=ax[3],
    #                         label='PnL',
    #                         drawstyle='steps-pre')

    #     plt.show()
