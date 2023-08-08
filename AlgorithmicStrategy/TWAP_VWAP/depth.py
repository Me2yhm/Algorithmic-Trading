from typing import TypedDict, List
from collections import OrderedDict
from DataManager import DataStream, TimeType, OT, DataSet

class SnapShot(TypedDict):
    timestamp: TimeType
    ask: OrderedDict[float, float]
    bid: OrderedDict[float, float]

class OrderDepthCalculator:

    def __init__(self, snapshot: SnapShot, n: int):
        self.snapshot = snapshot
        self.n = n
        self.n_depth = self.get_n_depth()
        self.total_volume = self.calculate_total_volume()

    def get_n_depth(self) -> List[float]:
        ask_volumes = list(self.snapshot['ask'].values())[:self.n][::-1]
        bid_volumes = list(self.snapshot['bid'].values())[:self.n]
        ask_volumes = [0.0] * (self.n - len(ask_volumes)) + ask_volumes
        bid_volumes += [0.0] * (self.n - len(bid_volumes))
        return ask_volumes + bid_volumes

    @staticmethod
    def exponential_decay_weight(distance_to_optimal: int, decay_rate: float = 0.1) -> float:
        return 1.0 / (1.0 + decay_rate * distance_to_optimal)

    def calculate_total_volume(self) -> float:
        return sum(self.n_depth)

    def calculate_weighted_average_depth(self, decay_rate: float = 0.1) -> float:
        weighted_sum = 0.0
        total_weight = 0.0

        for idx, volume in enumerate(self.n_depth):
            distance_to_optimal = abs(idx-n+1)

            # 计算指数衰减权重
            weight = OrderDepthCalculator.exponential_decay_weight(distance_to_optimal, decay_rate)

            weighted_sum += volume * weight
            total_weight += weight

        if total_weight == 0.0:
            return 0.0

        # 计算加权平均订单深度
        weighted_average_depth = weighted_sum / total_weight

        return weighted_average_depth

    def calculate_buy_average_depth(self, decay_rate: float = 0.1) -> float:
        bid_volumes = list(self.snapshot['bid'].values())[:self.n]
        bid_volumes += [0.0] * (self.n - len(bid_volumes))

        weighted_sum = 0.0
        total_weight = 0.0

        for idx, volume in enumerate(bid_volumes):
            distance_to_optimal = abs(idx)
            weight = OrderDepthCalculator.exponential_decay_weight(distance_to_optimal, decay_rate)
            weighted_sum += volume * weight
            total_weight += weight

        if total_weight == 0.0:
            return 0.0

        buy_average_depth = weighted_sum / total_weight
        return buy_average_depth

    def calculate_sell_average_depth(self, decay_rate: float = 0.1) -> float:
        ask_volumes = list(self.snapshot['ask'].values())[:self.n][::-1]
        ask_volumes = [0.0] * (self.n - len(ask_volumes)) + ask_volumes

        weighted_sum = 0.0
        total_weight = 0.0

        for idx, volume in enumerate(ask_volumes):
            distance_to_optimal = abs(idx-self.n+1)
            weight = OrderDepthCalculator.exponential_decay_weight(distance_to_optimal, decay_rate)
            weighted_sum += volume * weight
            total_weight += weight

        if total_weight == 0.0:
            return 0.0

        sell_average_depth = weighted_sum / total_weight
        return sell_average_depth

    #宽度
    # def calculate_spread(self) -> float:
    #     ask_prices = list(self.snapshot['ask'].keys())[:self.n][::-1]
    #     bid_prices = list(self.snapshot['bid'].keys())[:self.n]
    #
    #     if not ask_prices or not bid_prices:
    #         return 0.0  # Return 0 if no ask or bid prices available
    #
    #     max_ask = max(ask_prices)
    #     min_bid = min(bid_prices)
    #
    #     spread = max_ask - min_bid
    #     return spread

# 测试
n = 10 #盘口挡位
decay_rates = [1]#衰减率
snapshot = SnapShot(
    timestamp=1630373400,
    ask=OrderedDict([(10.0, 100.0), (11.0, 150.0), (12.0, 200.0), (13.0, 300.0), (14.0, 250.0), (15.0, 180.0)]),
    bid=OrderedDict([(9.0, 80.0), (8.0, 120.0), (7.0, 160.0), (6.0, 250.0), (5.0, 300.0), (4.0, 220.0)]))

order_depth_calculator = OrderDepthCalculator(snapshot, n)
#宽度
#spread = order_depth_calculator.calculate_spread()

for decay_rate in decay_rates:
    weighted_average_depth = order_depth_calculator.calculate_weighted_average_depth(decay_rate)
    buy_average_depth = order_depth_calculator.calculate_buy_average_depth(decay_rate)
    sell_average_depth = order_depth_calculator.calculate_sell_average_depth(decay_rate)

    print(f"买盘{n}档的平均订单深度: {buy_average_depth}")
    print(f"卖盘{n}档的平均订单深度: {sell_average_depth}")
    print(f"衰减率为 {decay_rate} 时，买卖{n}档的加权平均订单深度: {weighted_average_depth}")
    print(order_depth_calculator.n_depth)
    print(order_depth_calculator.total_volume)
    #宽度
    #print(f"买卖{n}档的盘口宽度: {spread}")
