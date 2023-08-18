from collections import OrderedDict
from typing import List

from Schema import SnapShot


class OrderDepthCalculator:
    def __init__(self, snapshot: SnapShot, n: int):
        self.snapshot = snapshot
        self.n = n
        self.n_depth = self.get_n_depth()
        self.total_volume = self.calculate_total_volume()
        self.weighted_average_depth: float = None

    def get_n_depth(self) -> List[float]:
        ask_volumes = list(self.snapshot["ask"].values())[: self.n][::-1]
        bid_volumes = list(self.snapshot["bid"].values())[: self.n]
        ask_volumes = [0.0] * (self.n - len(ask_volumes)) + ask_volumes
        bid_volumes += [0.0] * (self.n - len(bid_volumes))
        return ask_volumes + bid_volumes

    @staticmethod
    def exponential_decay_weight(
        distance_to_optimal: int, decay_rate: float = 0.1
    ) -> float:
        return 1.0 / (1.0 + decay_rate * distance_to_optimal)

    def calculate_total_volume(self) -> float:
        return sum(self.n_depth)

    def calculate_weighted_average_depth(self, decay_rate: float = 0.1) -> float:
        weighted_sum = 0.0
        total_weight = 0.0

        for idx, volume in enumerate(self.n_depth):
            distance_to_optimal = abs(idx - self.n + 1)

            # 计算指数衰减权重
            weight = OrderDepthCalculator.exponential_decay_weight(
                distance_to_optimal, decay_rate
            )

            weighted_sum += volume * weight
            total_weight += weight

        if total_weight == 0.0:
            return 0.0

        # 计算加权平均订单深度
        self.weighted_average_depth = weighted_sum / total_weight

        return self.weighted_average_depth
