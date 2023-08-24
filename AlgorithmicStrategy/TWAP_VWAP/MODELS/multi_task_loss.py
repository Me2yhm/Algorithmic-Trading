from typing import Literal

from torch import nn, Tensor
import torch as t


class MultiTaskLoss:
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        eta: float = 1.0,
        gamma: float = 1.0,
        direction: Literal['BUY', "ASK"] = 'BUY',
    ):
        total_weights: float = sum([alpha, beta, gamma, gamma])
        self.alpha: float = alpha / total_weights
        self.beta: float = beta / total_weights
        self.eta: float = eta / total_weights
        self.gamma: float = gamma / total_weights
        self.direction = direction
        self.multiplier = -1 if direction == 'ASK' else 1

        self.mse_loss: nn.Module = nn.MSELoss()
        self.cross_loss: nn.Module = nn.CrossEntropyLoss()

    def calculate_loss(
        self,
        pred_volume_percent: Tensor,
        true_volume_percent: Tensor,
        history_volume_percent: Tensor,
        VWAP_market: Tensor,
        VWAP_ML: Tensor,
    ):
        # 定义每个任务的损失函数
        volume_percent_market_loss = self.mse_loss(
            pred_volume_percent, true_volume_percent
        )
        volume_percent_history_loss = self.mse_loss(
            pred_volume_percent, history_volume_percent
        )

        vwap_loss = self.multiplier * (VWAP_ML - VWAP_market)

        sum_volume_percent = t.sum(pred_volume_percent)
        volume_percent_penalty = t.abs(sum_volume_percent - 1)

        # 结合所有任务的损失，并加权求和
        total_loss = (
            self.alpha * volume_percent_market_loss
            + self.beta * volume_percent_history_loss
            + self.eta * vwap_loss
            + self.gamma * volume_percent_penalty
        )

        return total_loss
