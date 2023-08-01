import torch.nn as nn

class MultiTaskLoss:
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, eta: float = 1.0, gamma: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.gamma = gamma
        self.mse_loss = nn.MSELoss()

    def calculate_loss(self, pred_returns, pred_volume_percent, true_returns, true_volume_percent, VWAP_market, VWAP_ML):
        # 定义每个任务的损失函数
        returns_loss = self.mse_loss(pred_returns, true_returns)
        volume_percent_loss = self.mse_loss(pred_volume_percent, true_volume_percent)
        vwap_loss = self.mse_loss(VWAP_market, VWAP_ML)
        sum_volume_percent = torch.sum(pred_volume_percent, dim=1)
        volume_percent_penalty = self.mse_loss(sum_volume_percent, torch.ones_like(sum_volume_percent))

        # 结合所有任务的损失，并加权求和
        total_loss = self.alpha * returns_loss + self.beta * volume_percent_loss + self.eta * vwap_loss + self.gamma * volume_percent_penalty

        return total_loss

