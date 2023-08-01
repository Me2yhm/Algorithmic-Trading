def multi_task_loss(pred_returns, pred_volume_percent, true_returns, true_volume_percent,
                    VWAP_market, VWAP_ML, Volume_rest, alpha=1.0, beta=1.0, eta=1.0, gamma=1.0):
    # 定义每个任务的损失函数
    returns_loss = nn.MSELoss()(pred_returns, true_returns)
    volume_percent_loss = nn.MSELoss()(pred_volume_percent, true_volume_percent)
    vwap_loss = nn.MSELoss()(VWAP_market, VWAP_ML)
    sum_volume_percent = torch.sum(pred_volume_percent, dim=1)
    volume_percent_penalty = nn.MSELoss()(sum_volume_percent, torch.ones_like(sum_volume_percent))

    # 结合所有任务的损失，并加权求和
    total_loss = alpha * returns_loss + beta * volume_percent_loss + eta * vwap_loss + gamma * volume_percent_penalty

    return total_loss

