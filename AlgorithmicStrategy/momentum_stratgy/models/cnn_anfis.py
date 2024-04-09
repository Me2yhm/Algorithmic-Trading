import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
from sklearn.decomposition import PCA
import sys


sys.path.append("e:\\workspace\\Algorithmic-Trading\\")
from AlgorithmicStrategy.momentum_stratgy.anfis_pytorch.anfis import AnfisNet
from AlgorithmicStrategy.momentum_stratgy.anfis_pytorch.membership import make_gauss_mfs
from AlgorithmicStrategy.momentum_stratgy.dataloader import train_loader
from AlgorithmicStrategy.momentum_stratgy.anfis_pytorch.experimental import (
    calc_error,
    plotErrors,
    plotResults,
)


class convFeature(nn.Module):
    def __init__(self, num_classes, input_dim, seq_len):
        super(convFeature, self).__init__()

        # 加载预训练的VGG16模型
        self.input_dim = input_dim
        self.seq_len = seq_len
        L = (seq_len // 16) * (input_dim // 16) * 128
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(L, 4096),
            nn.Linear(4096, 4096),
        )

        # 全连接层
        self.fc = nn.Linear(4096, num_classes)
        # self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        # 提取图像特征
        with torch.no_grad():
            features = self.cnn(x)
        output = self.fc(features)
        # output = self.soft(output)

        return output


def make_anfis(num_in: int, num_mfs=5, num_out=1, hybrid=False):
    """
    Make an ANFIS model, auto-calculating the (Gaussian) MFs.
    I need the x-vals to calculate a range and spread for the MFs.
    Variables get named x0, x1, x2,... and y0, y1, y2 etc.
    """
    num_invars = num_in
    invars = []
    for i in range(num_invars):
        sigma = 1
        mulist = torch.linspace(-1, 1, num_mfs).tolist()
        invars.append(("x{}".format(i), make_gauss_mfs(sigma, mulist)))
    outvars = ["y{}".format(i) for i in range(num_out)]
    model = AnfisNet("Simple classifier", invars, outvars, hybrid=hybrid)
    return model


def train_anfis_with(
    model: torch.nn.Module,
    data: DataLoader,
    optimizer: Optimizer,
    criterion: torch.nn.MSELoss,
    epochs=500,
    show_plots=False,
):
    """
    Train the given model using the given (x,y) data.
    """
    errors = []  # Keep a list of these for plotting afterwards
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print(
        "### Training for {} epochs, training size = {} cases".format(
            epochs, data.dataset.tensors[0].shape[0]
        )
    )
    for t in range(epochs):
        # Process each mini-batch in turn:
        for x, y_actual in data:
            y_pred = model(x)
            # Compute and print loss
            loss = criterion(
                y_pred.to(dtype=torch.float), y_actual.to(dtype=torch.float)
            )
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Epoch ending, so now fit the coefficients based on all data:
        x, y_actual = data.dataset.tensors
        with torch.no_grad():
            model[1].fit_coeff(x, y_actual)
        # Get the error rate for the whole batch:
        y_pred = model(x)
        mse, rmse, perc_loss = calc_error(y_pred, y_actual)
        errors.append(perc_loss)
        # Print some progress information as the net is trained:
        if epochs < 30 or t % 10 == 0:
            print(model)
            print(
                "epoch {:4d}: MSE={:.5f}, RMSE={:.5f} ={:.2f}%".format(
                    t, mse, rmse, perc_loss
                )
            )
    # End of training, so graph the results:
    if show_plots:
        plotErrors(errors)
        y_actual = data.dataset.tensors[1]
        y_pred = model(data.dataset.tensors[0])
        plotResults(y_actual, y_pred)


if __name__ == "__main__":
    import warnings

    import pandas as pd

    warnings.filterwarnings("ignore", category=FutureWarning)

    indicators = ["sma60", "sma120", "rsi"]
    data_path = r"E:\workspace\Algorithmic-Trading\AlgorithmicStrategy\datas\000333.SZ\snapshot\gtja\2023-08-03.csv"
    data = pd.read_csv(data_path)
    data = data[data["time"] >= 93000000].reset_index(drop=True)
    data.iloc[:, 1:] = data.iloc[:, 1:].astype(float)
    feat = convFeature(2, 45, 60)
    models = make_anfis(2, 5, 2, hybrid=False)
    model = nn.Sequential(feat, models)
    datalader = train_loader(data, 60, -500, 64)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.99)
    criterion = torch.nn.MSELoss(reduction="sum")
    train_anfis_with(model, datalader, optimizer, criterion)
