import skorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
from sklearn.decomposition import PCA
import sys
import numpy as np


sys.path.append("e:\\workspace\\Algorithmic-Trading\\")
from AlgorithmicStrategy.momentum_stratgy.anfis_pytorch.anfis import AnfisNet
from AlgorithmicStrategy.momentum_stratgy.anfis_pytorch.membership import make_gauss_mfs
from AlgorithmicStrategy.momentum_stratgy.dataloader import train_loader
from AlgorithmicStrategy.momentum_stratgy.anfis_pytorch.experimental import (
    calc_error,
    plotErrors,
    plotResults,
)


def get_zscore(pcg: list):
    pcg = np.where(np.isinf(pcg), 0, pcg)
    zscores = [(pcg[i] - np.mean(pcg)) / np.std(pcg) for i in range(len(pcg))]
    return zscores


class convFeature(nn.Module):

    def __init__(self, num_classes, input_dim, seq_len):

        super(convFeature, self).__init__()
        # 加载预训练的VGG16模型
        self.input_dim = input_dim
        self.seq_len = seq_len

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
            nn.Flatten(),
        )

        # 全连接层
        self.pca = PCA(n_components=num_classes)
        self.num_cls = num_classes

    def cal_zscore(self, pcg):
        zscores = [get_zscore(pcg[:, i]) for i in range(self.num_cls)]
        return torch.tensor(zscores).T

    def forward(self, x):

        # 提取图像特征

        with torch.no_grad():

            features = self.cnn(x)

        features_pca = self.pca.fit_transform(features.numpy())
        output = torch.tensor(features_pca, dtype=float)

        return output


class normalize(nn.Module):
    def __init__(self, num_class: int):
        super(normalize, self).__init__()
        self.num_class = num_class

    def cal_zscore(self, x):
        if torch.all(torch.eq(x, torch.zeros_like(x))):
            return x
        mean = x.mean(dim=0)
        std = x.std(dim=0)
        return (x - mean) / std

    def tag_zs(self, x) -> list:
        con = [
            x >= 1.5,
            (x < 1.5) & (x >= 0.5),
            (x < 0.5) & (x >= -0.5),
            (x < -0.5) & (x >= -1.5),
            x < 1.5,
        ]
        vals = [
            torch.tensor(4.0, requires_grad=True),
            torch.tensor(3.0, requires_grad=True),
            torch.tensor(2.0, requires_grad=True),
            torch.tensor(1.0, requires_grad=True),
            torch.tensor(0.0, requires_grad=True),
        ]
        x = torch.where(
            con[0],
            vals[0],
            torch.where(
                con[1],
                vals[1],
                torch.where(
                    con[2],
                    vals[2],
                    torch.where(con[3], vals[3], vals[4]),
                ),
            ),
        )
        return x

    def forward(self, x):
        output = self.cal_zscore(x)
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


def make_model(input_dim: int, seq_len: int):
    feat = convFeature(2, input_dim, seq_len)
    models = make_anfis(2, 5, 2)
    norm = normalize(2)
    model = nn.Sequential(feat, models, norm)
    return model


def train_model(datalader, model):
    X, y = datalader.dataset.tensors
    net = skorch.NeuralNet(
        model,
        max_epochs=20,
        train_split=None,
        criterion=torch.nn.MSELoss,
        # criterion__reduction="sum",
        optimizer=torch.optim.SGD,
        optimizer__lr=1e-4,
        optimizer__momentum=0.99,
    )
    net.fit(X, y)
    return model


if __name__ == "__main__":
    pass
