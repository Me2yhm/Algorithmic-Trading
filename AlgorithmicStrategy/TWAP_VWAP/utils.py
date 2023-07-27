import numpy as np
import torch as t
from torch import nn, optim
import random
from log import logger
import matplotlib.pyplot as plt
from typing import Sequence

plt.style.use("seaborn-pastel")


def setup_seed(seed: int):
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    t.backends.cudnn.deterministic = True


def save_model(
        model: nn.Module,
        optimizer: optim,
        epoch: int,
        loss: float,
        path: str,
        **kwargs):
    save_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    save_dict.update(kwargs)
    t.save(save_dict, path)
    logger.info("Saved model to {}".format(path))

    """
    device = torch.device("cuda")
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  
    # Choose whatever GPU device number you want
    model.to(device)
    """


def plotter(x: Sequence, y: Sequence = None, ylabel: str = '', xlabel: str = "epochs", **kwargs):
    fontdict = kwargs.get('fontdict', {'fontsize': 20})
    plt.figure(figsize=(8, 6))
    if y is None:
        y = range(len(x))
    plt.plot(x, y, lw=2, label=ylabel)
    plt.ylabel(ylabel, fontdict=fontdict)
    plt.xlabel(xlabel, fontdict=fontdict)
    plt.grid(True, alpha=0.7)
    plt.legend(loc='best', fontsize=fontdict['fontsize'])
    plt.show()
    if path := kwargs.get("path", ""):
        plt.savefig(path, dpi=200)


if __name__ == "__main__":
    plotter(range(8), ylabel="acc")
