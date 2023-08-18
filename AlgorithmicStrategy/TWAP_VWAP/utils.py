from pathlib import Path

import numpy as np
import torch as t
from torch import nn, optim
import random
from log import logger
import matplotlib.pyplot as plt
from typing import Sequence, Union
from collections import deque

plt.style.use("seaborn-pastel")


def setup_seed(seed: int):
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    t.backends.cudnn.deterministic = True


def save_model(
        model: nn.Module, optimizer: optim, epoch: int, loss: float, path: str, **kwargs
):
    save_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
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


def plotter(
        x: Sequence, y: Sequence = None, ylabel: str = "", xlabel: str = "epochs", **kwargs
):
    fontdict = kwargs.get("fontdict", {"fontsize": 20})
    plt.figure(figsize=(8, 6))
    if y is None:
        y = range(len(x))
    plt.plot(x, y, lw=2, label=ylabel)
    plt.ylabel(ylabel, fontdict=fontdict)
    plt.xlabel(xlabel, fontdict=fontdict)
    plt.grid(True, alpha=0.7)
    plt.legend(loc="best", fontsize=fontdict["fontsize"])
    if kwargs.get("show", True):
        plt.show()
    if path := kwargs.get("path", ""):
        plt.savefig(Path(__file__).parent / path, dpi=200)


def categorical(
        labels: Union[np.ndarray, t.Tensor, list],
        classes: Union[np.ndarray, t.Tensor, list, int] = None,
):
    """
    :param labels: class vector to be converted into a matrix
            (integers from 0 to num_classes).
    :param classes: total number of classes or class vectors
    :return:
    """
    # Get the number of unique categories
    if classes is None:
        classes = set(labels)

    if isinstance(classes, int):
        num_categories = classes
    else:
        num_categories = max(classes) + 1

    if isinstance(labels, list):
        labels = np.array(labels, dtype=np.int64)

    # Convert the labels to a one-hot encoded tensor
    one_hot_labels = t.zeros(len(labels), num_categories)
    one_hot_labels.scatter_(1, t.tensor(labels).unsqueeze(1), 1)

    return one_hot_labels


class LimitedQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = deque()

    @property
    def size(self):
        return len(self.queue)

    def push(self, item):
        if self.size >= self.max_size:
            self.queue.popleft()  # 移除最老的元素
        self.queue.append(item)

    @property
    def items(self):
        return list(self.queue)




if __name__ == "__main__":
    plotter(range(8), ylabel="acc", show=False, path="./PICS/test.png")
    print(categorical([1, 2, 3, 8]))

    # 创建一个最大容量为5的LimitedQueue
    my_queue = LimitedQueue(5)

    # 添加数据到队列
    my_queue.push(1)
    my_queue.push(2)
    my_queue.push(3)
    my_queue.push(4)
    my_queue.push(5)

    # 打印当前队列中的数据
    print(my_queue.items)  # 输出: [1, 2, 3, 4, 5]

    # 添加新数据，会剔除最老的数据
    my_queue.push(6)
    print(my_queue.items)  # 输出: [2, 3, 4, 5, 6]
