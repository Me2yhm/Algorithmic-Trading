import pandas as pd
from collections import deque


class LimitedQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = deque()
        self.columns = None

    @property
    def size(self):
        return len(self.queue)

    # 接收一行数据并生成矩阵
    def push(self, item: pd.DataFrame):
        if self.size >= self.max_size:
            self.queue.popleft()  # 移除最老的元素
            self.queue.append(item)
        else:
            self.queue.append(item)

    def to_df(self):
        return pd.concat(self.queue, ignore_index=True)
        # return pd.DataFrame(self.queue)

    @property
    def items(self):
        return list(self.queue)

    def clear(self):
        self.queue.clear()
