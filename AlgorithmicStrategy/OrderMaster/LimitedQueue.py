import pandas as pd
from collections import deque

class LimitedQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = deque()

    @property
    def size(self):
        return len(self.queue)
    

    #接收一行数据并生成矩阵
    def push(self, item: pd.DataFrame):
        if self.size >= self.max_size:
            self.queue.popleft()  # 移除最老的元素
            self.queue.append(item)

        else:
            self.queue.append(item)
            if self.size == self.max_size:
                self.df = pd.concat(self.queue, ignore_index=True,axis=1)

    def to_df(self):
        return pd.concat(self.queue, ignore_index=True)

    @property
    def items(self):
        return list(self.queue)

    def form_matrix(self):
        matrix = None
        if self.df.shape[0]:  #队列中有100条数据才生成矩阵
            matrix = self.df.values
            return matrix.T
        else:
            return matrix
