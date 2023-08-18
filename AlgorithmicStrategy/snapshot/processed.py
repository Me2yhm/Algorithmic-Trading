import pandas as pd
import numpy as np


class FeatureEngineering:

    def __init__(self, data):
        self.data = data

    def calculate_spread(self):
        self.data['spread'] = self.data['ask1_price'] - self.data['bid1_price']

    def calculate_volume_diff(self):
        self.data['volume_diff'] = self.data['ask1_volume'] - self.data[
            'bid1_volume']

    def calculate_depth(self):
        self.data[
            'depth'] = self.data['bid1_volume'] + self.data['ask1_volume']

    def calculate_price_change_speed(self):
        self.data['price_change_speed'] = (self.data['last_price'] -
                                           self.data['last_price'].shift(1)
                                           ) / self.data['last_price'].shift(1)

    def calculate_volume(self):
        self.data['volume'] = self.data['volume']

    def calculate_ma(self, window=10):
        self.data['MA'] = self.data['last_price'].rolling(window=window).mean()

    def calculate_rsi(self, window_length=14):
        delta = self.data['last_price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        average_gain = gain.rolling(window_length).mean()
        average_loss = loss.rolling(window_length).mean()
        rs = average_gain / average_loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

    def calculate_atr(self, window=14):
        high_low = self.data['ask1_price'] - self.data['bid1_price']
        high_close = np.abs(self.data['ask1_price'] -
                            self.data['last_price'].shift())
        low_close = np.abs(self.data['bid1_price'] -
                           self.data['last_price'].shift())
        true_range = pd.concat([high_low, high_close, low_close],
                               axis=1).max(axis=1)
        self.data['ATR'] = true_range.rolling(window=window).mean()

    def calculate_stochastic(self, n=14):
        lowest_low = self.data['bid1_price'].rolling(window=n).min()
        highest_high = self.data['ask1_price'].rolling(window=n).max()
        self.data['Stochastic'] = (self.data['last_price'] - lowest_low) / (
            highest_high - lowest_low) * 100

    def calculate_buy_sell_pressure(self):
        self.data['buy_sell_pressure'] = self.data['ask1_volume'] - self.data[
            'bid1_volume']

    def run_feature_engineering(self):
        self.calculate_spread()
        self.calculate_volume_diff()
        self.calculate_depth()
        self.calculate_price_change_speed()
        self.calculate_volume()
        self.calculate_ma()
        self.calculate_rsi()
        self.calculate_atr()
        self.calculate_stochastic()
        self.calculate_buy_sell_pressure()


# 读取数据文件
data = pd.read_csv('../datas/601021.SH/snapshot/gtja/2023-07-20.csv')

# 创建特征工程对象
fe = FeatureEngineering(data)

# 运行特征工程
fe.run_feature_engineering()

fe.data.to_csv('processed_data.csv', index=False)
