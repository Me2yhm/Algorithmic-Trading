from collections import defaultdict, deque


class account:
    def __init__(self, vwap, total_volume=10000000, len=80) -> None:
        self.total_volume = total_volume
        self.vwap = vwap
        self.remain_volume = self.total_volume
        self.deals = defaultdict(list)
        self.cost = 0
        self.unit = total_volume / len
        self.has_signal = False
        self.prices = []

    def get_wins(self, marketprice):
        return [1 if p < marketprice else 0 for p in self.prices]

    def cal_win_rate(self, marketprice):
        wintimes = self.get_wins(marketprice)
        return round((sum(wintimes) / len(wintimes)) * 100, 2)

    def execute_signal(self, signal, snap):
        if signal[0] >= snap["ask1_price"]:
            price = snap["ask1_price"]
            remain_volume = self.remain_volume
            self.remain_volume = max(remain_volume - signal[1], 0)
            vol = remain_volume - self.remain_volume
            self.cost += vol * price
            time = snap["time"]
            if vol > 0:
                self.deals[time] = [price, vol]
                self.prices.append(price)
            self.has_signal = False

    def cal_avcost(self):
        return self.cost / (self.total_volume - self.remain_volume)

    def cal_bp(self):
        return round((self.vwap - self.cal_avcost()) / self.vwap * 10000, 2)

    def tag_zs(self, zs):
        if zs >= 1.5:
            return 2
        elif 0.5 <= zs < 1.5:
            return 1
        elif -0.3 < zs < 0.3:
            return 0
        elif -1.5 < zs <= -0.3:
            return -1
        else:
            return -2

    def generate_sig(self, pred, snap):
        tag = list(map(self.tag_zs, pred))
        signal = []
        if tag[0] >= 0:
            price = snap["ask1_price"]
        elif tag[0] == -1:
            price = snap["bid1_price"]
        else:
            price = snap["bid2_price"]
        if tag[1] == 2:
            vol = 5 * self.unit
        elif tag[1] == 1:
            vol = 3 * self.unit
        elif tag[1] == 0:
            vol = 2 * self.unit
        elif tag[1] == -1:
            vol = self.unit
        else:
            vol = 0
        signal.extend([price, vol])
        if signal[1] > 0:
            self.has_signal = True
        return signal


def get_line(zscore_pri_ind, i):
    return zscore_pri_ind.iloc[i, :].tolist()


def is_full(d: deque):
    if len(d) == d.maxlen:
        return True
    else:
        return False
