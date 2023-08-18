import pandas as pd
import numpy as np
from datetime import time
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

sns.set_style("whitegrid")


def volume_order_imbalance(data, kws):
    drop_first = kws.setdefault("drop_first", True)

    current_bid_price = data['bid1_price']

    bid_price_diff = current_bid_price - current_bid_price.shift()

    current_bid_vol = data['bid1_volume']

    nan_ = current_bid_vol[current_bid_vol == 0].index

    bvol_diff = current_bid_vol - current_bid_vol.shift()

    bid_increment = np.where(
        bid_price_diff > 0, current_bid_vol,
        np.where(bid_price_diff < 0, 0,
                 np.where(bid_price_diff == 0, bvol_diff, bid_price_diff)))

    current_ask_price = data['ask1_price']

    ask_price_diff = current_ask_price - current_ask_price.shift()

    current_ask_vol = data['ask1_volume']

    avol_diff = current_ask_vol - current_ask_vol.shift()

    ask_increment = np.where(
        ask_price_diff < 0, current_ask_vol,
        np.where(ask_price_diff > 0, 0,
                 np.where(ask_price_diff == 0, avol_diff, ask_price_diff)))

    _ = pd.Series(bid_increment - ask_increment, index=data.index)

    if drop_first:
        _.loc[_.groupby(_.index.date).apply(lambda x: x.index[0])] = np.nan

    _.loc[nan_] = np.nan

    return _


def price_weighted_pressure(data, kws):
    n1 = kws.setdefault("n1", 1)
    n2 = kws.setdefault("n2", 10)

    bench = kws.setdefault("bench_type", "MID")

    _ = np.arange(n1, n2 + 1)

    if bench == "MID":
        bench_prices = calc_mid_price(data)
    elif bench == "SPECIFIC":
        bench_prices = kws.get("bench_price")
    else:
        raise Exception("")

    def unit_calc(bench_price):
        bid_d = [
            bench_price / (bench_price - data["bid%s_price" % s]) for s in _
        ]
        bid_denominator = sum(bid_d)
        bid_weights = [d / bid_denominator for d in bid_d]
        press_buy = sum([
            data["bid%s_volume" % (i + 1)] * w
            for i, w in enumerate(bid_weights)
        ])

        ask_d = [
            bench_price / (data['ask%s_price' % s] - bench_price) for s in _
        ]
        ask_denominator = sum(ask_d)
        ask_weights = [d / ask_denominator for d in ask_d]
        press_sell = sum([
            data['ask%s_volume' % (i + 1)] * w
            for i, w in enumerate(ask_weights)
        ])

        return (np.log(press_buy) - np.log(press_sell)).replace(
            [-np.inf, np.inf], np.nan)

    return unit_calc(bench_prices)


def calc_mid_price(data):
    return (data['bid1_price'] + data['ask1_price']) / 2


def get_mid_price_change(data, drop_first=True):
    _ = calc_mid_price(data).pct_change()
    if drop_first:
        _.loc[_.groupby(_.index.date).apply(lambda x: x.index[0])] = np.nan
    return _


def indicator_reg_analysis(data, indicator, k=20, m=5):
    r_mid_price = get_mid_price_change(data)

    def reg(x, y):
        _y = y.shift(-1).rolling(k).mean().shift(-k + 1).dropna()
        indep = sm.add_constant(
            pd.concat([x.shift(i) for i in range(m + 1)], axis=1).dropna())

        index = _y.index.intersection(indep.index)
        res = sm.OLS(_y.reindex(index), indep.reindex(index),
                     missing="drop").fit()

        return res

    return r_mid_price.groupby(
        r_mid_price.index.date).apply(lambda y: reg(indicator.loc[y.index], y))


def weighted_price(data, n1, n2):
    _ = np.arange(n1, n2 + 1)
    numerator = sum([
        data['bid%s_volume' % i] * data['bid%s_price' % i] +
        data['ask%s_volume' % i] * data['ask%s_price' % i] for i in _
    ])
    denominator = sum(data['bid%s_volume' % i] + data['ask%s_volume' % i]
                      for i in _)

    return numerator / denominator


def length_imbalance(data, n):
    _ = np.arange(1, n + 1)

    imb = {
        s: (data["bid%s_volume" % s] - data["ask%s_volume" % s]) /
        (data["bid%s_volume" % s] + data["ask%s_volume" % s])
        for s in _
    }

    return pd.concat(imb.values(), keys=imb.keys()).unstack().T


def height_imbalance(data, n):
    _ = np.arange(2, n + 1)

    bid_height = [(data['bid%s_price' % (i - 1)] - data['bid%s_price' % i])
                  for i in _]
    ask_height = [(data['ask%s_price' % i] - data['ask%s_price' % (i - 1)])
                  for i in _]

    r = {
        i + 2: (b - ask_height[i]) / (b + ask_height[i])
        for i, b in enumerate(bid_height)
    }

    r = pd.concat(r.values(), keys=r.keys()).unstack().T

    return r


def intraday_moving_average(price, windows=20):
    return price.groupby(
        price.index.date).apply(lambda x: x.rolling(windows).mean().shift())


def trend_to_ma(price, windows=20, dir="over", return_ma=False):
    ma = intraday_moving_average(price, windows)
    if dir == "over":
        _ = price[price > ma].index
    else:
        _ = price[price < ma].index
    return _, ma if return_ma else _


def sampling_data(data, rule):
    sample_data = data.resample(rule).last()
    return format_times(sample_data)


def format_times(data):
    times_am = (time(9, 30), time(11, 30))
    times_pm = (time(13, 0), time(14, 57))
    _ = data.index.time
    return data[((_ > times_am[0]) & (_ < times_am[1])) | ((_ > times_pm[0]) &
                                                           (_ < times_pm[1]))]


def spread(data, type="relative"):
    s = data['ask1_price'] - data['bid1_price']
    return s / (
        0.5 *
        (data['ask1_price'] + data['bid1_price'])) if type == "relative" else s
