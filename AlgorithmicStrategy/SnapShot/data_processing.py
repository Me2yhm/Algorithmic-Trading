import numpy as np
import math


def SnapshotTrans(snapshot):
    trade = snapshot[(snapshot['time'] >= 93000000)
                     & (snapshot['time'] < 145700000)]
    cols = [
        'time', 'bid1_price', 'bid2_price', 'bid3_price', 'bid4_price',
        'bid5_price', 'bid1_volume', 'bid2_volume', 'bid3_volume',
        'bid4_volume', 'bid5_volume', 'ask1_price', 'ask2_price', 'ask3_price',
        'ask4_price', 'ask5_price', 'ask1_volume', 'ask2_volume',
        'ask3_volume', 'ask4_volume', 'ask5_volume'
    ]
    new_snapshot = trade[cols]

    num_rows = new_snapshot.shape[0]
    MidPrice = []
    AvgPrice = []

    for i in range(num_rows):
        pb1 = new_snapshot.bid1_price.iloc[i]
        pb2 = new_snapshot.bid2_price.iloc[i]
        pb3 = new_snapshot.bid3_price.iloc[i]
        pb4 = new_snapshot.bid4_price.iloc[i]
        pb5 = new_snapshot.bid5_price.iloc[i]
        qb1 = new_snapshot.bid1_volume.iloc[i] * 1000
        qb2 = new_snapshot.bid2_volume.iloc[i] * 10
        qb3 = new_snapshot.bid3_volume.iloc[i]
        qb4 = new_snapshot.bid4_volume.iloc[i] * 0.1
        qb5 = new_snapshot.bid5_volume.iloc[i] * 0.01

        pa1 = new_snapshot.ask1_price.iloc[i]
        pa2 = new_snapshot.ask2_price.iloc[i]
        pa3 = new_snapshot.ask3_price.iloc[i]
        pa4 = new_snapshot.ask4_price.iloc[i]
        pa5 = new_snapshot.ask5_price.iloc[i]
        qa1 = new_snapshot.ask1_volume.iloc[i] * 1000
        qa2 = new_snapshot.ask2_volume.iloc[i] * 10
        qa3 = new_snapshot.ask3_volume.iloc[i]
        qa4 = new_snapshot.ask4_volume.iloc[i] * 0.1
        qa5 = new_snapshot.ask5_volume.iloc[i] * 0.01

        if pa1 == 0 or pb1 == 0:
            pa1 = new_snapshot.ask1_price.iloc[i - 1]
            qa1 = new_snapshot.ask1_volume.iloc[i - 1] * 1000
            pb1 = new_snapshot.bid1_price.iloc[i - 1]
            qb1 = new_snapshot.bid1_volume.iloc[i - 1] * 1000

        mp = ((pb1 * qb1 + pb2 * qb2 + pb3 * qb3 + pb4 * qb4 + pb5 * qb5) /
              (qb1 + qb2 + qb3 + qb4 + qb5) +
              (pa1 * qa1 + pa2 * qa2 + pa3 * qa3 + pa4 * qa4 + pa5 * qa5) /
              (qa1 + qa2 + qa3 + qa4 + qa5)) / 2
        MidPrice.append(mp)

        ap = (pb1 + pa1) / 2
        AvgPrice.append(ap)

        t = new_snapshot.time.iloc[i]
        t = t / 1000
        hour = t // 10000
        minute = (t - hour * 10000) // 100
        second = t % 100
        new_snapshot.time.iloc[i] = (hour - 9) * 3600 + minute * 60 + second

    new_snapshot['MidPrice'] = MidPrice
    new_snapshot['AvgPrice'] = AvgPrice

    return new_snapshot


def cal_paras(new_snapshot):
    num_rows = new_snapshot.shape[0]
    std_mp = 0
    std_ap = 0
    count = 0
    mp0 = new_snapshot.MidPrice.iloc[0]
    ap0 = new_snapshot.AvgPrice.iloc[0]

    for i in range(1, num_rows):
        t1 = new_snapshot.time.iloc[i]
        t2 = new_snapshot.time.iloc[i - 1]
        mp = new_snapshot.MidPrice.iloc[i] - mp0
        ap = new_snapshot.AvgPrice.iloc[i] - ap0
        count += 1

        if t1 == t2:
            count -= 1
        else:
            std_mp += (mp**2) / (t1 - t2)
            std_ap += (ap**2) / (t1 - t2)

    sigma_mp = math.sqrt(std_mp / count)
    sigma_ap = math.sqrt(std_ap / count)

    spread = np.mean(new_snapshot['ask1_price']) - np.mean(
        new_snapshot['bid1_price'])
    k = 2 / spread

    return sigma_mp, sigma_ap, k
