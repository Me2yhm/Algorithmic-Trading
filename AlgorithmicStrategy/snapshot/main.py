import pandas as pd
# from model import AvellanedaStoikov
from model_1 import TradingSimulator
import data_processing as dp

if __name__ == '__main__':
    """
    Execute here
    """
    snapshot = pd.read_csv('../datas/601021.SH/snapshot/gtja/2023-07-20.csv')
    data = dp.SnapshotTrans(snapshot)

    # Calculate parameters
    sigma_mp, sigma_ap, k = dp.cal_paras(data)

    # Extract necessary data for Avellaneda-Stoikov model
    MP = data['MidPrice'].tolist()
    AP = data['AvgPrice'].tolist()
    time = data['time'].tolist()
    buy = data['bid1_price'].tolist()
    sell = data['ask1_price'].tolist()
    AS_mp = TradingSimulator(prices=MP,
                             T=time,
                             buy=buy,
                             sell=sell,
                             sigma=sigma_mp,
                             model='AS',
                             number_of_simulation=1000,
                             k=k)
    AS_mp.setTrivialDeltaValue(0.05)
    AS_mp.execute()
    AS_mp.visualize()

    AS_ap = TradingSimulator(prices=AP,
                             T=time,
                             buy=buy,
                             sell=sell,
                             sigma=sigma_ap,
                             model='AS',
                             number_of_simulation=1000,
                             k=k)

    AS_ap.execute()
    AS_ap.visualize()