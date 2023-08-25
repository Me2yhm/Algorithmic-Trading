from TWAP import TWAP
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from AlgorithmicStrategy import DataSet,OrderBook





if __name__ == "__main__":

    tick_folder = Path(r'D:\算法交易\Algorithmic-Trading\AlgorithmicStrategy\datas\000157.SZ\tick\gtja')
    tick_files = list(tick_folder.glob("*.csv"))
    for tick_file in tick_files:
        tick = DataSet(data_path=tick_file, ticker="SZ")
        ob = OrderBook(data_api=tick)
        break

    twap = TWAP(ob,tick,2400,1500,'000157.SZ','BUY')
    twap.get_time_dict()
    for i in range(20230703093003000,20230703145700000):
        twap.timeStamp = i
        twap.signal_update()
        if twap.trade:
            twap.stratgy_update()
            print(twap.signal)
            print(twap.ts_list)
            print(twap.delta_vwap)


