from MODELS import TWAP
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from AlgorithmicStrategy import DataSet, OrderBook


if __name__ == "__main__":
    cwd = Path(__file__).parent
    tick_folder = cwd / "../datas/000157.SZ/tick/gtja/"
    tick_files = list(tick_folder.glob("*.csv"))
    total_begin = 9_15_00_000
    trade_begin = 9_30_03_000
    end = 14_57_00_000
    for tick_file in tick_files:
        tick = DataSet(data_path=tick_file, ticker="SZ")
        ob = OrderBook(data_api=tick)

        twap = TWAP(ob, tick, 2400, 6000, 2500, "BUY")
        twap.get_time_dict()
        twap.get_trade_times()
        for i in range(
            twap.tick.file_date_num + trade_begin, twap.tick.file_date_num + end
        ):
            twap.timeStamp = i
            twap.signal_update()
            if twap.trade:
                twap.strategy_update()
                print(twap.signal)
                print(twap.ts_list)
                print(twap.vwap_loss)
