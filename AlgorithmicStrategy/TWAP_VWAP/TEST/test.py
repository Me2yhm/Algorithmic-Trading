from AlgorithmicStrategy import Writer, OrderBook, DataSet
from pathlib import Path
from AlgorithmicStrategy import TradeTime
tick_path = Path.cwd().parents[1] / "datas/002703.SZ/tick/gtja/2023-07-03.csv"
tick = DataSet(data_path=tick_path, ticker="SZ")
tt = TradeTime(begin=93000000, end=145700000, tick=tick)
time_dict = tt.generate_signals()
print(len(time_dict))
print(time_dict[-1])
update = tt.generate_timestamps('update', interval=3000, limits=(0,0))
print(len(update))
print(list(update)[0])
