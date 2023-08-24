import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from AlgorithmicStrategy import OrderBook,DataSet
import pickle
from tqdm import tqdm

tick_folder = Path(r'D:\算法交易\Algorithmic-Trading\AlgorithmicStrategy\datas\000157.SZ\tick\gtja')
tick_files = list(tick_folder.glob("*.csv"))
pkl_folder = Path(r'D:\算法交易\Algorithmic-Trading\AlgorithmicStrategy\TWAP_VWAP\PKL')
if not pkl_folder.exists():
    pkl_folder.mkdir(parents=True, exist_ok=True)

for tick_file in tqdm(tick_files):
    tick = DataSet(data_path=tick_file, ticker="SZ")
    ob = OrderBook(data_api=tick)
    ob.update()
    with open(pkl_folder.joinpath(str(tick_file.name).replace('.csv','') + ".pkl"), "wb") as file:
        pickle.dump(ob.snapshots, file)


