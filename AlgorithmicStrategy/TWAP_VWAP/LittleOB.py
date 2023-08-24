import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from AlgorithmicStrategy import OrderBook,DataSet
import csv
from tqdm import tqdm
from collections import OrderedDict

tick_folder = Path(r'D:\算法交易\Algorithmic-Trading\AlgorithmicStrategy\datas\000157.SZ\tick\gtja')
tick_files = list(tick_folder.glob("*.csv"))
csv_folder = Path(r'D:\算法交易\Algorithmic-Trading\AlgorithmicStrategy\TWAP_VWAP\LittleOB')
if not csv_folder.exists():
    csv_folder.mkdir(parents=True, exist_ok=True)
little_ob = OrderedDict()
columns = ['Timestamp', 'ask1', 'bid1', 'VWAP']

for tick_file in tqdm(tick_files):
    tick = DataSet(data_path=tick_file, ticker="SZ")
    ob = OrderBook(data_api=tick)
    ob.update()
    for ts in ob.snapshots.keys():
        if ts%1000000000 >= 93000000 and ts%1000000000 <= 145700000:
            if list(ob.snapshots[ts]['ask'].keys()):
                little_ob[ts] = OrderedDict()
                little_ob[ts]['ask1'] = list(ob.snapshots[ts]['ask'].keys())[0]
            if list(ob.snapshots[ts]['bid'].keys()):
                little_ob[ts]['bid1'] = list(ob.snapshots[ts]['bid'].keys())[0]
            if list(ob.snapshots[ts]['total_trade'].values()):
                little_ob[ts]['VWAP'] = list(ob.snapshots[ts]['total_trade'].values())[2]
    data_list = []
    for ts, values in little_ob.items():
        data_list.append([ts] + [values.get(key, '') for key in columns[1:]])
    with open(csv_folder.joinpath(tick_file.name.replace('.csv','littleob')+'.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns) 
        writer.writerows(data_list)


