import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from AlgorithmicStrategy import OrderBook,DataSet
import csv
from collections import OrderedDict
from tqdm import tqdm
class LlobWriter:

    def __init__(self,tick:DataSet,orderbook:OrderBook,file_name:Path):
        self.tick = tick
        self.ob = orderbook
        self.columns = ['Timestamp', 'ask1', 'bid1', 'VWAP']
        self.little_ob = OrderedDict()
        self.file_name = file_name
    

    def write_llob(self):
        for ts in self.ob.snapshots.keys():
            if ts%1000000000 >= 93000000 and ts%1000000000 <= 145700000:
                if list(self.ob.snapshots[ts]['ask'].keys()):
                    self.little_ob[ts] = OrderedDict()
                    self.little_ob[ts]['ask1'] = list(self.ob.snapshots[ts]['ask'].keys())[0]
                if list(self.ob.snapshots[ts]['bid'].keys()):
                    self.little_ob[ts]['bid1'] = list(self.ob.snapshots[ts]['bid'].keys())[0]
                if list(self.ob.snapshots[ts]['total_trade'].values()):
                    self.little_ob[ts]['VWAP'] = list(self.ob.snapshots[ts]['total_trade'].values())[2]
        data_list = []
        for ts, values in self.little_ob.items():
            data_list.append([ts] + [values.get(key, '') for key in self.columns[1:]])
        with open(str(self.file_name)+'.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.columns) 
            writer.writerows(data_list)





if __name__ == "__main__":
    tick_folder = Path.cwd() / "../datas/000157.SZ/tick/gtja/"
    tick_files = list(tick_folder.glob("*.csv"))
    ticker = "000157.SZ"
    file_name = 'llob'

    for tick_file in tqdm(tick_files):
        tick = DataSet(tick_file, ticker=ticker)
        ob = OrderBook(data_api=tick)
        llob = LlobWriter(tick,ob,Path().cwd()/file_name)
        ob.update()

        llob.write_llob()