import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from AlgorithmicStrategy import OrderBook,DataSet
import csv
from tqdm import tqdm
from collections import OrderedDict

class LittleOB:

    def __init__(self,tick:DataSet,orderbook:OrderBook):
        self.tick = tick
        self.ob = orderbook
        self.columns = ['Timestamp', 'ask1', 'bid1', 'VWAP']
        self.little_ob = OrderedDict()
    

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
        with open(csv_folder.joinpath(tick_file.name.replace('.csv','littleob')+'.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.columns) 
            writer.writerows(data_list)


