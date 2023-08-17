import pandas as pd
import os
from collections import OrderedDict

class Normalized_reader:
    def __init__(self, filepath) -> None:
        self.filepath = filepath
        self.filenames = []
        self.df_total = dict()
        self.df_input = OrderedDict()
        self.trade_record = OrderedDict()
        self.timestamp_list = []
        for filename in os.listdir(self.filepath):
            if filename.endswith('.csv'):
                self.filenames.append(filename)      
                self.df_total[filename] = pd.read_csv(self.filepath + filename)  
        
    def get_df_input(self, filename):
        self.trade_record = OrderedDict()
        self.df_input = OrderedDict()
        for limit in range(100, len(self.df_total[filename])):
            trade_price = self.df_total[filename].iloc[limit, 1]
            trade_time = self.df_total[filename].iloc[limit, 0]
            self.trade_record[trade_time] = {'trade_volume':None,'trade_price':trade_price} 
            self.df_input[trade_time] = self.df_total[filename].iloc[limit-100:limit, 2:-2]
        
        self.timestamp_list = list(self.df_input.keys())

        return self.df_input, self.trade_record

    

