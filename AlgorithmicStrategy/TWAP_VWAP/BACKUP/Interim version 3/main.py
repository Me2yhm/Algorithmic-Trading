__author__ = ['Alkaid']
from DataManager import DataSet
from OrderBook import OrderBook
from pathlib import Path
from Depth import OrderDepthCalculator
from Writer import Writer
import datetime
from datetime import datetime, timedelta
from Normalizer import Normalizer
from Normalized_reader import Normalized_reader
from Time import *
from copy import deepcopy
import warnings
import pandas as pd
import os
warnings.filterwarnings('ignore')

def get_date_data(current_dir, date:str, ticker:str, trade_delay = 500):
    current_dir = Path(__file__).parent
    date_name = date[:4] + '-' + date[4:6] + '-' + date[6:]
    data_api = current_dir.parent / str("datas/" + ticker + "/tick/gtja/" + date_name + ".csv")
    tick = DataSet(data_api, date_column="time", ticker = ticker)
    ob = OrderBook(data_api=tick)
    end_stamp = date+'145700000'
    ob.update(int(end_stamp))
    writer_3s = Writer("000157_3s_"+date[4:]+".csv", None, rollback = 3000, bid_ask_num = 10)
    writer_3s.collect_data_order_book(ob, end_stamp, trade_delay)



if __name__ == "__main__":
    current_dir = Path(__file__).parent

    if 0:
        #盘口更新功能测试
        data_api = current_dir.parent / "datas/000157.SZ/tick/gtja/2023-07-06.csv"
        tick = DataSet(data_api, date_column="time", ticker="000157.SZ")
        ob = OrderBook(data_api=tick)
        ob.update(20230706145700000)
        print(ob.last_snapshot['timestamp'])

    # if 0:
    #     get_date_data(current_dir, "20230706", "000157.SZ")

    if 0:
        #取训练集盘口
        days = ['03', '04', '05', '06', '07', '10', '11', '12', '13', '14', '17', '18', '19', '20', '21', '24', '25']
        for date in ['202307'+ day for day in days]:
            print("processing " + date)
            get_date_data(current_dir, date, "000157.SZ")


    if 0:
        #取测试集盘口
        days = ['26', '27', '28']
        for date in ['202307'+ day for day in days]:
            print("processing " + date)
            get_date_data(current_dir, date, "000157.SZ")

        days = ['02', '03']
        for date in ['202308'+ day for day in days]:
            print("processing " + date)
            get_date_data(current_dir, date, "000157.SZ")

    # if 0:
    #     file_path = 'C:/Users/14913/Documents/GitHub/Algorithmic-Trading/AlgorithmicStrategy/Orderbook1/训练集3s/'
    #     output_path = 'C:/Users/14913/Documents/GitHub/Algorithmic-Trading/AlgorithmicStrategy/Orderbook1/归一化训练集3s/'
    #     Normalizer_002703 = Normalizer(file_path, is_train = True)
    #     Normalizer_002703.normalized_csv_files(True, output_path)
    #     # Normalizer_002703.get_csv_filenames()
    #     # Normalizer_002703.get_continuous_params()
    #     # Normalizer_002703.get_dis_norm()
    #     # df_o = deepcopy(Normalizer_002703.df_origin)
    #     # df = Normalizer_002703.to_normalize(df_o)
    #     # print(df.columns)
    #     print(Normalizer_002703.df_his_feature)


    if 0:
        #Normalizer功能测试
        file_path_train = 'C:/Users/14913/Documents/GitHub/Algorithmic-Trading/AlgorithmicStrategy/Orderbook1/训练集3s/'
        file_path_test = 'C:/Users/14913/Documents/GitHub/Algorithmic-Trading/AlgorithmicStrategy/Orderbook1/测试集3s/'
        output_path_train = 'C:/Users/14913/Documents/GitHub/Algorithmic-Trading/AlgorithmicStrategy/Orderbook1/归一化训练集3s/'
        output_path_test = 'C:/Users/14913/Documents/GitHub/Algorithmic-Trading/AlgorithmicStrategy/Orderbook1/归一化测试集3s/'
        Normalizer_002703 = Normalizer(file_path_train)
        Normalizer_002703.initialize_output(is_train = True, output_path = output_path_train, output = True)
        Normalizer_002703.file_folder = file_path_test
        Normalizer_002703.initialize_output(False, output_path_test, True)

        # trade_update_time = Trade_Update_time(start_timestamp = "093000000", end_timestamp = "145700000")
        # trade_update_time.get_trade_update_time()
        # limit = 0
        # trade_record = {}
        # df_input = pd.DataFrame
        # for time, attibute in trade_update_time.time_dict.items():
        #     if limit >= 100:
        #         if attibute['trade'] == True: 
        #             df_input = Normalizer_002703.df_total
        #             trade_volume = None
        #             trade_price = ob.search_candle(20230704e9+int(time)+3000)[4]
        #             trade_record[(20230704e9+int(time)+3000)] = {'trade_volume':None,
        #                                                         'trade_price':trade_price}
                    
        #     if attibute['update'] == True:
        #         Normalizer_002703.normalize_by_time(ob, 20230704e9+int(time)+3000, 3000, 'None')      
        #         limit = limit + 1
            
        #     if limit == 98: 
        #         break

        # print(trade_record)
        # print(df_input.shape)

    # if 0:
    #     # data_api = current_dir.parent / "datas/000157.SZ/tick/gtja/2023-07-04.csv"
    #     # tick = DataSet(data_api, date_column="time", ticker="000157.SZ")
    #     # ob = OrderBook(data_api=tick)
    #     # ob.update(20230704e9+145700000)
    #     trade_update_time = Trade_Update_time(start_timestamp = "093000000", end_timestamp = "145700000")
    #     trade_update_time.get_trade_update_time()
    #     limit = 0
    #     trade_record = {}
    #     df_input = pd.DataFrame()
    #     trade_type = 'ask'
    #     path_train = 'C:/Users/14913/Documents/GitHub/Algorithmic-Trading/AlgorithmicStrategy/Orderbook1/归一化训练集3s/'
    #     total_filenames = []
    #     for filename in os.listdir(path_train):
    #         if filename.endswith('.csv'):
    #             total_filenames.append(filename)
        
    #     df_norm0704 = pd.read_csv(path_train + "norm000157_3s_0704.csv")
    #     # for time, attibute in trade_update_time.time_dict.items():
    #         # if limit >= 100:
    #             # if attibute['trade'] == True: 
    #     for limit in range(100, len(df_norm0704)):
    #         df_input = df_norm0704.iloc[limit-100:limit]
    #         trade_volume = None 
    #         if trade_type == 'ask':
    #             trade_price = df_norm0704.iloc[limit, 5] # 用卖一交易
    #         else:
    #             trade_price = df_norm0704.iloc[limit, 25] # 用买一交易
    #         trade_record[limit-100] = {'trade_volume':None,'trade_price':trade_price} #
        
    #     print(trade_record)
    #     print(df_input.shape)


    if 0:
        print(20230704e9)

    if 1: 
        #Normalized_reader 功能测试
        path_train = r"D:\Fudan\Work\JoyE_work\AlgorithmicStrategy\AlgorithmicStrategy\Orderbook1\归一化训练集3s\\"
        Normalized_reader_000157 = Normalized_reader(path_train)
        normalized_000157_0703 = Normalized_reader_000157.filenames[0]
        Normalized_reader_000157.get_df_input(normalized_000157_0703)
        timestamplist = Normalized_reader_000157.timestamp_list
        print(Normalized_reader_000157.trade_record[timestamplist[0]])
        print(Normalized_reader_000157.df_input[timestamplist[0]].shape)

        print(Normalized_reader_000157.trade_record[timestamplist[1000]])
        print(Normalized_reader_000157.df_input[timestamplist[1000]].shape)


        