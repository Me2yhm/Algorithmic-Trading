__author__ = ['Alkaid']
from DataManager import DataSet
from OrderBook import OrderBook
from pathlib import Path
from Depth import OrderDepthCalculator
from Writer import Writer
import datetime
from datetime import datetime, timedelta
from Normalizer import Normalizer
from copy import deepcopy

def get_date_data(current_dir, date:str, ticker:str):
    current_dir = Path(__file__).parent
    date_name = date[:4] + '-' + date[4:6] + '-' + date[6:]
    data_api = current_dir.parent / str("datas/" + ticker + "/tick/gtja/" + date_name + ".csv")
    tick = DataSet(data_api, date_column="time", ticker = ticker)
    ob = OrderBook(data_api=tick)
    ob.update(int(date+'145700000'))
    writer_3s = Writer("002703_3s_"+date[4:]+".csv", None, rollback = 3000, bid_ask_num = 10)
    writer_3s.collect_data_order_book(ob)



if __name__ == "__main__":
    current_dir = Path(__file__).parent

    if 0:
        data_api = current_dir.parent / "datas/000157.SZ/tick/gtja/2023-07-04.csv"
        tick = DataSet(data_api, date_column="time", ticker="000157.SZ")
        ob = OrderBook(data_api=tick)
        ob.update(20230704145700000)
        writer_3s = Writer("002703_3s_0704.csv", None, rollback = 3000, bid_ask_num = 10)
        writer_3s.collect_data_order_book(ob)

    if 0:
        days = ['03', '04', '05', '06', '07', '10', '11', '12', '13', '14', '17', '18', '19', '20', '21', '24', '25']
        for date in ['202307'+ day for day in days]:
            print("processing " + date)
            get_date_data(current_dir, date, "000157.SZ")

    if 0:
        t1 = 20230703093000000
        t1 = datetime.strptime(str(t1), '%Y%m%d%H%M%S%f')
        t2 = 20230703093100000
        t2 = datetime.strptime(str(t2), '%Y%m%d%H%M%S%f')
        time_interval = timedelta(milliseconds=1)
        t3 = t2 - t1
        print(t3)
        print(t3/time_interval)
        print(3*1e5)

    if 0:
        days = ['26', '27', '28']
        for date in ['202307'+ day for day in days]:
            print("processing " + date)
            get_date_data(current_dir, date, "000157.SZ")

        days = ['02', '03']
        for date in ['202308'+ day for day in days]:
            print("processing " + date)
            get_date_data(current_dir, date, "000157.SZ")

    if 0:
        file_path = 'C:/Users/14913/Documents/GitHub/Algorithmic-Trading/AlgorithmicStrategy/Orderbook1/训练集3s/'
        output_path = 'C:/Users/14913/Documents/GitHub/Algorithmic-Trading/AlgorithmicStrategy/Orderbook1/归一化训练集3s/'
        Normalizer_002703 = Normalizer(file_path, is_train = True)
        Normalizer_002703.normalized_csv_files(True, output_path)
        # Normalizer_002703.get_csv_filenames()
        # Normalizer_002703.get_continuous_params()
        # Normalizer_002703.get_dis_norm()
        # df_o = deepcopy(Normalizer_002703.df_origin)
        # df = Normalizer_002703.to_normalize(df_o)
        # print(df.columns)
        print(Normalizer_002703.df_his_feature)
    if 0:
        0

    if 1:
        file_path_train = 'C:/Users/14913/Documents/GitHub/Algorithmic-Trading/AlgorithmicStrategy/Orderbook1/训练集3s/'
        # file_path_test = 'C:/Users/14913/Documents/GitHub/Algorithmic-Trading/AlgorithmicStrategy/Orderbook1/测试集3s/'
        output_path = 'C:/Users/14913/Documents/GitHub/Algorithmic-Trading/AlgorithmicStrategy/Orderbook1/归一化测试集3s/'
        Normalizer_002703 = Normalizer(file_path_train)
        Normalizer_002703.initialize_output(is_train = True, output_path = None, output = False)
        # Normalizer_002703.folder_path = file_path_test
        # Normalizer_002703.initialize_output(False, output_path, True)
        data_api = current_dir.parent / "datas/000157.SZ/tick/gtja/2023-07-04.csv"
        tick = DataSet(data_api, date_column="time", ticker="000157.SZ")
        ob = OrderBook(data_api=tick)
        Normalizer_002703.normalize_by_time(ob, 20230704093100000, 3000, 'None')
        Normalizer_002703.normalize_by_time(ob, 20230704093200000, 3000, 'None')
        Normalizer_002703.normalize_by_time(ob, 20230704093200000, 3000, 'None')
        print(Normalizer_002703.df_total)


        