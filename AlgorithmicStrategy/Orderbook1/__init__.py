
__author__ = ['Alkaid']
from DataManager import DataSet
from OrderBook import OrderBook
from pathlib import Path
from AbstractFeatures import Pastfeature
from depth import OrderDepthCalculator

#测试函数
def test_iter(range1):
    for i in range(0, range1):
        datas = tick.fresh()
        ob.single_update(datas)

def show_oid_map(oid:int, ob:OrderBook):
    print(ob.oid_map[oid])    

def show_order_num(price1:float, price2:float):
    print("-----")
    print('ask_num')
    print(ob.last_snapshot["ask_num"][price1])
    print('bid_num')
    print(ob.last_snapshot["bid_num"][price2])    
    print("-----")

def show_total_order_num():
    print("-----")
    total = 0
    for j in ob.last_snapshot["bid_num"].values(): total += j
    print('bid_num ', total)
    total = 0
    for j in ob.last_snapshot["ask_num"].values(): total += j
    print('ask_num ', total)
    print("-----")


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    data_api = current_dir.parent / "datas/000001.SZ/tick/gtja/2023-07-03.csv"
    tick = DataSet(data_api, date_column="time", ticker="000001.SZ")
    ob = OrderBook(data_api=tick)
    candle = Pastfeature(data_api=tick)

    if 0:
        ob.update(20230703093700130)
        #模块1 任意时刻的candle数据
        print(ob.search_candle(20230703093700130))

        #模块2 任意时段的candle数据
        print(ob.get_candle_slot(20230703093000000, 20230703093700130))
        # print(ob.candle_tick[20230508093000150]) #前一次收盘、开盘、最高、最低、收盘

    if 1:
        ob.update(20230703103700130)
        #模块2 任意时刻的超级盘口，包含买卖十档、买卖十档交易量、买卖十档订单数、买卖十档累计新陈代谢
        print(ob.get_super_snapshot(10, 20230703103700130))
     

    if 0:
        ob.update(20230703093700130)
        #模块3 任意时刻成交单VWAP、成交量、成交单数、被动方新陈代谢
        print(ob.search_snapshot(20230703093700130)["total_trade"])
        #模块3 任意时间段成交单VWAP、成交量、成交单数、被动方新陈代谢
        print(ob._get_avg_trade(20230703093700130, 20230703094000130)) 

    if 0:
        ob.update(20230703093700130)
        #模块4 任意时刻的平均市场深度
        print(ob.search_snapshot(20230703093700130)["order_depth"]["weighted_average_depth"])


    if 0 :
        #功能测试
        test_iter(6861)
        # show_total_order_num()
        # show_oid_map(136)
        # show_oid_map(29897)
        # show_oid_map(396)
        # show_oid_map(224385)    
        # print(ob.last_snapshot["candle_tick"][20230508092459840])
        # print(ob.last_snapshot["ask_order_stale"])
        # print(ob.last_snapshot["bid_order_stale"])
        print(ob.last_snapshot["order_depth"])
        # print(ob.last_snapshot["bid_num_death"])
        test_iter(3)
        # show_total_order_num()
        # show_oid_map(136)
        # show_oid_map(29897)
        # show_oid_map(396)
        # show_oid_map(224385)
        print(ob.last_snapshot["total_trade"])
        print(ob.last_snapshot["candle_tick"])
        # print(ob.last_snapshot["ask_order_stale"])
        # print(ob.last_snapshot["bid_order_stale"])
        print(ob.last_snapshot["order_depth"])
        # print(ob.last_snapshot["bid_num_death"])

        # print(ob._get_avg_stale(20230703092459840, 20230703092500000))
        # print(ob._get_avg_trade(20230703092459840, 20230703092500000))

    if 0 :
        n = 10 #盘口挡位
        decay_rates = [2]#衰减率
        test_iter(6861)
        # show_oid_map(136)
        # show_oid_map(29897)
        # show_oid_map(396)
        # show_oid_map(224385)    
        order_depth_calculator = OrderDepthCalculator(ob.last_snapshot, n)

        for decay_rate in decay_rates:
            weighted_average_depth = order_depth_calculator.calculate_weighted_average_depth(decay_rate)
            print(f"衰减率为 {decay_rate} 时，买卖{n}档的加权平均订单深度: {weighted_average_depth}")
            print(order_depth_calculator.n_depth)  
            print(order_depth_calculator.total_volume)        

    if 0:
        candle.update(20230703093153280)
        print(candle.tick_volume, candle.order_num, candle.previous_close, candle.open, candle.close, candle.high, candle.low)
