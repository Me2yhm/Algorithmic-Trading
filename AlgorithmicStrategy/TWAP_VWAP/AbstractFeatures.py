from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / "../OrderMaster"))
from DataManager import DataSet, DataStream, OT
from typing import Union


class Pastfeature:
    def __init__(self, data_api: Union[DataStream, DataSet]):
        self.vwap: float = None
        self.tick_volume: float = None
        self.order_num: float = None
        self.previous_close: float = None
        self.close: float = None
        self.open: float = None
        self.high: float = None
        self.low: float = None
        self.data_api: Union[DataStream, DataSet] = data_api
        self.data_cache: list[OT] = []
        self.data: list[OT] = []
        self.pf_list : list = []

    def next_batch(self, until: int = None) -> list[OT]:
        if self.data_cache:
            # 如果缓存有数据，先处理缓存的数据
            data = self.data_cache.pop(0)
        else:
            data = self.data_api.fresh()[0]
        timestamp_this = data[self.data_api.date_column]

        if until is not None and timestamp_this > until:
            self.data_cache.append(data)
            return []

        res = [data]
        while True:
            try:
                next_data = self.data_api.fresh()
            except StopIteration:
                break
            timestamp_next = next_data[0][self.data_api.date_column]

            if timestamp_next != timestamp_this:
                self.data_cache.extend(next_data)
                break
            else:
                res.extend(next_data)

        return res

    def get_datas(self, until):
        datas = []
        while True:
            try:
                data = self.next_batch(until=until)
                self.get_candlestick(data)
                if data:
                    datas.extend(data)
                else:
                    break
            except StopIteration:
                break
        return datas

    def get_tick_volume(self, datas):
        tick_volume = 0
        for data in datas:
            if data["oid"] == 0 or data["ptype"] == 0:
                tick_volume += data["volume"]
        return tick_volume

    def get_order_num(self, datas):
        oid_list_non = []
        oid_list = []
        order_num = 0
        for data in datas:
            if data["oid"] != 0 or data["ptype"] != 0:
                oid_list.append(data["oid"])
        oid_list_non = list(set(oid_list))
        order_num = len(oid_list_non)
        return order_num

    def get_previous_close(self, datas):
        previous_close = 0.00
        for data in datas:
            if (data["oid"] == 0 or data["ptype"] == 0) and data["price"]:
                previous_close = data["price"]
        return previous_close

    def get_candlestick(self, datas):
        candle = []
        for data in datas:
            if (data["oid"] == 0 or data["ptype"] == 0) and data["price"]:
                candle.append(float(data["price"]))
        if candle:
            self.open = candle[0]
            self.close = candle[-1]
            self.high = max(candle)
            self.low = min(candle)

    def get_all_previous(self, until):
        datas = self.get_datas(until=until)
        self.tick_volume = self.get_tick_volume(datas)
        self.order_num = self.get_order_num(datas)
        self.previous_close = self.get_previous_close(datas)

    def get_all(self, until):
        datas = self.get_datas(until=until)
        self.tick_volume = self.get_tick_volume(datas)
        self.order_num = self.get_order_num(datas)

#获取单个时间戳的特征
    def update(self, until):
        self.get_all_previous(until=until - 1)
        self.get_all(until=until)

#获取时间段内的特征
    def get_period(self, time, start: int = None, stop: int = None):
        #如果不给定时间就从开盘开始到收盘截至（连续竞价阶段）
        if start is None:
            start = 20230301092500000
        if stop is None:
            stop = 20230301145700000
        previou_close_zero = 0.0
        num = 0
        pf_list = []
        for i in range(start, stop, time):
            candle_list = []
            tick_volume = 0
            order_num = 0
            if i + time <= stop:
                for t in range(i, i + time + 1):
                    self.update(t)
                    tick_volume += self.tick_volume
                    order_num += self.order_num
                    candle_list.extend([self.open, self.high, self.low, self.close])
                    if num == 0:
                        previou_close_zero = self.previous_close
                        num += 1
            else:
                for t in range(i, stop):
                        self.update(t)
                        tick_volume += self.tick_volume
                        order_num += self.order_num
                        candle_list.extend([self.open, self.high, self.low, self.close])
                        if num == 0:
                            previou_close_zero = self.previous_close
                            num += 1
            if pf_list:
                self.previous_close = pf_list[-1]
            else:
                self.previous_close = previou_close_zero
            if candle_list:
                self.open = candle_list[1]
                self.close = candle_list[-1]
                self.high = max(candle_list)
                self.low = min(candle_list)
            self.tick_volume = tick_volume
            self.order_num = order_num
            pf_list.extend([self.tick_volume,self.order_num,self.previous_close,self.open, self.high, self.low, self.close])
        self.pf_list = pf_list

            



if __name__ == "__main__":
    current_dir = Path(__file__).parent
    data_api = Path(__file__).parent / "../datas/000001.SZ/tick/gtja/2023-07-03.csv"
    tick = DataSet(data_api, date_column="time", ticker="000001.SZ")

    test = Pastfeature(data_api=tick)
    # test.update(20230301095100700)
    # print(test.tick_volume, test.order_num, test.previous_close, test.open, test.close, test.high, test.low)
    # test.update(20230301094636370)
    # print(test.tick_volume,test.order_volume)
    test.get_period(3000, 20230703094636370, 20230703094656370)
    print(test.pf_list)


