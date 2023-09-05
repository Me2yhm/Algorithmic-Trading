from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent / "../OrderMaster"))
from DataManager import DataSet, DataStream, OrderTick
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
        self.data_cache: list[OrderTick] = []
        self.data: list[OrderTick] = []

    def next_batch(self, until: int = None) -> list[OrderTick]:
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
        order_num = 0.0
        for data in datas:
            if data["oid"] != 0 or data["ptype"] != 0:
                oid_list.append(data["oid"])
        oid_list_non = list(set(oid_list))
        order_num = len(oid_list_non)
        return order_num

    def get_previous_close(self, datas):
        previous_close = 0.0
        for data in datas:
            if (data["oid"] == 0 or data["ptype"] == 0) and data["price"]:
                previous_close = data["price"]
        return previous_close

    def get_candlestick(self, datas):
        candle = []
        for data in datas:
            if (data["oid"] == 0 or data["ptype"] == 0) and data["price"]:
                candle.append(data["price"])
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

    def update(self, until):
        self.get_all_previous(until=until - 1)
        self.get_all(until=until)


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    data_api = current_dir.parent / "datas/000001.SZ/tick/gtja/2023-05-08.csv"
    tick = DataSet(data_api, date_column="time", ticker="000001.SZ")

    test = Pastfeature(data_api=tick)
    test.update(20230508093735610)
    print(test.tick_volume, test.order_num, test.previous_close, test.open, test.close, test.high, test.low)
    # test.update(20230301091500070)
    # print(test.tick_volume,test.order_volume)
