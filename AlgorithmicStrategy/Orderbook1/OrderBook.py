import json
from collections import OrderedDict
from typing import TypedDict, Literal, Union
from pathlib import Path
import numpy as np
import numpy as np
import copy

from DataManager import DataStream, TimeType, OT, DataSet


class LifeTime(TypedDict, total=False):
    oid: int
    price: float
    volume: int
    rest: int
    birth: int
    death: int
    life: int
    AS: str

class Excecuted_trade(TypedDict):
    total_price: float
    price: float
    volume: int
    order_num: int
    passive_num: int
    passive_stale_total: int

class SnapShot(TypedDict):
    timestamp: TimeType
    ask: OrderedDict[float, float]
    bid: OrderedDict[float, float]
    ask_num: OrderedDict[float, int] #存量
    bid_num: OrderedDict[float, int] #存量
    bid_order_stale : OrderedDict[float, float] #增量
    ask_order_stale: OrderedDict[float, float] #增量
    ask_num_death: OrderedDict[float, int] #增量
    bid_num_death: OrderedDict[float, int] #增量
    total_trade: Excecuted_trade #增量


class OrderBook:
    __slots__ = (
        "snapshots",
        "last_snapshot",
        "data_api",
        "oid_map",
        "data_cache",
        "skip_order",
    )

    # noinspection PyTypeChecker
    def __init__(self, data_api: Union[DataStream, DataSet]):
        self.snapshots: OrderedDict[TimeType, SnapShot] = OrderedDict()
        self.last_snapshot: SnapShot = None
        self.data_api: Union[DataStream, DataSet] = data_api
        self.oid_map: dict[int, LifeTime] = dict()
        self.data_cache: list[OT] = []
        self.skip_order: list[int] = []

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

    def update(self, until: int = None):
        while True:
            try:
                res = self.next_batch(until=until)
                if res:
                    self.single_update(res)
                else:
                    break
            except StopIteration:
                break

    def single_update(self, datas: list[OT] = None):
        if datas is None:
            datas = self.next_batch()
        datas: list[OT] = sorted(datas, key=lambda x: x["oid"], reverse=True)
        for data in datas:
            if self.last_snapshot is None:
                self.last_snapshot = SnapShot(
                    timestamp=data[self.data_api.date_column],
                    ask=OrderedDict(),
                    bid=OrderedDict(),
                    ask_num=OrderedDict(), #初始化卖单数
                    bid_num=OrderedDict(), #初始化买单数
                    ask_order_stale=OrderedDict(),
                    ask_num_death=OrderedDict(),
                    bid_order_stale=OrderedDict(),
                    bid_num_death=OrderedDict(),     
                    total_trade=Excecuted_trade(order_num=0,
                                                volume=0,
                                                price=0.0,
                                                total_price=0.0,
                                                passive_num=0,
                                                passive_stale_total=0
                                                ), #初始化总成交单数                                                                                  
                )

            if data["oid"] != 0 or data["ptype"] != 0: #不是成交单
                if data["flag"] in [1,2]: #如果是深市的委托单或者沪市的非撤买撤卖委托单
                    self.oid_map[data["oid"]] = LifeTime(
                        oid=data["oid"],
                        price=data["price"],
                        volume=data["volume"],
                        birth=data[self.data_api.date_column],
                        rest=data["volume"],
                        AS="bid" if data["flag"] == 1 else "ask",
                    )
                    self._update_from_order(data)
                    self._order_num_change(data, None, 1)
                if data["flag"] == 3: #如果是沪市的撤买单
                    self._log_oid(data, key="oid")                   
                if data["flag"] == 4: #如果是沪市的撤卖单
                    self._log_oid(data, key="oid")    

            else:
                if self.data_api.ticker.endswith("SZ"):
                    if data["flag"] == 3:
                        # 异常订单：000001.SZ, 2746350，从未有人发起
                        self._log_oid(data, key="oidb")
                    elif data["flag"] == 4:
                        self._log_oid(data, key="oids")
                    else:
                        self._cal_log_oid(data, key="oidb")
                        self._cal_log_oid(data, key="oids")
                        self._trade_update(data) #对成交单数量进行更新

                    tmp_oid_idx = "oidb" if data["flag"] in [1, 3] else "oids"
                    if data["price"] == 0.0:
                        if data[tmp_oid_idx] in self.oid_map:
                            data["price"] = self.oid_map[data[tmp_oid_idx]]["price"]
                        else:
                            continue
                    assert data["price"] != 0.0, data
                    self._update_from_tick(data)

                elif self.data_api.ticker.endswith("SH"):
                    # data.flag may be 0,1,2
                    if data["flag"] == 2:
                        self._cal_log_oid(data, key="oidb")
                        self._trade_update(data) #对成交单数量进行更新
                    elif data["flag"] == 1:
                        self._cal_log_oid(data, key="oids")
                        self._trade_update(data) #对成交单数量进行更新
                    else:
                        self._cal_log_oid(data, key="oidb")
                        self._cal_log_oid(data, key="oids")
                        self._trade_update(data) #对成交单数量进行更新
                    tmp_oid_idx = "oidb" if data["flag"] in [1, 3] else "oids"
                    if data["price"] == 0.0:
                        if data[tmp_oid_idx] in self.oid_map:
                            data["price"] = self.oid_map[data[tmp_oid_idx]]["price"]
                        else:
                            continue
                    self._update_from_tick(data)

    def _order_num_change(self, data:OT, key:str, direction:Literal[1,-1]):
        if direction == 1:
            if data["flag"] == 1 :
                if data["price"] not in self.last_snapshot["bid_num"]:
                    self.last_snapshot["bid_num"][data["price"]] = 1 
                else:
                    self.last_snapshot["bid_num"][data["price"]] += 1 
            if data["flag"] == 2 :
                if data["price"] not in self.last_snapshot["ask_num"]:
                    self.last_snapshot["ask_num"][data["price"]] = 1                     
                else:
                    self.last_snapshot["ask_num"][data["price"]] += 1   
        
        if direction == -1:
            price = self.oid_map[data[key]]["price"]   
            if key == "oidb":
                self.last_snapshot["bid_num"][price] -= 1                           
            if key == "oids":
                self.last_snapshot["ask_num"][price] -= 1     
        return  

    #计算所有成交单的数据                          
    def _trade_update(self, data:OT): 
        #先计算主动成交单
        #如果主动成交单的rest = volume则为第一次计算，仅计算一次成交单数量的增加，volume仍然继续计算
        oidp = data['oidb'] if data['flag'] == 1 else data["oids"]
        if self.oid_map[oidp]['rest'] == 0:
            self.last_snapshot["total_trade"]["order_num"] += 1
        self.last_snapshot["total_trade"]["volume"] += data["volume"]
        total_price = data["price"] * data["volume"]
        self.last_snapshot["total_trade"]["total_price"] += total_price
        #再计算被动成交单，如果全部成交了就在passive部分增加
        oid = data['oids'] if data['flag'] == 1 else data["oidb"]
        if self.oid_map[oid]['rest'] == 0:    
            self.last_snapshot["total_trade"]["order_num"] += 1
            self.last_snapshot["total_trade"]["passive_num"] += 1
            self.last_snapshot["total_trade"]["passive_stale_total"] += self.oid_map[oid]["life"]
            self.last_snapshot["total_trade"]["volume"] += self.oid_map[oid]["rest"]
            total_price = data["price"] * self.oid_map[oid]["rest"]
            self.last_snapshot["total_trade"]["total_price"] += total_price

        self.last_snapshot["total_trade"]["price"] = (self.last_snapshot["total_trade"]["total_price"]
                                                    /self.last_snapshot["total_trade"]["volume"])
                       

    def _order_stale_update(self, data:OT, key:str):
        oid = data[key]
        life = self.oid_map[oid]["life"]
        price = self.oid_map[oid]["price"]
        AS = self.oid_map[oid]["AS"]
        if AS == 'bid' :
            if price not in self.last_snapshot["bid_order_stale"]:
                self.last_snapshot["bid_order_stale"][price] = life
                self.last_snapshot["bid_num_death"][price] = 1 
            else:
                self.last_snapshot["bid_order_stale"][price] += life 
                self.last_snapshot["bid_num_death"][price] += 1 
        if AS == 'ask' :
            if price not in self.last_snapshot["ask_order_stale"]:
                self.last_snapshot["ask_order_stale"][price] = life
                self.last_snapshot["ask_num_death"][price] = 1                     
            else:
                self.last_snapshot["ask_order_stale"][price] += life
                self.last_snapshot["ask_num_death"][price] += 1 


    def _cal_log_oid(self, data: OT, key: str):
        try:
            tmp_val = self.oid_map[data[key]]["rest"] - data["volume"]
            if tmp_val != 0:
                self.oid_map[data[key]]["rest"] = tmp_val
            elif tmp_val == 0:
                self.oid_map[data[key]]["rest"] = 0
                self.oid_map[data[key]]["death"] = data[self.data_api.date_column]
                self.oid_map[data[key]]["life"] = (
                        data[self.data_api.date_column] - self.oid_map[data[key]]["birth"]
                )
                self._order_num_change(data, key, -1)
                self._order_stale_update(data, key)
        except KeyError:
            pass

    def _log_oid(self, data: OT, key: str):
        try:
            self.oid_map[data[key]]["rest"] = 0
            self.oid_map[data[key]]["death"] = data[self.data_api.date_column]
            self.oid_map[data[key]]["life"] = (
                    data[self.data_api.date_column] - self.oid_map[data[key]]["birth"]
            )
            self._order_num_change(data, key, -1)
            self._order_stale_update(data, key)
        except KeyError:
            return

    @staticmethod
    def _order_change(
            snap: SnapShot, AS: Literal["ask", "bid"], direction: Literal[1, -1], data: OT
    ):
        snap[AS][data["price"]] = (
                snap[AS].get(data["price"], 0) + direction * data["volume"]
        )
        if snap[AS][data["price"]] == 0:
            del snap[AS][data["price"]]

        snap["ask"] = OrderedDict(
            sorted(snap["ask"].items(), key=lambda x: x[0], reverse=False)
        )

        snap["bid"] = OrderedDict(
            sorted(snap["bid"].items(), key=lambda x: x[0], reverse=True)
        )

    @staticmethod
    def _tick_change(snap: SnapShot, data: OT, direction: Union[str, list[str]] = None):
        if direction is None:
            direction = ["ask", "bid"]
        assert data["price"] != 0.0, data
        TradePrice = data["price"]
        TradeQty = data["volume"]
        A_p_v = list(snap["ask"].items())
        B_p_v = list(snap["bid"].items())

        if "bid" in direction:
            rest = TradeQty
            for p, v in B_p_v:
                if p >= TradePrice:
                    if v >= rest:
                        v -= rest
                        if v == 0:
                            del snap["bid"][p]
                        else:
                            snap["bid"][p] = v
                    else:
                        rest -= v
                        del snap["bid"][p]
                else:
                    break

        if "ask" in direction:
            rest = TradeQty
            for p, v in A_p_v:
                if p <= TradePrice:
                    if v >= rest:
                        v -= rest
                        if v == 0:
                            del snap["ask"][p]
                        else:
                            snap["ask"][p] = v
                    else:
                        rest -= v
                        del snap["ask"][p]
                else:
                    break

        snap["ask"] = OrderedDict(
            sorted(snap["ask"].items(), key=lambda x: x[0], reverse=False)
        )

        snap["bid"] = OrderedDict(
            sorted(snap["bid"].items(), key=lambda x: x[0], reverse=True)
        )

    def _update_from_order(self, data: OT):
        snap: SnapShot = self.last_snapshot.copy()
        snap["timestamp"]: int = data[self.data_api.date_column]
        AS: Literal["bid", "ask"] = "bid" if data["flag"] in [1, 3] else "ask"  # type: ignore
        direction: Literal[1, -1] = 1 if data["flag"] in [1, 2] else -1  # type: ignore
        # assert data["price"] != 0.0, data
        if data["price"] == 0.0:
            self.skip_order.append(data["oid"])
            return
        if self.data_api.ticker.endswith("SZ") or self.data_api.ticker.endswith("SH"):
            self._order_change(snap, AS, direction, data)
        else:
            raise NotImplementedError

        self.last_snapshot = snap
        self.snapshots[data[self.data_api.date_column]] = copy.deepcopy(self.last_snapshot)  # type: ignore

    def _update_from_tick(self, data: OT):
        snap: SnapShot = self.last_snapshot.copy()
        snap["timestamp"]: int = data[self.data_api.date_column]
        # assert data["price"] != 0.0, data
        if self.data_api.ticker.endswith("SZ"):
            if data["flag"] in [1, 2]:
                direction = []
                if data["oids"] not in self.skip_order:
                    direction.append("ask")
                if data["oidb"] not in self.skip_order:
                    direction.append("bid")
                self._tick_change(snap, data, direction=direction)
            elif data["flag"] == 3:
                self._order_change(snap, "bid", -1, data)
            elif data["flag"] == 4:
                self._order_change(snap, "ask", -1, data)
        elif self.data_api.ticker.endswith("SH"):
            if data["flag"] == 2:
                self._tick_change(snap, data, direction=["bid"])
            elif data["flag"] == 1:
                self._tick_change(snap, data, direction=["ask"])
            else:
                self._tick_change(snap, data)
        else:
            raise NotImplementedError
        self.snapshots[data[self.data_api.date_column]] = copy.deepcopy(self.last_snapshot)

    def search_closet_time(self, query_stamp: int):
        logged_timestamp: np.ndarray = np.array(list(self.snapshots.keys()))
        search_timestamp = logged_timestamp[logged_timestamp <= query_stamp]
        return search_timestamp[-1]

    #计算两个时间戳之间的成交订单平均新陈代谢
    def _get_avg_stale(self, timestamp_1:TimeType, timestamp_2:TimeType): 
        snap2 = self.search_snapshot(timestamp_2)
        snap1 = self.search_snapshot(timestamp_1)
        ask_total_stale2 = copy.deepcopy(snap2["ask_order_stale"]) 
        bid_total_stale2 = copy.deepcopy(snap2["bid_order_stale"]) 
        # print(snap2["timestamp"])
        # print(snap2["ask_num_death"])
        # print(snap1["timestamp"])
        # print(snap1["ask_num_death"])
        for key in list(ask_total_stale2.keys()):
            if key in snap1["ask_order_stale"].keys():        
                ask_total_stale2[key] -= snap1["ask_order_stale"][key]
                if ask_total_stale2[key] == 0: 
                    del ask_total_stale2[key]
                    continue
                ask_death_change = (snap2["ask_num_death"][key] 
                                    - snap1["ask_num_death"][key])
                ask_total_stale2[key] = ask_total_stale2[key] / ask_death_change
            else:
                ask_total_stale2[key] = ask_total_stale2[key] / snap2["ask_num_death"][key]          

        for key in list(bid_total_stale2.keys()):
            if key in snap1["bid_order_stale"].keys():            
                bid_total_stale2[key] -= snap1["bid_order_stale"][key]
                if bid_total_stale2[key] == 0: 
                    del bid_total_stale2[key]
                    continue
                bid_death_change = (snap2["bid_num_death"][key] 
                                    - snap1["bid_num_death"][key])
                bid_total_stale2[key] = bid_total_stale2[key] / bid_death_change    
            else:
                bid_total_stale2[key] = bid_total_stale2[key] / snap2["bid_num_death"][key]    

        return ask_total_stale2, bid_total_stale2

    def _get_avg_trade(self, timestamp_1, timestamp_2):
        snap1 = self.search_snapshot(timestamp_1)       
        snap2 = self.search_snapshot(timestamp_2)

        total_trade_2 = copy.deepcopy(snap2["total_trade"])
        total_trade_2["order_num"] -= snap1["total_trade"]["order_num"]
        total_trade_2["passive_num"] -=  snap1["total_trade"]["passive_num"]
        total_trade_2["total_price"] -=  snap1["total_trade"]["total_price"]
        total_trade_2["volume"] -= snap1["total_trade"]["volume"]
        total_trade_2["passive_stale_total"] -=  snap1["total_trade"]["passive_stale_total"]
        total_trade_2["price"] = total_trade_2["total_price"] / total_trade_2["volume"]

        return total_trade_2
 

    @staticmethod
    def print_json(dict_like: dict):
        print(json.dumps(dict_like, indent=2))

    def search_snapshot(self, query_stamp: int):
        closest_time = self.search_closet_time(query_stamp)
        return self.snapshots[closest_time]


#测试函数
def test_iter(range1):
    for i in range(0, range1):
        datas = tick.fresh()
        ob.single_update(datas)

def show_oid_map(oid):
    print(ob.oid_map[oid])    

def show_order_num(p1, p2):
    print("-----")
    print('ask_num')
    print(ob.last_snapshot["ask_num"][p1])
    print('bid_num')
    print(ob.last_snapshot["bid_num"][p2])    
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
    data_api = current_dir.parent / "datas/000001.SZ/tick/gtja/2023-05-08.csv"
    tick = DataSet(data_api, date_column="time", ticker="000001.SZ")
    ob = OrderBook(data_api=tick)

    if 1 :
        test_iter(6861)
        show_total_order_num()
        # show_oid_map(136)
        # show_oid_map(29897)
        # show_oid_map(396)
        # show_oid_map(224385)    
        print(ob.last_snapshot["total_trade"])
        # print(ob.last_snapshot["ask_order_stale"])
        # print(ob.last_snapshot["bid_order_stale"])
        print(ob.last_snapshot["ask_num_death"])
        print(ob.last_snapshot["bid_num_death"])
        test_iter(3)
        show_total_order_num()
        # show_oid_map(136)
        # show_oid_map(29897)
        # show_oid_map(396)
        # show_oid_map(224385)
        print(ob.last_snapshot["total_trade"])
        # print(ob.last_snapshot["ask_order_stale"])
        # print(ob.last_snapshot["bid_order_stale"])
        print(ob.last_snapshot["ask_num_death"])
        print(ob.last_snapshot["bid_num_death"])
        # print(ob.oid_map[398])
        # print(ob.oid_map[224385])
        # print(ob.snapshots[20230508092459840])
        # print(ob.last_snapshot["bid_order_stale"])
        print(ob._get_avg_stale(20230508092459840, 20230508092500000))
        print(ob._get_avg_trade(20230508092459840, 20230508092500000))



    if 0 :
        ob.update(20230508093639200)
        show_total_order_num()

    if 0:
        ob.update(20230508092400020)
        snap = ob.search_snapshot(20230508092400000)
        print(snap["timestamp"])
        # print(snap["ask_num"])
        print(snap["total_trade"])
        # print(ob.search_snapshot(20230508092400000)["ask"])


#除了ask和bid两个之外更新lastsnapshot后所有之前的snapshot属性都会更新
    if 0:
        ob.update(20230508093500020)
        snap = ob.search_snapshot(20230508092400000)
        print(snap["timestamp"])
        # print(snap["ask_num"])
        print(snap["total_trade"])
        print(ob.search_snapshot(20230508093100020)["total_trade"])
        # print(ob.search_snapshot(20230508092600020)["ask_num"])
        # print(ob.search_snapshot(20230508092400000)["ask"])
        # print(ob._get_avg_stale(20230508092400000, 20230508092500020))



