

# Guide for OrderMaster

## DataManager

基类`DataBase`， 文件夹数据管理类`DataStream`，文件数据管理类`DataSet`。

三者均为`Iterator`类，基类无法直接实例化，需要继承并实现其中的`__next__`方法。

### DataStream

实例化时需要传入参数：数据文件夹路径`data_folder`，股票的`ticker`

```python
def __init__(
        self, data_folder: PathType, ticker: str, delimiter: str = ",", **kwargs
    )
```

为实现相应功能，该类包含了诸多内部函数即属性，请查看源码。其中，不推荐使用以`_`开头的内部函数。

主要使用的函数仅为`fresh`, 用法如下：
```python
from DataManager import DataStream
data_api = path_to_your_folder
tick = DataStream(data_api, date_column="time", ticker="000001.SZ")
print(tick.fresh())
```

该方法仅会返回数据文件中的一行，等价于调用`next(tick)`。

可以传入参数`num`，默认为1，返回一条数据，类型为字典。传入大于1的整数，会返回`num`条数据，类型为`list[dict]`


### DataSet

与`DataStream`不同的是，该类需要传入的是文件路径，其他函数与`DataStream`基本一致。

## OrderBook

处理tick数据的类，内部实现了深市和沪市的两种处理机制。

### 数据异常

可能存在以下几类数据异常
1. 先出现撤单，后出现挂单
2. 先出现买卖，后出现挂单
3. 有挂单信息，但没有相应的价格（与2基本同时出现，目前没有完善的处理该事件）
4. 有撤单信息，但该单没有相应的挂单

### search_closet_time

返回已存储的`SnapShots`中与给定时间戳最接近的时间戳（向前查找）。

### search_snapshot

返回已存储的`SnapShots`中与给定时间戳最接近的snapshot（向前查找）。

### single_update

更新到下一个时间戳。

### update

更新到指定的时间戳。

## 使用示例

```python
current_dir = Path(__file__).parent
data_api = Path(__file__).parent / "../datas/000001.SZ/tick/gtja/2023-03-01.csv"
tick = DataSet(data_api, date_column="time", ticker="000001.SZ")
ob = OrderBook(data_api=tick)

# example 1
# datas = tick.fresh()
# ob.single_update(datas)
# print(ob.last_snapshot)
# datas = tick.fresh()
# ob.single_update(datas)
# print(ob.last_snapshot)
# datas = tick.fresh()
# ob.single_update(datas)
# print(ob.last_snapshot)
# datas = tick.fresh()
# ob.single_update(datas)
# print(ob.last_snapshot)

# example 2
# ob.single_update()
# print(ob.last_snapshot)
# ob.single_update()
# print(ob.last_snapshot)
# ob.single_update()
# print(ob.last_snapshot)

# example 3
timestamp = 20230301093103000
ob.update(until=timestamp)
near = ob.search_snapshot(timestamp)
print(near["timestamp"])
print(near["bid"])
print(near["ask"])
```