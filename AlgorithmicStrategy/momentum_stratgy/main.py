from abc import ABC, abstractmethod
from pathlib import Path
from AlgorithmicStrategy.base import AlgorithmicStrategy
from ..TWAP_VWAP.OrderMaster.DataManager import OT, DataSet
from ..TWAP_VWAP.OrderMaster.OrderBook import OrderBook
from typing import List, Dict
from pandas import DataFrame


class modelType(ABC):
    """
    模型的抽象基类，用于实现指标更新或者模型增量训练

    """

    @abstractmethod
    def model_update(tick) -> dict[str, float]:
        """
        模型接受一个tick参数(目前还不确定是什么数据类型)，根据tick数据返回模型计算结果
        """


class momentumStratgy(AlgorithmicStrategy):
    """
    动量算法类, 有如下属性
    orderbook: orderbook类, 可以撮合盘口, 记录盘口状态
    tick: 储存逐笔数据
    timeStamp: 记录当前时间戳
    deal: 成交记录
    possession: 调仓记录
    model_indicator: 指标计算结果
    """

    orderbook: OrderBook
    tick: List[Dict]
    timeStamp: int
    deal: List[Dict]
    possession: List[Dict]
    model_indicator: Dict[str, float]

    def update_oderbook(self, symbol, datestr) -> None:
        """
        流式读取数据，更新盘口
        """
        data_api = (
            Path(__file__).parent / f"../../datas/{symbol}/tick/gtja/{datestr}.csv"
        )
        tick = DataSet(data_api, date_column="time", ticker=symbol)

    def model_update(self, model: modelType) -> None:
        """
        盘口更新过后，根据更新过的数据增量地更新指标或者训练模型
        """
        pass

    def signal_update(self) -> dict:
        """
        调用model_update函数，根据函数结果返回信号
        """
        pass

    def exam_update(self) -> dict:
        """
        根据返回的信号计算胜率、赔率、换手率等——可以流式？
        """
        pass
