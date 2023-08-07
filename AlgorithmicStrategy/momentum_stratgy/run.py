from pathlib import Path
from typing import Union

from AlgorithmicStrategy.momentum_stratgy.modelType import modelType
from .main import momentumStratgy
from .modelType import modelType
from ..OrderMaster.DataManager import DataStream, DataSet
from ..OrderMaster.OrderBook import OrderBook


def run(
    datestr: str, symbol: str, stratgy: Union[type, momentumStratgy], model: modelType
):
    tick_path = (
        Path(__file__).parent.parent / f"./datas/{symbol}/tick/gtja/{datestr}.csv"
    )
    data_api = DataSet(tick_path, date_column="time", ticker=symbol)
    orderbook = OrderBook(data_api)
    stratgy(orderbook)
    while True:
        try:
            tick_data = data_api.fresh(1)
            stratgy.update_orderbook(tick_data)
            if stratgy.new_timeStamp is False:
                continue
            stratgy.model_update(model=model)
            stratgy.signal_update()
            stratgy.stratgy_update()
        except StopIteration:
            break
    if stratgy.new_timeStamp is False:
        stratgy.model_update(model=model)
        stratgy.signal_update()
        stratgy.stratgy_update()
    print(stratgy.win_rate[datestr])


if __name__ == "__main__":
    pass
