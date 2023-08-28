import logging
from pathlib import Path
from typing import Union
from pathlib import Path

from .main import momentumStratgy, reverse_strategy
from .modelType import modelType
from .ReverseMomentum import Model_reverse
from ..OrderMaster.DataManager import DataSet
from ..OrderMaster.OrderBook import OrderBook

parent_path = Path(__file__).parent
logging.basicConfig(
    filename=Path.joinpath(parent_path, "reverse.log"),
    level=logging.INFO,
    format="%(asctime)s-%(message)s",
)


def run(
    datestr: str,
    symbol: str,
    stratgy_type: Union[type[momentumStratgy], momentumStratgy],
    model: modelType,
):
    tick_path = (
        Path(__file__).parent.parent / f"./datas/{symbol}/tick/gtja/{datestr}.csv"
    )
    data_api = DataSet(tick_path, date_column="time", ticker=symbol)
    orderbook = OrderBook(data_api)
    stratgy = stratgy_type(orderbook)
    while True:
        try:
            tick_data = data_api.fresh(1)
            stratgy.update_orderbook(tick_data)
            if stratgy.new_timeStamp is False:
                continue
            logging.info(stratgy.model_update(model=model))
        except StopIteration:
            break
    # 确保最后一行tick数据也被更新
    if stratgy.new_timeStamp is False:
        print(stratgy.date)
        stratgy.model_update(model=model)


if __name__ == "__main__":
    model = Model_reverse()
    run(
        datestr="2023-07-17",
        symbol="601012.SH",
        stratgy_type=reverse_strategy,
        model=model,
    )
