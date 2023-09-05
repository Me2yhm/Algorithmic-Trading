import logging
import csv
from pathlib import Path
from typing import Union, Iterable, Any

from .utils import write_to_csv
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
    record_index: bool = False,
):
    tick_path = (
        Path(__file__).parent.parent / f"./datas/{symbol}/tick/gtja/{datestr}.csv"
    )
    file_path = Path(__file__).parent / "./strategy_result.csv"
    data_api = DataSet(tick_path, date_column="time", ticker=symbol)
    orderbook = OrderBook(data_api)
    stratgy = stratgy_type(orderbook, symbol)
    while True:
        try:
            tick_data = data_api.fresh(1)
            stratgy.update_orderbook(tick_data)
            if stratgy.new_timeStamp is False:
                continue
            stratgy.model_update(model=model)
            win_rate = stratgy.strategy_update()
            data = [stratgy.timeStamp, stratgy.current_price, win_rate]
            if all(data):
                write_to_csv(data, filepath=file_path)
                logging.info("write a new line")
        except StopIteration:
            break
    # 确保最后一行tick数据也被更新
    if stratgy.new_timeStamp is False:
        print(stratgy.date)
        stratgy.model_update(model=model)
        win_rate = stratgy.strategy_update()
        data = [stratgy.timeStamp, stratgy.current_price, win_rate]
        if all(data):
            write_to_csv(data, filepath=file_path)
            logging.info("write the last line")


if __name__ == "__main__":
    model = Model_reverse()
    run(
        datestr="2023-07-26",
        symbol="601155.SH",
        stratgy_type=reverse_strategy,
        model=model,
        record_index=True,
    )
