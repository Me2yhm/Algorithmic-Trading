import logging
import csv
from pathlib import Path
from typing import Union, Iterable, Any

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


def write_to_csv(line: Iterable[Any], filepath: str):
    with open(filepath, mode="+a", encoding="utf_8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(line)


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
    file_path = Path(__file__).parent / "./index_90.csv"
    data_api = DataSet(tick_path, date_column="time", ticker=symbol)
    orderbook = OrderBook(data_api)
    stratgy = stratgy_type(orderbook)
    while True:
        try:
            tick_data = data_api.fresh(1)
            stratgy.update_orderbook(tick_data)
            if stratgy.new_timeStamp is False:
                continue
            stratgy.model_update(model=model)
            if len(stratgy.model_indicator) > 1:
                data = list(stratgy.model_indicator[-1].values())
                if all(data):
                    write_to_csv(data, filepath=file_path)
                    logging.info("write a new line")
                pass
        except StopIteration:
            break
    # 确保最后一行tick数据也被更新
    if stratgy.new_timeStamp is False:
        print(stratgy.date)
        stratgy.model_update(model=model)
        data = list(stratgy.model_indicator[-1].values())
        if all(data):
            write_to_csv(data, filepath=file_path)
            logging.info("write the last line")


if __name__ == "__main__":
    model = Model_reverse()
    run(
        datestr="2023-07-17",
        symbol="601012.SH",
        stratgy_type=reverse_strategy,
        model=model,
        record_index=True,
    )
