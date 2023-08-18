from pathlib import Path
from typing import Union
# from .ReverseMomentum import Model_reverse
from .RM_new import Model_reverse
from .main import momentumStratgy, reverse_strategy
from .modelType import modelType
from ..OrderMaster.DataManager import DataStream, DataSet
from ..OrderMaster.OrderBook import OrderBook

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
            if stratgy.timeStamp >= 20230717092500000:
                print(stratgy.current_price,stratgy.price_list)
            # stratgy.model_update(model=model) 
            # stratgy.signal_update()
            # stratgy.stratgy_update()
        except StopIteration:
            break
    # 确保最后一行tick数据也被更新
    # if stratgy.new_timeStamp is False:
    #     stratgy.model_update(model=model)
        # stratgy.signal_update()
        # stratgy.stratgy_update()
    # print(stratgy.win_rate[datestr])


if __name__ == "__main__":
    model = Model_reverse()
    run(datestr='2023-07-17',
    symbol='601012.SH',
    stratgy_type=reverse_strategy,
    model=model)
    
