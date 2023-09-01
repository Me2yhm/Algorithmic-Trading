from datetime import datetime
from datetime import datetime
from pathlib import Path
from typing import Literal, cast
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import torch as t


from AlgorithmicStrategy import (
    DataSet,
    OrderBook,
    AlgorithmicStrategy,
    signal,
    possession,
    LimitedQueue,
    Writer,
    Standarder,
    TradeTime,
)
from MODELS import logger, OCET


class VWAP(AlgorithmicStrategy):
    def __init__(
        self,
        orderbook: OrderBook,
        tick: DataSet,
        trade_volume: int,
        queue: LimitedQueue,
        writer: Writer,
        normer: Standarder,
        direction: Literal["BUY", "SELL"],
        trade_time,
        model: OCET,
        logger,
        **kwargs,
    ):
        super().__init__(orderbook=orderbook, **kwargs)
        self.normer = normer
        self.writer = writer
        self.queue = queue
        self.direction = direction
        self.trade_volume = trade_volume
        self.tick = tick
        self.orderbook = orderbook
        self.trade: bool = False
        self.trade_time: dict[int, dict] = dict(trade_time)
        self.model = model
        self.logger = logger

        self.key_map = {"BUY": "ask", "SELL": "bid"}

    def model_update(self) -> None:
        self.writer.file.close()
        self.normer.fit_transform_for_files(self.writer.filename, output=simu_folder / "NORM")
        self.model.train()

    def signal_update(self) -> None:
        tmp: dict = self.trade_time.get(self.timeStamp, None)
        if tmp is not None:
            if tmp["trade"]:
                if self.queue.size == 100:
                    df = self.queue.to_df()
                    df_normalized = self.normer.fit_transform_for_dataframe(df)
                    temp_data = t.tensor(
                        df_normalized.values[np.newaxis, np.newaxis, :, 1:-1],
                        dtype=t.float32,
                    )
                    vol_percent_pred = self.model(temp_data)
                    price = list(
                        self.orderbook.last_snapshot[
                            self.key_map[self.direction]
                        ].keys()
                    )[0]
                    self.logger.info(
                        f"{self.timeStamp} Pred volume%: {float(vol_percent_pred)} at Price {price}"
                    )
                    real_vol = int(float(vol_percent_pred) * self.trade_volume)
                    self.logger.info(f"(Real:{real_vol})")
                    if float(vol_percent_pred) != 0:
                        sig = signal(
                            timestamp=self.timeStamp,
                            symbol=self.tick.ticker,
                            direction=self.direction,
                            price=price,
                            volume=real_vol,
                        )
                        self.signals[self.date].append(sig)

                        self.logger.info(sig)

                        if self.date not in self.possessions:
                            self.possessions[self.date] = possession(
                                code=self.tick.ticker,
                                volume=real_vol,
                                averagePrice=price,
                                cost=0,
                            )
                        else:
                            poss = self.possessions[self.date]
                            poss["averagePrice"] = (
                                poss["volume"] * poss["averagePrice"] + real_vol * price
                            ) / (poss["volume"] + real_vol)
                            poss["volume"] = poss["volume"] + real_vol
                            self.possessions[self.date] = poss

                        self.logger.info(self.possessions[self.date])

            if tmp["update"]:
                newest_data = self.writer.collect_data_by_timestamp(
                    self.orderbook,
                    timestamp=self.timeStamp,
                    timestamp_prev=self.writer.get_prev_timestamp(self.timeStamp),
                )
                self.writer.csvwriter.writerow(newest_data)
                self.queue.push(
                    pd.DataFrame([newest_data], columns=self.writer.columns)
                )

    def strategy_update(self) -> None:
        pass


def get_newest_model(path: Path, suffix: str = "ocet"):
    files = list(path.glob(f"*.{suffix}"))
    newest_model = sorted(files, key=lambda x: int(x.stem), reverse=True)[0]
    return newest_model


if __name__ == "__main__":
    model_save_path: Path = Path().cwd() / "MODEL_SAVE_1"
    newest_model_path = get_newest_model(model_save_path)
    logger.info(f"Using: {newest_model_path.name}")

    tick_folder = Path.cwd() / "../datas/000157.SZ/tick/gtja/"
    tick_files = list(tick_folder.glob("*.csv"))
    tick_files.sort(key=lambda x: datetime.strptime(x.stem, "%Y-%m-%d"))

    simu_folder = Path().cwd() / "REAL"
    logger.info(f"Simulating folder: {simu_folder}")

    ticker = "000157.SZ"
    logger.info(f"Simulating: {ticker}")

    ocet = OCET(
        num_classes=1,
        dim=100,
        depth=2,
        heads=4,
        dim_head=25,
        mlp_dim=200,
    )

    para_dict = t.load(newest_model_path, map_location=t.device("cpu"))
    ocet.load_state_dict(para_dict["model_state_dict"])

    lq = LimitedQueue(max_size=100)
    standard = Standarder(file_folder=simu_folder / "RAW", train=False, limits=5)
    standard.fresh_files()

    # 交易时间起始
    total_begin = 9_15_00_000
    trade_begin = 9_30_03_000
    end = 14_57_00_000

    direction = "BUY"

    trade_volume = 2000

    with t.no_grad():
        for tick_file in tick_files[-4:]:
            logger.info(f"Date: {tick_file.stem}")
            tick = DataSet(tick_file, ticker=ticker)
            ob = OrderBook(data_api=tick)
            raw_file = simu_folder / "RAW" / tick_file.name
            writer = Writer(filename=raw_file, rollback=3000)

            past_files = standard.get_past_files(tick_file)
            if len(past_files) == 0:
                continue
            standard.read_files(past_files)
            standard.generate_hist_feature(past_files)
            standard.fit(standard.hist_df)
            logger.info(f"Using past feature: {[_.stem for _ in past_files]}")

            lq.clear()
            tt = TradeTime(begin=trade_begin, end=end, tick=tick)
            trade_time = tt.generate_signals(
                trade_interval=6000,
                trade_limits=(-2500, 2500),
                update_interval=3000,
                update_limits=(0, 0),
            )
            trader = VWAP(
                orderbook=ob,
                tick=tick,
                writer=writer,
                normer=standard,
                direction=direction,
                queue=lq,
                trade_volume=trade_volume,
                trade_time=trade_time,
                model=ocet,
                logger=logger,
            )

            for timestamp in range(
                tick.file_date_num + total_begin, tick.file_date_num + end
            ):
                datas = trader.tick.next_batch(until=timestamp)
                if datas:
                    trader.update_orderbook(datas)
                    # print(trader.orderbook.last_snapshot)
                trader.timeStamp = timestamp
                try:
                    trader.signal_update()
                    if trader.trade:
                        trader.strategy_update()
                except IndexError:
                    continue
            trader.model_update()
            break
