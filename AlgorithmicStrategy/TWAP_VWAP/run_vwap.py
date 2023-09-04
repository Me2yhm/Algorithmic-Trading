from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch as t
from torch import optim

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
    LLobWriter,
)
from MODELS import (
    logger,
    log_train,
    OCET,
    setup_seed,
    MultiTaskLoss,
    JoyeLOB,
    LittleOB,
    save_model,
)


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
        args,
        littleob: LLobWriter,
        **kwargs,
    ):
        super().__init__(orderbook=orderbook, **kwargs)
        self.end_trade = False
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
        self.args = args
        self.littleob = littleob

        self.key_map = {"BUY": "ask", "SELL": "bid"}

    @logger.catch()
    def model_update(self) -> None:
        self.writer.file.close()
        self.logger.info(f"关闭{self.tick.file_date}Writer")
        self.normer.add_file(self.writer.filename)
        self.logger.info(f"将{self.tick.file_date}Raw文件加入归一化模块")
        self.normer.fit_transform_for_files(
            self.writer.filename, output=simu_folder / "NORM"
        )
        self.logger.info(f"生成{self.tick.file_date}NORM文件")
        train_files = train_folder.glob("*.csv")
        self.generate_LLOB_by_ORDERBOOK()
        self.logger.info(f"生成{self.tick.file_date}LLOB文件")
        self.model.train()
        self.logger.info(f"模型进入训练模式")
        global newest_model_path
        init_epoch = int(newest_model_path.stem)
        for epc in range(init_epoch, init_epoch + self.args.epoch):
            loss_log = []
            for file in train_files:
                if file not in joye_data:
                    joye_data.push(file)
                tick = DataSet(file, ticker="000157.SZ")
                llob_file = little_lob_folder / file.name
                if llob_file not in llob:
                    llob.push(llob_file)
                tt = TradeTime(begin=9_30_00_000, end=14_57_00_000, tick=tick)
                pred_trade_volume_fracs = []
                true_trade_volume_fracs = []
                hist_trade_volume_fracs = []
                trade_price = []
                for ts, action in tt.generate_signals():
                    if action["trade"]:
                        time_search, X, volume_hist, volume_today = joye_data.batch(
                            file, timestamp=ts
                        )
                        if time_search is not None:
                            X = t.tensor(X, dtype=t.float32)
                            pred_frac = ocet(X)

                            _, price = llob.batch(llob_file, ts)

                            trade_price.append(price)

                            hist_trade_volume_fracs.append(volume_hist)
                            pred_trade_volume_fracs.append(pred_frac)
                            true_trade_volume_fracs.append(volume_today)

                market_vwap = llob.get_VWAP(llob_file)
                pred_trade_volume_fracs = t.squeeze(t.stack(pred_trade_volume_fracs))
                trade_price = t.Tensor(trade_price)

                if t.sum(pred_trade_volume_fracs) < 1:
                    rest = 1 - t.sum(pred_trade_volume_fracs)
                    _, final_price = llob.batch(
                        llob_file, tick.file_date_num + 14_57_00_000
                    )
                    additional_vwap = rest * final_price
                    pred_vwap = (
                        t.sum(pred_trade_volume_fracs * trade_price) + additional_vwap
                    )

                if t.sum(pred_trade_volume_fracs) > 1:
                    pred_vwap = t.sum(
                        pred_trade_volume_fracs
                        * trade_price
                        / t.sum(pred_trade_volume_fracs).item()
                    )

                if t.sum(pred_trade_volume_fracs) == 1:
                    pred_vwap = t.sum(pred_trade_volume_fracs * trade_price)

                hist_trade_volume_fracs = t.Tensor(hist_trade_volume_fracs)
                hist_trade_volume_fracs = hist_trade_volume_fracs / t.sum(
                    hist_trade_volume_fracs
                )

                true_trade_volume_fracs = t.Tensor(true_trade_volume_fracs)
                true_trade_volume_fracs = true_trade_volume_fracs / t.sum(
                    true_trade_volume_fracs
                )

                loss = loss_func.calculate_loss(
                    pred_trade_volume_fracs,
                    true_trade_volume_fracs,
                    hist_trade_volume_fracs,
                    market_vwap,
                    pred_vwap,
                )
                optimizer.zero_grad()
                loss.backward()
                loss_log.append(loss.item())
                log_train(epoch=epc, epochs=args.epoch, file=file.stem, loss=loss.item())

            save_model(
                model=ocet,
                optimizer=optimizer,
                epoch=epc,
                loss=float(np.mean(loss_log)),
                path=model_save_path / f"{epc}.ocet",
            )

    @t.no_grad()
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
                    pred_trade_abs_volume = int(
                        float(vol_percent_pred) * self.trade_volume
                    )
                    if pred_trade_abs_volume != 0:
                        sig = signal(
                            timestamp=self.timeStamp,
                            symbol=self.tick.ticker,
                            direction=self.direction,
                            price=price,
                            volume=pred_trade_abs_volume,
                        )
                        self.signals[self.date].append(sig)
                        self.logger.info(sig)

                        if self.date not in self.possessions:
                            self.possessions[self.date] = possession(
                                code=self.tick.ticker,
                                volume=pred_trade_abs_volume,
                                averagePrice=price,
                                cost=0,
                            )
                        else:
                            poss = self.possessions[self.date]

                            temp_volume = poss["volume"] + pred_trade_abs_volume
                            if temp_volume >= self.trade_volume:
                                pred_trade_abs_volume = (
                                    self.trade_volume - poss["volume"]
                                )

                            poss["averagePrice"] = (
                                poss["volume"] * poss["averagePrice"]
                                + pred_trade_abs_volume * price
                            ) / (poss["volume"] + pred_trade_abs_volume)
                            poss["volume"] = poss["volume"] + pred_trade_abs_volume

                            self.possessions[self.date] = poss

                        # self.logger.info(self.possessions[self.date])
                        self.logger.info(
                            f"{self.timeStamp} Pred trade volume: {pred_trade_abs_volume} at Price {price}, "
                            f"rest volume {self.trade_volume - self.possessions[self.date]['volume']}"
                        )

                        if self.possessions[self.date]["volume"] >= self.trade_volume:
                            self.end_trade = True

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

    def generate_LLOB_by_ORDERBOOK(self):
        self.littleob.write_llob()


def get_newest_model(path: Path, suffix: str = "ocet"):
    files = list(path.glob(f"*.{suffix}"))
    newest_model = sorted(files, key=lambda x: int(x.stem), reverse=True)[0]
    return newest_model


if __name__ == "__main__":
    parser = ArgumentParser(description="Arguments for the strategy", add_help=True)
    parser.add_argument("-s", "--seed", type=int, default=2333, help="set random seed")
    parser.add_argument("-e", "--epoch", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="./DATA/ML")
    parser.add_argument("--model-save", type=str, default="./MODEL_SAVE")
    args = parser.parse_args()

    setup_seed(args.seed)

    model_save_path: Path = Path().cwd() / "MODEL_SAVE"
    newest_model_path = get_newest_model(model_save_path)
    logger.info(f"Using: {newest_model_path.name}")

    tick_folder = Path.cwd() / "../datas/000157.SZ/tick/gtja/"
    tick_files = list(tick_folder.glob("*.csv"))
    tick_files.sort(key=lambda x: datetime.strptime(x.stem, "%Y-%m-%d"))

    simu_folder = Path().cwd() / "REAL"
    logger.info(f"Simulating folder: {simu_folder}")
    train_folder = simu_folder / "NORM"
    train_files = list(train_folder.glob("*.csv"))
    little_lob_folder = simu_folder / "LLOB"

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
    optimizer = optim.Adam(ocet.parameters(), lr=0.0001, weight_decay=0.0005)
    optimizer.load_state_dict(para_dict["optimizer_state_dict"])

    lq = LimitedQueue(max_size=100)
    standard = Standarder(file_folder=simu_folder / "RAW", train=False, limits=5)
    standard.fresh_files()

    direction = "BUY"
    joye_data = JoyeLOB(window=100)
    llob = LittleOB(direction=direction)
    loss_func = MultiTaskLoss(alpha=0.5, beta=0.5)

    # 交易时间起始
    total_begin = 9_15_00_000
    trade_begin = 9_30_03_000
    end = 14_57_00_000

    trade_volume = 2000

    with t.no_grad():
        for tick_file in tick_files[-4:]:
            logger.info(f"Date: {tick_file.stem}")
            tick = DataSet(tick_file, ticker=ticker)
            ob = OrderBook(data_api=tick)
            raw_file = simu_folder / "RAW" / tick_file.name
            writer = Writer(filename=raw_file, rollback=3000)
            llob_writer = LLobWriter(
                tick=tick, orderbook=ob, file_name=simu_folder / "LLOB" / tick_file.name
            )

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
                args=args,
                littleob=llob_writer,
            )

            for timestamp in range(
                tick.file_date_num + total_begin, tick.file_date_num + end
            ):
                if trader.end_trade:
                    break
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
