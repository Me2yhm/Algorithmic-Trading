
from pathlib import Path
from AlgorithmicStrategy import OrderBook, DataSet, LLobWriter
from tqdm import tqdm

if __name__ == "__main__":
    tick_folder = Path.cwd() / "./DATA/000157.SZ/tick/gtja/"
    print(tick_folder)
    tick_files = list(tick_folder.glob("*.csv"))
    ticker = "000157.SZ"

    for tick_file in tqdm(tick_files):
        tick = DataSet(tick_file, ticker=ticker)
        ob = OrderBook(data_api=tick)
        llob = LLobWriter(
            tick, ob, Path().cwd() / "./DATA/ML/LittleOB" / tick_file.name
        )
        ob.update()

        llob.write_llob()
