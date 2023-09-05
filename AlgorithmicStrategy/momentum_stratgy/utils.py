import csv
from typing import Iterable, Any


def data_mark(price_list: list) -> list:
    price_max = max(price_list)
    price_min = min(price_list)
    return [(price_max - origin) / (price_max - price_min) for origin in price_list]


def write_to_csv(line: Iterable[Any], filepath: str):
    with open(filepath, mode="+a", encoding="utf_8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(line)
