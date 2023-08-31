def data_mark(price_list: list) -> list:
    price_max = max(price_list)
    price_min = min(price_list)
    return [(price_max - origin) / (price_max - price_min) for origin in price_list]
