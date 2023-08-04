def get_date(timestamp: int):
    date = str(timestamp)[:8]
    return "-".join([date[:4], date[4:6], date[6:]])
