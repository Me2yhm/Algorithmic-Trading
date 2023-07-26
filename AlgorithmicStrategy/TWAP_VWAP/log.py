from loguru import logger
from pathlib import Path
from datetime import date


def init_logger(path: str = './RUN_LOG'):
    log_path: Path = Path(path)
    if not log_path.exists():
        log_path.mkdir()
    logger.add(sink=log_path / '{}.log'.format(date.today()))
    logger.info("Initializing logger".center(40, '='))


init_logger()
