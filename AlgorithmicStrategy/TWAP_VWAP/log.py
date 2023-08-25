from typing import Union

from loguru import logger
from pathlib import Path
from datetime import date


def init_logger(path: Union[str, Path]):
    log_path: Path = Path(path)
    if not log_path.exists():
        log_path.mkdir()
    logger.add(sink=log_path / "{}.log".format(date.today()))


def log_eval(epoch: int, acc: float, **kwargs):
    logger.info("Evaluating Network".center(40, "="))
    logger.info("Test set: Epoch: {}, Current Accuracy: {:.4f}".format(epoch, acc))
    if kwargs:
        logger.info(f"Other arguments: {kwargs}")


def log_train(
    epoch: int, epochs: int, file:str, loss: float, **kwargs
):
    logger.info(
        f"Epoch: {epoch}/{epochs},file {file}, Loss: {loss}"
    )
    if kwargs:
        logger.info(f"Other arguments: {kwargs}")


init_logger(path=Path(__file__).parent / "RUN_LOG")
