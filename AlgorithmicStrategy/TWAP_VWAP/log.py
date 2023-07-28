from loguru import logger
from pathlib import Path
from datetime import date


def init_logger(path: str = './RUN_LOG'):
    log_path: Path = Path(path)
    if not log_path.exists():
        log_path.mkdir()
    logger.add(sink=log_path / '{}.log'.format(date.today()))


def eval_log(
        epoch: int,
        acc: float,
        **kwargs
):
    logger.info('Evaluating Network'.center(40, '='))
    logger.info("Test set: Epoch: {}, Current Accuracy: {:.4f}".format(
        epoch, acc
    ))
    if kwargs:
        logger.info(f"Other arguments: {kwargs}")


def train_log(
        epoch: int,
        epochs: int,
        step: int,
        steps: int,
        loss: float,
        acc: float,
        **kwargs
):
    logger.info(f"Epoch: {epoch}/{epochs}, Step: {step}/{steps},  Loss: {loss}, Acc: {acc}")
    if kwargs:
        logger.info(f"Other arguments: {kwargs}")


init_logger()
