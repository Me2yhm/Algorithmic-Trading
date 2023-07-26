from log import logger
from utils import setup_seed
import torch as t
from argparse import ArgumentParser
from pathlib import Path

if __name__ == '__main__':
    parser = ArgumentParser(description='Arguments for the strategy')
    parser.add_argument('-s', '--seed', type=int, default=2333, help='random seed')
    parser.add_argument('-e', '--epoch', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='./DATA/ML')
    parser.add_argument('--model-save', type=str, default='./MODEL_SAVE')
    args = parser.parse_args()

    logger.info("Starting")

    dataset_path: Path = Path(args.dataset)
    assert dataset_path.exists(), 'Dataset path does not exist!'

    model_save_path: Path = Path(args.model_save)
    if not model_save_path.exists():
        model_save_path.mkdir()

    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    logger.info("Set device: {}({})".format(device.type, device.index))

    setup_seed(args.seed)
    logger.info("Set seed: {}".format(args.seed))
