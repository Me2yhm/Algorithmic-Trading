from log import logger, eval_log, train_log
from utils import setup_seed
import torch as t
from argparse import ArgumentParser
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = ArgumentParser(description='Arguments for the strategy', add_help=True)
    parser.add_argument('-s', '--seed', type=int, default=2333, help='set random seed')
    parser.add_argument('-e', '--epoch', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='./DATA/ML')
    parser.add_argument('--model-save', type=str, default='./MODEL_SAVE')
    args = parser.parse_args()

    logger.info("Starting".center(40, '='))

    dataset_path: Path = Path(args.dataset)
    assert dataset_path.exists(), 'Dataset path does not exist!'

    model_save_path: Path = Path(args.model_save)
    if not model_save_path.exists():
        model_save_path.mkdir()

    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    logger.info(f"Set device: {device}")

    setup_seed(args.seed)
    logger.info("Set seed: {}".format(args.seed))

    train_log(epoch=1, epochs=args.epoch, step=1, steps=20, loss=1.5, acc=0.65)
    train_log(epoch=1, epochs=args.epoch, step=10, steps=20, loss=1.5, acc=0.65)
    eval_log(epoch=1, acc=0.53)
