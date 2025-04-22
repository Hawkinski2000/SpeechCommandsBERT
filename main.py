import argparse

from ddp import setup_ddp, destroy_ddp
from config import EncoderConfig
from model import build_model
from train import Trainer
from evaluate import evaluate
from tune import Tuner


"""
simple launch:
    Train: python main.py --train
    Evaluate: python main.py --eval
    Tune: python main.py --tune

DDP launch for e.g. 8 GPUs:
    Train: torchrun --standalone --nproc_per_node=8 main.py --train
    Evaluate: torchrun --standalone --nproc_per_node=8 main.py --eval
    Tune: torchrun --standalone --nproc_per_node=8 main.py --tune

torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
"""


def main():
    ddp_config = setup_ddp()

    parser = argparse.ArgumentParser(description="Run the model in different modes.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train the model")
    group.add_argument("--eval", action="store_true", help="Evaluate the model")
    group.add_argument("--tune", action="store_true", help="Tune the model")
    args = parser.parse_args()

    if args.train:
        model, raw_model = build_model(ddp_config, EncoderConfig)
        trainer = Trainer(EncoderConfig)
        trainer.train(model, raw_model, ddp_config)

    if args.eval:
        model, raw_model = build_model(ddp_config, EncoderConfig)
        evaluate(model, raw_model, ddp_config)

    if args.tune:
        trainer = Trainer(EncoderConfig)
        tuner = Tuner(ddp_config, trainer)
        tuner.tune()

    destroy_ddp()

if __name__ == '__main__':
    main()