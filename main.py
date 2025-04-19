from ddp import setup_ddp, destroy_ddp
from config import EncoderConfig
from model import build_model
from train import Trainer
from evaluate import evaluate
from tune import Tuner 


"""
simple launch:
    python main.py

DDP launch for e.g. 8 GPUs:
    torchrun --standalone --nproc_per_node=8 main.py

torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
"""


def main():
    ddp_config = setup_ddp()

    choice = input("Enter 'train', 'eval', or 'tune': ")

    if choice == 'train':
        model, raw_model = build_model(ddp_config, EncoderConfig)
        trainer = Trainer(EncoderConfig)
        trainer.train(model, raw_model, ddp_config)

    if choice == 'eval':
        model, raw_model = build_model(ddp_config, EncoderConfig)
        evaluate(model, raw_model, ddp_config)

    if choice == 'tune':
        trainer = Trainer(EncoderConfig)
        tuner = Tuner(ddp_config, trainer)
        tuner.tune()

    destroy_ddp()

if __name__ == '__main__':
    main()