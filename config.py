from dataclasses import dataclass


@dataclass
class EncoderConfig:
    # Fixed hyperparameters
    block_size: int = 101 # max sequence length (spectrograms)
    n_ctx: int = (block_size) // 2 + 1 # sequence length after 2 Conv1D layers
    n_mels: int = 80
    n_classes: int = 36 # number of speech command classes

    # Tuned hyperparameters
    n_embd: int = 512 # embedding dimension
    n_layer: int = 4 # number of layers
    n_head: int = 4 # number of heads
    B: int = 64 # batch size
    attn_pdrop: float = 0.1 # attention dropout
    resid_pdrop: float = 0.1 # residual dropout
    mlp_pdrop: float = 0.1 # mlp dropout
    max_lr: float = 0.0018
    weight_decay: float = 0.1 # weight decay
    warmup_ratio: float = 0.1 # warmup ratio (% of max steps)