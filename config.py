from dataclasses import dataclass


@dataclass
class EncoderConfig:
    # Fixed hyperparameters
    block_size: int = 101 # max sequence length (spectrograms)
    n_ctx: int = (block_size) // 2 + 1 # sequence length after 2 Conv1D layers
    n_mels: int = 80
    n_classes: int = 36 # number of speech command classes

    # Tuned hyperparameters
    n_layer: int = 2 # number of layers
    n_head: int = 8 # number of heads
    n_embd: int = 512 # embedding dimension
    B: int = 16 # batch size
    attn_pdrop: float = 0.3668828881050512 # attention dropout
    resid_pdrop: float = 0.3513009705241147 # residual dropout
    mlp_pdrop: float = 0.3587858310392814 # mlp dropout
    max_lr: float = 0.0028408216514440683 # max learning rate
    weight_decay: float = 0.1 # weight decay
    warmup_ratio: float = 0.1 # warmup ratio (% of max steps)