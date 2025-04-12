from dataclasses import dataclass


@dataclass
class EncoderConfig:
    block_size: int = 101 # max sequence length (spectrograms)
    n_layer: int = 4 # number of layers
    n_head: int = 4 # number of heads
    n_embd: int = 512 # embedding dimension
    n_mels: int = 80
    n_ctx: int = (block_size) // 2 + 1 # sequence length after 2x Conv1D layers
    n_classes: int = 36 # number of speech command classes
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    mlp_pdrop: float = 0.1