import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np


class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch
        # nh = heads, hs = head size, C (number of channels) = nh * hs
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,k, v, is_causal=False) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.attn_dropout(y) # dropout after attention
        # output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y) # dropout after projection
        return y
    

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.fc_dropout = nn.Dropout(config.mlp_pdrop)
        self.proj_dropout = nn.Dropout(config.mlp_pdrop)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.fc_dropout(x) # Dropout after activation
        x = self.c_proj(x)
        x = self.proj_dropout(x) # Dropout after projection
        return x
    

class EncoderBlock(nn.Module):

    def __init__(self, encoder_config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(encoder_config.n_embd)
        self.self_attn = SelfAttention(encoder_config)
        self.ln_2 = nn.LayerNorm(encoder_config.n_embd)
        self.mlp = MLP(encoder_config)

    def forward(self, encoder_x):
        encoder_x = encoder_x + self.self_attn(self.ln_1(encoder_x))
        encoder_x = encoder_x + self.mlp(self.ln_2(encoder_x))
        return encoder_x
    

class BERT(nn.Module):

    def __init__(self, encoder_config, device, master_process):
        super().__init__()
        self.encoder_config = encoder_config
        self.device = device
        self.master_process = master_process

        encoder_pe = self.sinusoids(self.encoder_config.n_ctx,
                                    self.encoder_config.n_embd)
        self.register_buffer("positional_embedding", encoder_pe)
        self.transformer = nn.ModuleDict(dict(
            encoder_conv1 = nn.Conv1d(self.encoder_config.n_mels,
                                      self.encoder_config.n_embd,
                                      kernel_size=3,
                                      padding=1),
            encoder_conv2 = nn.Conv1d(self.encoder_config.n_embd,
                                      self.encoder_config.n_embd,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1),
            encoder_h = nn.ModuleList(
                [EncoderBlock(self.encoder_config)
                    for _ in range(self.encoder_config.n_layer)]),
            ln_f = nn.LayerNorm(self.encoder_config.n_embd),
        ))
        self.lm_head = nn.Linear(self.encoder_config.n_embd,
                                 self.encoder_config.n_classes, 
                                 bias=False)

        self.to(device)

        # init params
        self.apply(self._init_weights)

    # Returns sinusoids for positional embedding
    def sinusoids(self, length, channels, max_timescale=1000):
        
        assert channels % 2 == 0
        
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)

        range_values = torch.arange(channels // 2)
        inv_timescales = torch.exp(-log_timescale_increment * range_values)

        time_steps = torch.arange(length)[:, np.newaxis]
        scaled_time = time_steps * inv_timescales[np.newaxis, :]

        return torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], dim=1
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.encoder_config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, spectrogram, targets=None):
        # spectrogram is of shape (B, T)
        encoder_T = spectrogram.size()[1]
        assert encoder_T <= self.encoder_config.block_size, (
            f"Cannot forward sequence of length {encoder_T}"
        )
        
        # forward the spectrograms through the Conv1D + GELU layers
        encoder_x = F.gelu(self.transformer.encoder_conv1(spectrogram))
        encoder_x = F.gelu(self.transformer.encoder_conv2(encoder_x))
        encoder_x = encoder_x.permute(0, 2, 1) # (B, 51, n_embd)

        # forward the spectrograms through the sinusoidal position embedding
        encoder_x = (
            encoder_x + self.positional_embedding).to(encoder_x.dtype)
        # encoder_x = (B, 51, n_embd)
        for encoder_block in self.transformer.encoder_h:
            encoder_x = encoder_block(encoder_x)

        # forward the final layernorm and the classifier.
        # The first time step is taken out of the sequence of 51,
        # similar to a "CLS" token in BERT models.
        encoder_x = encoder_x[:, :1, :].squeeze(1) # (B, n_embd)
        encoder_x = self.transformer.ln_f(encoder_x)
        logits = self.lm_head(encoder_x) # (B, 36)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.encoder_config.n_classes), targets)
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items()
                      if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight
        # decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases
        # and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if self.master_process:
            print(f"num decayed parameter tensors: "
                  f"{len(decay_params)}, with {num_decay_params:,} params")
            print(f"num non-decayed parameter tensors: "
                  f"{len(nodecay_params)}, "
                  f"with {num_nodecay_params:,} params")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(
            torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if self.master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups,
                                      lr=learning_rate,
                                      betas=(0.9, 0.95),
                                      eps=1e-8,
                                      fused=use_fused)
        return optimizer
    

def build_model(ddp_config, config):
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    torch.set_float32_matmul_precision('high')

    ddp = ddp_config["ddp"]
    device = ddp_config["device"]
    master_process = ddp_config["master_process"]
    ddp_local_rank = ddp_config["ddp_local_rank"]

    # create model
    model = BERT(config, device, master_process)
    model.to(device)
    use_compile = True
    if use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    return model, raw_model