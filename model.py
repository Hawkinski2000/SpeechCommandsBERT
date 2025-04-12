import os
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
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
        self.attn_dropout = nn.Dropout(config.attn_pdrop)  # Dropout after attention
        self.resid_dropout = nn.Dropout(config.resid_pdrop)  # Dropout after projection
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
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

def sinusoids(length, channels, max_timescale=1000):
    # Returns sinusoids for positional embedding
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class BERT(nn.Module):

    def __init__(self, encoder_config, device, master_process):
        super().__init__()
        self.encoder_config = encoder_config
        self.device = device
        self.master_process = master_process

        encoder_pe = sinusoids(self.encoder_config.n_ctx, self.encoder_config.n_embd)
        self.register_buffer("positional_embedding", encoder_pe)
        self.transformer = nn.ModuleDict(dict(
            encoder_conv1 = nn.Conv1d(self.encoder_config.n_mels, self.encoder_config.n_embd, kernel_size=3, padding=1),
            encoder_conv2 = nn.Conv1d(self.encoder_config.n_embd, self.encoder_config.n_embd, kernel_size=3, stride=2, padding=1),
            encoder_h = nn.ModuleList([EncoderBlock(self.encoder_config) for _ in range(self.encoder_config.n_layer)]),
            ln_f = nn.LayerNorm(self.encoder_config.n_embd),
        ))
        self.lm_head = nn.Linear(self.encoder_config.n_embd, self.encoder_config.n_classes, bias=False)

        self.to(device)

        # init params
        self.apply(self._init_weights)

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
        assert encoder_T <= self.encoder_config.block_size, f"Cannot forward sequence of length {encoder_T}, block size is only {self.encoder_config.block_size}"
        
        # forward the spectrograms through the Conv1D + GELU layers
        encoder_x = F.gelu(self.transformer.encoder_conv1(spectrogram))
        encoder_x = F.gelu(self.transformer.encoder_conv2(encoder_x))
        encoder_x = encoder_x.permute(0, 2, 1)

        # forward the spectrograms through the sinusoidal position embedding
        encoder_x = (encoder_x + self.positional_embedding).to(encoder_x.dtype) # (64, 512, 1501)

        for encoder_block in self.transformer.encoder_h:
            encoder_x = encoder_block(encoder_x)

        # forward the final layernorm and the classifier
        encoder_x = self.transformer.ln_f(encoder_x[:, :1, :]) # (B, 1, n_embd)
        logits = self.lm_head(encoder_x) # (B, 1(CLS token), 35(classes))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.encoder_config.n_classes), targets)
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if self.master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if self.master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

class DataLoader:
    def __init__(self, B, process_rank, num_processes, split, device, master_process):
        self.B = B  # Number of examples per batch
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.device = device
        self.master_process = master_process
        
        self.audio_file_path = r"data/audio"
        self.labels_file_path = r"data/labels"
        assert os.path.exists(self.audio_file_path), f"File {self.audio_file_path} not found"
        assert os.path.exists(self.labels_file_path), f"File {self.labels_file_path} not found"

        self.audio_shards = sorted([s for s in os.listdir(self.audio_file_path) if split in s])
        self.label_shards = sorted([s for s in os.listdir(self.labels_file_path) if split in s])
        self.audio_shards = [os.path.join(self.audio_file_path, s) for s in self.audio_shards]
        self.label_shards = [os.path.join(self.labels_file_path, s) for s in self.label_shards]

        assert len(self.audio_shards) > 0, f"no audio shards found for split {split}"
        assert len(self.label_shards) > 0, f"no label shards found for split {split}"
        assert len(self.audio_shards) == len(self.label_shards), f"missing one or more shards for {split} split"
        
        if master_process:
            print(f"found {len(self.audio_shards)} audio shards for {split} split")
            print(f"found {len(self.label_shards)} label shards for {split} split")
        
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.spectrograms = torch.load(self.audio_shards[self.current_shard], map_location=self.device, weights_only=True)
        self.labels = torch.load(self.label_shards[self.current_shard], map_location=self.device, weights_only=True).to(dtype=torch.long)
        self.current_position = self.B * self.process_rank

    def next_batch(self):
        # Spectrograms
        encoder_x = self.spectrograms[self.current_position : self.current_position+self.B].to(self.device) # (64, 80, 101)
        
        # Labels
        y = self.labels[self.current_position : self.current_position+self.B].to(self.device)

       # advance the current position in the spectrograms and labels tensors
        self.current_position += self.B * self.num_processes

        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (self.B * self.num_processes + 1) > len(self.spectrograms):
            self.current_shard = (self.current_shard + 1) % len(self.audio_shards)
            self.spectrograms = torch.load(self.audio_shards[self.current_shard], map_location=self.device, weights_only=True)
            self.labels = torch.load(self.label_shards[self.current_shard], map_location=self.device, weights_only=True).to(dtype=torch.long)
            self.current_position = self.B * self.process_rank

        return encoder_x, y