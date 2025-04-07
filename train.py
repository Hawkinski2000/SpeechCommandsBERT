import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------

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

@dataclass
class EncoderConfig:
    block_size: int = 101 # max sequence length (spectrograms)
    n_layer: int = 4 # number of layers
    n_head: int = 4 # number of heads
    n_embd: int = 512 # embedding dimension
    n_mels: int = 80
    n_ctx: int = (block_size) // 2 + 1 # sequence length after 2x Conv1D layers
    n_classes: int = 35 # number of speech command classes
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    mlp_pdrop: float = 0.1

def sinusoids(length, channels, max_timescale=1000):
    # Returns sinusoids for positional embedding
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

class BERT(nn.Module):

    def __init__(self, encoder_config):
        super().__init__()
        self.encoder_config = encoder_config

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
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------

def load_data(filename):
    ptt = torch.load(filename, map_location=device, weights_only=True)
    return ptt

class DataLoader:
    def __init__(self, B, process_rank, num_processes, split):
        self.B = B  # Number of examples per batch
        self.process_rank = process_rank
        self.num_processes = num_processes

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
        self.spectrograms = load_data(self.audio_shards[self.current_shard]) 
        self.labels = load_data(self.label_shards[self.current_shard]).to(dtype=torch.long)
        self.current_position = self.B * self.process_rank

    def next_batch(self):
        # Spectrograms
        encoder_x = self.spectrograms[self.current_position : self.current_position+B].to(device) # (64, 80, 101)
        
        # Labels
        y = self.labels[self.current_position : self.current_position+B].to(device)

       # advance the current position in the spectrograms and labels tensors
        self.current_position += self.B * self.num_processes

        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (self.B * self.num_processes + 1) > len(self.spectrograms):
            self.current_shard = (self.current_shard + 1) % len(self.audio_shards)
            self.spectrograms = load_data(self.audio_shards[self.current_shard])
            self.labels = load_data(self.label_shards[self.current_shard]).to(dtype=torch.long)
            self.current_position = self.B * self.process_rank

        return encoder_x, y

# -----------------------------------------------------------------------------
# simple launch:
# python train.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 16384 # 2**14
B = 64 # micro batch size
encoder_T = EncoderConfig.block_size # Spectrogram sequence length (101)
assert total_batch_size % (B * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * ddp_world_size"
grad_accum_steps = total_batch_size // (B * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoader(B=B, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoader(B=B, process_rank=ddp_rank, num_processes=ddp_world_size, split="validation")
test_loader = DataLoader(B=B, process_rank=ddp_rank, num_processes=ddp_world_size, split="test")

torch.set_float32_matmul_precision('high')

# create model
model = BERT(EncoderConfig)
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
use_compile = True # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr = 18e-4
min_lr = max_lr * 0.1
warmup_steps = 100
max_steps = 1000 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=18e-4, device_type=device_type)

# Load the checkpoint if it exists, otherwise the model will train from scratch 
checkpoint_path = "checkpoints/checkpoint_1000.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Load model parameters
    state_dict = checkpoint['model_state_dict']
    raw_model_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("module."):
            new_key = key[len("module."):]  # strip the prefix
        raw_model_state_dict[new_key] = value
    raw_model.load_state_dict(raw_model_state_dict)

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Retrieve additional training state (e.g., step count)

    step = checkpoint['step']
    start_step = step + 1

    if master_process:
        print(f"Checkpoint loaded successfully! Resuming from step {step}.")
else:
    start_step = 0  # Starting from scratch if no checkpoint is found
    print("No checkpoint found. Initializing model from scratch.")

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

# Save model and optimizer state at a checkpoint
def save_checkpoint(model, optimizer, step, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at step {step} to {checkpoint_path}")

def train():
    train_losses = []
    val_losses = []
    local_dir = "loss_tensors"
    LOSS_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(LOSS_DIR, exist_ok=True)
    for step in range(start_step, max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # once in a while evaluate our validation loss
        if step > 0 and step % 50 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    encoder_x, y = val_loader.next_batch()
                    encoder_x, y = encoder_x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(encoder_x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                val_losses += [val_loss_accum.item()]

                train_loss_tensor = torch.tensor(train_losses, dtype=torch.long)
                val_loss_tensor = torch.tensor(val_losses, dtype=torch.long)
                train_loss_path = os.path.join(LOSS_DIR, f"train_loss{step}.pt")
                val_loss_path = os.path.join(LOSS_DIR, f"train_loss{step}.pt")
                torch.save(train_loss_tensor, train_loss_path)
                torch.save(val_loss_tensor, val_loss_path)

                train_steps = list(range(len(train_losses))) # every step
                val_steps = list(range(0, len(train_losses), 50)) # every 50 steps
                plt.plot(train_steps, train_losses, label="Train Loss")
                plt.plot(val_steps, val_losses, label="Validation Loss")
                plt.xlabel("Steps")
                plt.ylabel("Loss")
                plt.title("Training and Validation Loss")
                plt.savefig(f"loss_curve{step}.png")

                save_checkpoint(model, optimizer, step)

        # do one step of the optimization
        model.train()
        optimizer.zero_grad()
        train_loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            encoder_x, y = train_loader.next_batch()
            encoder_x, y = encoder_x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(encoder_x, y)
            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / grad_accum_steps
            train_loss_accum += loss.detach()
            loss.backward()
        if ddp:
            dist.all_reduce(train_loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize() # wait for the GPU to finish work

        t1 = time.time()
        dt = t1 - t0 # time difference in seconds

        if master_process:
            print(f"step {step:5d} | loss: {train_loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms")
            with open(log_file, "a") as f:
                f.write(f"{step} train {train_loss_accum.item():.6f}\n")
            train_losses += [train_loss_accum.item()]

train()

if ddp:
    destroy_process_group()