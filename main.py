import os
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from config import EncoderConfig
from model import BERT
from train import train


# simple launch:
# python main.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 main.py
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE


def setup_ddp():
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
    return {
        "ddp": ddp,
        "device": device,
        "ddp_rank": ddp_rank,
        "ddp_local_rank": ddp_local_rank,
        "ddp_world_size": ddp_world_size,
        "master_process": master_process,
    }

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

torch.set_float32_matmul_precision('high')

ddp_config = setup_ddp()
ddp = ddp_config["ddp"]
device = ddp_config["device"]
master_process = ddp_config["master_process"]
ddp_local_rank = ddp_config["ddp_local_rank"]

# create model
model = BERT(EncoderConfig, device, master_process)
model.to(device)
use_compile = True
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

train(model, raw_model, ddp_config)

if ddp:
    destroy_process_group()