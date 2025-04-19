import os
import math
import time

import torch
import torch.distributed as dist
import matplotlib.pyplot as plt

from dataloader import DataLoader


class Trainer():
    def __init__(self, config):
        self.config = config
        self.max_lr = config.max_lr
        self.min_lr = self.max_lr * 0.1
        self.warmup_steps = 20
        self.max_steps = 201
        self.B = config.B
        self.checkpoint_path = "checkpoints/checkpoint_200.pt"

    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_steps:
            return self.max_lr * (it+1) / self.warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.max_steps:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (
            (it - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    # Save model and optimizer state at a checkpoint
    def save_checkpoint(self, 
                        model, 
                        optimizer, 
                        step, 
                        checkpoint_dir="checkpoints"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        checkpoint_path = os.path.join(checkpoint_dir,
                                       f"checkpoint_{step}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at step {step} to {checkpoint_path}")

    def train(self, model, raw_model, ddp_config):
        ddp = ddp_config["ddp"]
        device = ddp_config["device"]
        master_process = ddp_config["master_process"]
        ddp_world_size = ddp_config["ddp_world_size"]
        ddp_rank = ddp_config["ddp_rank"]

        total_batch_size = 16384 # 2**14
        B = self.B # micro batch size
        assert total_batch_size % (B * ddp_world_size) == 0, (
            "make sure total_batch_size is divisible by B * ddp_world_size"
        )
        grad_accum_steps = total_batch_size // (B * ddp_world_size)
        if master_process:
            print(f"total desired batch size: {total_batch_size}")
            print(f"=> calculated gradient accumulation steps:"
                  f" {grad_accum_steps}"
            )

        train_loader = DataLoader(B=B, 
                                  process_rank=ddp_rank,
                                  num_processes=ddp_world_size,
                                  split="train",
                                  device=device,
                                  master_process=master_process)
        val_loader = DataLoader(B=B,
                                process_rank=ddp_rank,
                                num_processes=ddp_world_size,
                                split="validation",
                                device=device,
                                master_process=master_process)

        device_type = "cuda" if device.startswith("cuda") else "cpu"
        # optimize!
        optimizer = raw_model.configure_optimizers(weight_decay=0.1,
                                                   learning_rate=18e-4,
                                                   device_type=device_type)

        # Load the checkpoint if it exists, otherwise train from scratch
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path,
                                    map_location=device,
                                    weights_only=True)

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
                print(f"Checkpoint loaded successfully! Resuming from step "
                      f"{step}."
                )
        else:
            start_step = 0  # Starting from scratch if no checkpoint is found
            if master_process:
                print("No checkpoint found. Initializing model from scratch.")

        # create the log directory we will write checkpoints to and log to
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"log.txt")
        with open(log_file, "w") as f: # open for writing to clear the file
            pass

        train_losses = []
        val_losses = []
        local_dir = "loss_tensors"
        LOSS_DIR = os.path.join(os.path.dirname(__file__), local_dir)
        os.makedirs(LOSS_DIR, exist_ok=True)
        for step in range(start_step, self.max_steps):
            t0 = time.time()
            last_step = (step == self.max_steps - 1)

            # once in a while evaluate our validation loss
            if step % 5 == 0 or last_step:
                raw_model.eval()
                val_loader.reset()
                with torch.no_grad():
                    val_loss_accum = 0.0
                    val_loss_steps = 20
                    for _ in range(val_loss_steps):
                        encoder_x, y = val_loader.next_batch()
                        encoder_x, y = encoder_x.to(device), y.to(device)
                        with torch.autocast(device_type=device_type,
                                            dtype=torch.bfloat16):
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

                    train_loss_tensor = torch.tensor(train_losses,
                                                     dtype=torch.long)
                    val_loss_tensor = torch.tensor(val_losses,
                                                   dtype=torch.long)
                    train_loss_path = os.path.join(LOSS_DIR,
                                                   f"train_loss{step}.pt")
                    val_loss_path = os.path.join(LOSS_DIR,
                                                 f"train_loss{step}.pt")
                    torch.save(train_loss_tensor, train_loss_path)
                    torch.save(val_loss_tensor, val_loss_path)

                    if step > 0 and step % 50 == 0 or last_step:
                        # every step
                        train_steps = list(range(len(train_losses)))

                        # every 5 steps
                        val_steps = list(range(0, len(train_losses), 5))
                        
                        val_losses = val_losses[:len(val_steps)]

                        plt.plot(train_steps,
                                 train_losses,
                                 label="Train Loss",
                                 color="#00ffd0")
                        plt.plot(val_steps,
                                 val_losses,
                                 label="Validation Loss",
                                 color="#ff00e3")
                        plt.xlabel("Steps")
                        plt.ylabel("Loss")
                        plt.title("Training and Validation Loss")
                        plt.legend()
                        plt.savefig(f"loss_curve{step}.png")
                        plt.clf()

                        self.save_checkpoint(raw_model, optimizer, step)

            # do one step of the optimization
            model.train()
            optimizer.zero_grad()
            train_loss_accum = 0.0
            for micro_step in range(grad_accum_steps):
                encoder_x, y = train_loader.next_batch()
                encoder_x, y = encoder_x.to(device), y.to(device)
                if ddp:
                    model.require_backward_grad_sync = (
                        micro_step == grad_accum_steps - 1
                    )
                with torch.autocast(device_type=device_type,
                                    dtype=torch.bfloat16):
                    logits, loss = model(encoder_x, y)
                # we have to scale the loss to account for gradient
                # accumulation, because the gradients just add on each
                # successive backward(). addition of gradients corresponds to
                # a SUM in the objective, but instead of a SUM we want MEAN.
                # Scale the loss here so it comes out right
                loss = loss / grad_accum_steps
                train_loss_accum += loss.detach()
                loss.backward()
            if ddp:
                dist.all_reduce(train_loss_accum, op=dist.ReduceOp.AVG)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # determine and set the learning rate for this iteration
            lr = self.get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
            if device_type == "cuda":
                torch.cuda.synchronize() # wait for the GPU to finish work

            t1 = time.time()
            dt = t1 - t0 # time difference in seconds

            if master_process:
                print(f"step {step:5d} | loss: {train_loss_accum.item():.6f} "
                      f"| lr {lr:.4e} | norm: {norm:.4f} "
                      f"| dt: {dt*1000:.2f}ms")
                with open(log_file, "a") as f:
                    f.write(f"{step} train {train_loss_accum.item():.6f}\n")
                train_losses += [train_loss_accum.item()]

        return val_losses[-1]