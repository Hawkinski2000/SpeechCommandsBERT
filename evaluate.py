import torch
import os
from model import DataLoader
import torch.distributed as dist
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def evaluate(model, raw_model, ddp_config):
    B = 64
    ddp = ddp_config["ddp"]
    device = ddp_config["device"]
    master_process = ddp_config["master_process"]
    ddp_world_size = ddp_config["ddp_world_size"]
    ddp_rank = ddp_config["ddp_rank"]

    test_loader = DataLoader(B=B, process_rank=ddp_rank, num_processes=ddp_world_size, split="test", device=device, master_process=master_process)

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    # Load the checkpoint if it exists
    checkpoint_path = "checkpoints/checkpoint_200.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # Load model parameters
        state_dict = checkpoint['model_state_dict']
        raw_model_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith("module."):
                new_key = key[len("module."):] # strip the prefix
            raw_model_state_dict[new_key] = value
        raw_model.load_state_dict(raw_model_state_dict)

        if master_process:
            print(f"Checkpoint loaded successfully!")
    else:
        print("No checkpoint found.")

    raw_model.eval()
    test_loader.reset()
    with torch.no_grad():
        test_loss_accum = 0.0
        test_loss_steps = 60
        y_true = []
        y_pred = []
        for _ in range(test_loss_steps):
            encoder_x, y = test_loader.next_batch()
            encoder_x, y = encoder_x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(encoder_x, y)
            loss = loss / test_loss_steps
            test_loss_accum += loss.detach()
            y_true.append(y)
            y_pred.append(torch.argmax(logits, dim=1))
    if ddp:
        dist.all_reduce(test_loss_accum, op=dist.ReduceOp.AVG)
    if master_process:
        print(f"Test loss: {test_loss_accum.item():.4f}")

        y_true = torch.cat(y_true).cpu().numpy()
        y_pred = torch.cat(y_pred).cpu().numpy()
        print(len(y_pred))
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
        plt.savefig(f"confusion_matrix.png")