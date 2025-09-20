import torch
import os
import glob

from src.logger import logger
from src.utils import DEVICE

def get_model_dir(model):
    """Return folder path based on model class name."""
    model_name = model.__class__.__name__
    model_dir = os.path.join('./models', model_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def load_checkpoint(model, optimizer):
    model_dir = get_model_dir(model)
    ckpt_files = glob.glob(os.path.join(model_dir, "epoch_*.pth"))

    if ckpt_files:
        # Sort by epoch number extracted from filename
        ckpt_files.sort()
        latest_path = ckpt_files[-1]  # highest epoch

        checkpoint = torch.load(latest_path, weights_only=False, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"]
        logger.info(f"Resuming from {latest_path}")
        return start_epoch, history
    else:
        return 0, {'train_acc': [], 'test_acc': [],
                   'train_loss': [], 'test_loss': []}

def save_checkpoint(model, optimizer, epoch, history):
    model_dir = get_model_dir(model)
    ckpt_path = os.path.join(model_dir, f"epoch_{epoch:03d}.pth")

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
    }
    torch.save(checkpoint, ckpt_path)

def clear_checkpoints(model):
    model_dir = get_model_dir(model)
    if os.path.exists(model_dir):
        for f in glob.glob(os.path.join(model_dir, "*.pth")):
            os.remove(f)
        logger.info(f"Cleared all checkpoints in {model_dir}")