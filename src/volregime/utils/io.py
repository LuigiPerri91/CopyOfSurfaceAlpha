import pandas
import torch
import json
from pathlib import Path

def save_parquet(df, path):
    Path(path).parent_mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def load_parquet(path):
    return pandas.read_parquet(path)

def save_tensor(tensor, path):
    Path(path).parent_mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)

def load_tensor(path):
    return torch.load(path, weights_only=True)

def save_json(data, path):
    Path(path).parent_mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent =2, default=str)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_checkpoint(model, optimizer, epoch, val_loss, path):
    Path(path).parent_mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss
    },path)

def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt['epoch'], ckpt['val_loss']
