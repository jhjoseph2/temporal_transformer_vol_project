"""
Stub for a generic PyTorch training loop for later Transformer experiments.
"""

from typing import Dict

import torch
from torch.utils.data import DataLoader

from .metrics import regression_metrics


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    loss_fn,
    device: str = "cpu",
    clip_grad: float = 1.0 # <--- Added argument
) -> float:
    model.train()
    total_loss = 0.0
    
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        
        # Ensure target shape matches predictions [batch] vs [batch, 1]
        if y.ndim > 1: y = y.squeeze(-1)
        
        loss = loss_fn(preds, y)
        
        if torch.isnan(loss):
            print("[WARNING] NaN loss detected! Skipping batch.")
            continue
            
        loss.backward()
        
        # Critical: Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate_model(
    model,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    model.eval()
    ys = []
    preds = []
    for x, y in dataloader:
        x = x.to(device)
        y_hat = model(x)
        ys.append(y.numpy())
        preds.append(y_hat.cpu().numpy())

    import numpy as np

    ys = np.concatenate(ys)
    preds = np.concatenate(preds)

    return regression_metrics(ys, preds)
