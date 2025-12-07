import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

from src.config import ExperimentConfig
from src.data.vol_dataset import TimeSeriesVolDataset
from src.data.splitter import ExpandingWindowSplitter
from src.models.transformer_vol import VolTransformer
from src.training.loop import train_one_epoch, evaluate_model
from src.utils.seed import set_seed
from src.utils.logging import get_logger

logger = get_logger("walk_forward")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", type=str, required=True, choices=["sinusoidal", "time2vec", "alibi"])
    parser.add_argument("--epochs", type=int, default=30) # Fewer epochs per fold usually needed
    args = parser.parse_args()

    cfg = ExperimentConfig()
    cfg.embedding_type = args.embedding
    cfg.max_epochs = args.epochs
    # Auto-detect: Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    set_seed(42)

    # 1. Load Data
    df = pd.read_parquet(cfg.data_path)
    logger.info(f"Loaded data: {len(df)} rows")
    
    feature_cols = ["log_ret", "rvol"]
    target_col = "rvol"

    # 2. Setup Splitter
    # Train on ~2 years (500 days), Test on 3 months (60 days)
    splitter = ExpandingWindowSplitter(
        total_samples=len(df),
        initial_train_size=500, 
        test_size=120
    )

    all_preds = []
    all_actuals = []
    all_dates = []

    # 3. Walk-Forward Loop
    for fold, (train_idx, val_idx, test_idx) in enumerate(splitter.split()):
        logger.info(f"--- Fold {fold} ---")
        logger.info(f"Train: {train_idx[0]}-{train_idx[1]} | Test: {test_idx[0]}-{test_idx[1]}")

        # Create Datasets using explicit indices
        train_ds = TimeSeriesVolDataset(df, feature_cols, target_col, lookback=cfg.lookback, indices=train_idx)
        val_ds = TimeSeriesVolDataset(df, feature_cols, target_col, lookback=cfg.lookback, indices=val_idx)
        test_ds = TimeSeriesVolDataset(df, feature_cols, target_col, lookback=cfg.lookback, indices=test_idx)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

        # Re-Initialize Model (Fresh start for every fold ensures no leakage)
        model = VolTransformer(
            n_features=len(feature_cols),
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            embedding_type=cfg.embedding_type
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.MSELoss()

        # Train Loop (Shortened for brevity)
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(cfg.max_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, clip_grad=cfg.clip_grad)
            val_metrics = evaluate_model(model, val_loader, device)
            
            if val_metrics['mse'] < best_val_loss:
                best_val_loss = val_metrics['mse']
                best_model_state = model.state_dict()
        
        # Test Inference
        model.load_state_dict(best_model_state)
        
        # Save Predictions for this fold
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x = x.to(device)
                p = model(x).cpu().numpy()
                all_preds.append(p)
                all_actuals.append(y.numpy())
                
                # Get dates corresponding to these predictions
                # Global indices in df corresponding to the targets
                # The dataset returns indices relative to the split start
                # We need the global date from the dataframe
                # batch_start_idx = test_idx[0] + (i * cfg.batch_size) + cfg.lookback + cfg.horizon - 1
                # We can approximate or just track global indices if needed. 
                # For now, let's just stack values.

    # 4. Save Final Consolidated Results
    final_preds = np.concatenate(all_preds)
    final_actuals = np.concatenate(all_actuals)
    
    results_df = pd.DataFrame({'actual': final_actuals, 'predicted': final_preds})
    out_path = Path(f"results/wfv_{cfg.embedding_type}.csv")
    out_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(out_path, index=False)
    logger.info(f"Saved Walk-Forward results to {out_path}")

if __name__ == "__main__":
    main()