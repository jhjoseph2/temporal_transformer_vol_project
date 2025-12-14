import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import numpy as np

# Import from your existing src
from src.models.transformer_vol import VolTransformer
from src.data.lob_dataset import LOBDataset
from src.utils.seed import set_seed
from src.utils.logging import get_logger

logger = get_logger("lob_experiment")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", type=str, required=True, choices=["sinusoidal", "time2vec", "alibi", "ctlpe"])
    parser.add_argument("--train_file", type=str, required=True, help="Path to Training Fold .txt")
    parser.add_argument("--test_file", type=str, required=True, help="Path to Testing Fold .txt")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    # 1. Load Data
    logger.info("Initializing LOB Datasets...")
    # Standard HFT settings: Lookback 50 ticks, predict 10 ticks ahead
    train_ds = LOBDataset(args.train_file, lookback=50, horizon=10)
    test_ds = LOBDataset(args.test_file, lookback=50, horizon=10)
    
    # Large batch size is common/efficient for HFT
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    
    logger.info(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")

    # 2. Initialize Model
    # n_features=4 (Bid/Ask Price + Vol)
    model = VolTransformer(
        n_features=4, 
        d_model=64, 
        n_layers=2, 
        n_heads=4, 
        embedding_type=args.embedding,
        dropout=0.1
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    logger.info(f"Starting {args.embedding} experiment on {device}...")

    # Setup Checkpointing
    best_val_loss = float('inf')
    models_dir = Path("models/lob")
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / f"best_{args.embedding}.pt"

    # 3. Training Loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch['x'].to(device)
            t = batch['t'].to(device)
            y = batch['y'].to(device)
            
            optimizer.zero_grad()
            pred = model(x, t)
            
            # y is [batch, 1], pred is [batch] -> squeeze y
            loss = loss_fn(pred, y.squeeze(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation Pass
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                x = batch['x'].to(device)
                t = batch['t'].to(device)
                y = batch['y'].to(device)
                
                pred = model(x, t)
                val_loss += loss_fn(pred, y.squeeze(-1)).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Train MSE: {avg_train_loss:.7f} | Test MSE: {avg_val_loss:.7f}")

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            # logger.info("  > New best model saved.")

    # 4. Final Inference & Saving Results
    logger.info("Training complete. Loading best model for final evaluation...")
    
    # Load the best weights
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            t = batch['t'].to(device)
            y = batch['y'].to(device)
            
            pred = model(x, t)
            
            # Collect results (move to CPU numpy)
            all_preds.append(pred.cpu().numpy())
            all_actuals.append(y.squeeze(-1).cpu().numpy())

    # Concatenate all batches
    final_preds = np.concatenate(all_preds)
    final_actuals = np.concatenate(all_actuals)
    
    # Save to CSV
    results_df = pd.DataFrame({'actual': final_actuals, 'predicted': final_preds})
    
    results_dir = Path("results/lob")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"lob_{args.embedding}.csv"
    
    results_df.to_csv(out_path, index=False)
    logger.info(f"Saved LOB predictions to {out_path}")

if __name__ == "__main__":
    main()