import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import numpy as np

# Import your project modules
from src.config import ExperimentConfig
from src.data.vol_dataset import TimeSeriesVolDataset
from src.models.transformer_vol import VolTransformer
from src.training.loop import train_one_epoch, evaluate_model
from src.utils.seed import set_seed
from src.utils.logging import get_logger

logger = get_logger("train_transformer")

def save_predictions(model, loader, device, path):
    """
    Run inference and save Actual vs Predicted values to CSV.
    Essential for the 'Quant Analysis' phase later.
    """
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # Model returns [batch], y is [batch]
            p = model(x)
            preds.append(p.cpu().numpy())
            actuals.append(y.numpy())
    
    # Flatten list of arrays
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    
    # Save
    df = pd.DataFrame({'actual': actuals, 'predicted': preds})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Predictions saved to {path}")

def main():
    parser = argparse.ArgumentParser()
    # This argument lets us run 3 different experiments with 1 script
    parser.add_argument("--embedding", type=str, default="sinusoidal", 
                        choices=["sinusoidal", "learned", "time2vec", "alibi"],
                        help="Temporal embedding type to use")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    # 1. Setup Config & Device
    cfg = ExperimentConfig()
    
    # Override config defaults with command line args
    cfg.embedding_type = args.embedding 
    cfg.max_epochs = args.epochs
    
    set_seed(42)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting experiment: {cfg.embedding_type} on {device}")

    # 2. Data Loading
    if not Path(cfg.data_path).exists():
        logger.error(f"Data file not found at {cfg.data_path}. Run preprocessing first.")
        return

    df = pd.read_parquet(cfg.data_path)
    
    # We use log returns and realized vol as input features
    feature_cols = ["log_ret", "rvol"] 
    target_col = "rvol"
    
    # Create Datasets
    train_ds = TimeSeriesVolDataset(df, feature_cols, target_col, 
                                    lookback=cfg.lookback, horizon=cfg.horizon, split="train")
    val_ds = TimeSeriesVolDataset(df, feature_cols, target_col, 
                                  lookback=cfg.lookback, horizon=cfg.horizon, split="val")
    test_ds = TimeSeriesVolDataset(df, feature_cols, target_col, 
                                   lookback=cfg.lookback, horizon=cfg.horizon, split="test")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    logger.info(f"Data loaded. Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # 3. Model Initialization
    # The VolTransformer will internally swap embeddings based on cfg.embedding_type
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

    # 4. Training Loop
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    best_val_loss = float("inf")
    save_path = save_dir / f"best_{cfg.embedding_type}.pt" # Define path once
    
    logger.info("Starting training loop...")
    for epoch in range(cfg.max_epochs):
        # Pass the new clip_grad param
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, clip_grad=cfg.clip_grad)
        val_metrics = evaluate_model(model, val_loader, device)
        val_loss = val_metrics["mse"]

        # Log EVERY epoch so we can see if it's working
        logger.info(f"Epoch {epoch+1}/{cfg.max_epochs} | Train Loss: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    # 5. Final Test & Inference
    if save_path.exists():
        logger.info("Loading best model for testing...")
        model.load_state_dict(torch.load(save_path))
    else:
        logger.warning("No best model found (Training failed to improve). Using last epoch weights.")
    
    test_metrics = evaluate_model(model, test_loader, device)
    logger.info(f"Test Set MSE: {test_metrics['mse']:.6f} | MAE: {test_metrics['mae']:.6f}")
    
    # Save predictions to CSV for Phase 3 analysis
    save_predictions(model, test_loader, device, results_dir / f"{cfg.embedding_type}_preds.csv")

if __name__ == "__main__":
    main()