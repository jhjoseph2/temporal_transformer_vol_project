from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    # data
    data_path: str = "data/processed/spy_daily.parquet"
    lookback: int = 60         # sequence length
    horizon: int = 1           # predict 1 step ahead
    train_ratio: float = 0.7
    val_ratio: float = 0.15    # test = 1 - train - val

    # model params (for later Transformer)
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128
    dropout: float = 0.1

    # training params (for Transformer)
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 50
    device: str = "cuda"  # or "cpu"
    clip_grad: float = 1.0

    # Model Architecture Switch
    # Options: 'sinusoidal', 'learned', 'time2vec', 'alibi'
    embedding_type: str = "sinusoidal"