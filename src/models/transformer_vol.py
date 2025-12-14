import torch
import torch.nn as nn
from typing import Literal, Optional

# Import the ingredients we just prepared
from .time_embeddings import (
    SinusoidalPositionalEncoding, 
    LearnedPositionalEncoding, 
    Time2Vec, 
    CTLPE,
    get_alibi_bias
)

class VolTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 128,
        dropout: float = 0.1,
        embedding_type: str = 'sinusoidal'
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.embedding_type = embedding_type.lower()

        # 1. Input Projection Layer
        self.input_linear = nn.Linear(n_features, d_model)
        
        # 2. Initialize the specific embedding module (if needed)
        self.time_embedding = None
        if self.embedding_type == 'sinusoidal':
            self.time_embedding = SinusoidalPositionalEncoding(d_model)
        elif self.embedding_type == 'learned':
            self.time_embedding = LearnedPositionalEncoding(d_model)
        elif self.embedding_type == 'time2vec':
            # Time2Vec maps 1 time feature -> d_model dims
            self.time_embedding = Time2Vec(d_model)
        elif self.embedding_type == 'ctlpe':
            self.time_embedding = CTLPE(d_model)
        # Note: ALiBi requires no registered embedding layer, it uses a dynamic mask

        # 3. The Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True # Better stability for time series
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 4. Output Head
        self.out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [batch, seq_len, n_features]
        t: [batch, seq_len, 1] -> Actual continuous timestamps (e.g., normalized seconds)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # A. Feature Projection
        x_emb = self.input_linear(x) # [batch, seq, d_model]

        # B. Handle Additive Embeddings (Sinusoidal, Learned, Time2Vec)
        if self.embedding_type == 'sinusoidal':
            x_emb = self.time_embedding(x_emb)
            
        elif self.embedding_type == 'learned':
            x_emb = self.time_embedding(x_emb)
            
        elif self.embedding_type in ['time2vec', 'ctlpe']:
            # Create synthetic normalized time steps [0, 1] for the window
            # Shape: [batch, seq_len, 1]
            if t is None:
                # Fallback for Daily Data (Regular Intervals)
                t = torch.linspace(0, 1, seq_len, device=x.device).unsqueeze(0).unsqueeze(-1)
                t = t.expand(batch_size, -1, -1)
            
            # Now Time2Vec processes the ACTUAL irregular time intervals
            x_emb = x_emb + self.time_embedding(t)

        # C. Handle Attention Bias (ALiBi)
        # src_mask shape requirements for PyTorch nn.MultiheadAttention: 
        # (batch * num_heads, seq_len, seq_len)
        src_mask = None
        
        if self.embedding_type == 'alibi':
            # 1. Generate the bias for one batch [num_heads, seq, seq]
            alibi_bias = get_alibi_bias(seq_len, self.n_heads, device)
            
            # 2. Repeat for the whole batch -> [batch * num_heads, seq, seq]
            src_mask = alibi_bias.repeat(batch_size, 1, 1)

        # D. Transformer Encoding
        # We pass the calculated mask here. If embedding_type != alibi, src_mask is None.
        x_enc = self.encoder(x_emb, mask=src_mask)

        # E. Prediction Head (Pooling last token)
        last_token = x_enc[:, -1, :]
        return self.out(last_token).squeeze(-1)