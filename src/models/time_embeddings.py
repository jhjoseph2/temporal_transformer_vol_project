"""
Stubs for different temporal embedding modules.

For Week 1, you don't need these yet. We'll fill them in when working
on Transformer models.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        pos_emb = self.pe(positions)
        return x + pos_emb


class Time2Vec(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Linear component parameters
        self.w0 = nn.Parameter(torch.randn(1, 1)) # Shape explicitly for broadcasting
        self.b0 = nn.Parameter(torch.randn(1))
        
        # Periodic component parameters
        # Usage of Uniform initialization is generally more stable for frequencies
        self.w = nn.Parameter(torch.Tensor(d_model - 1))
        self.b = nn.Parameter(torch.Tensor(d_model - 1))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize frequencies to be small to avoid high-freq noise start
        nn.init.uniform_(self.w, 0, 1) 
        nn.init.uniform_(self.b, 0, 2 * math.pi)
        nn.init.normal_(self.w0, 0, 0.1)
        nn.init.zeros_(self.b0)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [batch, seq_len, 1]
        """
        # Linear term
        v0 = t * self.w0 + self.b0
        
        # Periodic term
        # Broadcasting: [B, S, 1] * [D-1] -> [B, S, D-1]
        v1 = torch.sin(t * self.w + self.b)
        
        return torch.cat([v0, v1], dim=-1)

class CTLPE(nn.Module):
    """
    Continuous Time Linear Positional Encoding.
    From 'Learning Continuous Time Representations' literature.
    
    Formula: PE(t)_i = w_i * t + b_i
    
    This allows the model to learn a linear function of time for each
    dimension of the d_model. Useful for capturing trends or decaying
    importance over time.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # We learn a separate slope (w) and bias (b) for every dimension
        # Shape: [1, 1, d_model] to broadcast over batch and sequence
        self.w = nn.Parameter(torch.randn(1, 1, d_model))
        self.b = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize with small values to prevent exploding gradients early on
        nn.init.normal_(self.w, mean=0.0, std=0.02)
        nn.init.zeros_(self.b)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [batch, seq_len, 1] - Continuous timestamps
        Returns:
            [batch, seq_len, d_model]
        """
        # t is [B, S, 1], w is [1, 1, D]. 
        # Broadcasting logic: [B, S, 1] * [1, 1, D] -> [B, S, D]
        return t * self.w + self.b

def get_alibi_slope(num_heads: int) -> torch.Tensor:
    """
    Calculate the specific slope for each attention head.
    Reference: Press et al., 'Train Short, Test Long' (ALiBi paper)
    """
    x = (2 ** 8) ** (1 / num_heads)
    return torch.tensor([1 / (x ** (i + 1)) for i in range(num_heads)])

def get_alibi_bias(seq_len: int, num_heads: int, device: torch.device) -> torch.Tensor:
    """
    Generate the ALiBi additive bias mask for the attention mechanism.
    Shape: [num_heads, seq_len, seq_len]
    """
    # 1. Create a distance matrix (0, -1, -2, ..., -seq_len)
    context_position = torch.arange(seq_len, device=device)[:, None]
    memory_position = torch.arange(seq_len, device=device)[None, :]
    
    # Causal masking logic: positions in the future are masked out (handled by standard mask)
    # ALiBi logic: positions in the past are penalized by distance
    relative_position = memory_position - context_position 
    relative_position = torch.abs(relative_position) * -1 # Make distances negative
    
    # 2. Get slopes for each head
    slopes = get_alibi_slope(num_heads).to(device)
    
    # 3. Broadcast to create shape [num_heads, seq_len, seq_len]
    # We unsqueeze slopes to [num_heads, 1, 1] so it broadcasts against [1, seq, seq]
    alibi_bias = slopes.unsqueeze(1).unsqueeze(1) * relative_position.unsqueeze(0)
    
    return alibi_bias