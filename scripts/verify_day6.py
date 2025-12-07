import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

# Ensure we can import from src
sys.path.append(os.getcwd())

from src.models.transformer_vol import VolTransformer
from src.models.time_embeddings import get_alibi_bias

def test_time2vec_gradients():
    print("\n=== TEST 1: Verifying Time2Vec Gradients ===")
    # 1. Initialize Model with Time2Vec
    model = VolTransformer(
        n_features=2, d_model=32, n_layers=1, n_heads=4, 
        embedding_type='time2vec'
    )
    
    # 2. Check initial weights
    # FIX: Access 1D tensor directly using [:3] instead of [0, :3]
    print(f"Initial Frequency Weight (w): {model.time_embedding.w.data[:3]} ...")
    initial_w = model.time_embedding.w.clone()
    
    # 3. Run a Dummy Step
    x = torch.randn(8, 20, 2) # [Batch, Seq, Feat]
    y = torch.randn(8)        # [Batch] target
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    
    output = model(x)
    loss = nn.MSELoss()(output, y)
    loss.backward()
    optimizer.step()
    
    # 4. Check if weights changed
    # FIX: Access 1D tensor directly
    print(f"Updated Frequency Weight (w): {model.time_embedding.w.data[:3]} ...")
    
    diff = torch.sum(torch.abs(model.time_embedding.w - initial_w)).item()
    if diff > 0:
        print(f"✅ SUCCESS: Time2Vec weights updated! (Diff: {diff:.6f})")
    else:
        print("❌ FAILURE: Time2Vec weights did not change. Gradients broken.")

def test_alibi_structure():
    print("\n=== TEST 2: Verifying ALiBi Mask Structure ===")
    seq_len = 5
    n_heads = 2
    
    # Generate the bias
    bias = get_alibi_bias(seq_len, n_heads, torch.device('cpu'))
    
    print(f"Bias Shape: {bias.shape} (Should be [{n_heads}, {seq_len}, {seq_len}])")
    
    # Check Head 0 (Long range penalty)
    print("\nHead 0 Bias Matrix (Diagonal should be 0, Past should be negative):")
    print(bias[0].numpy())
    
    # Verification Logic
    # 1. Diagonal should be 0
    diag_check = torch.all(torch.diagonal(bias[0]) == 0)
    
    # 2. Check logic: (t, t-2) should be MORE negative than (t, t-1)
    # Positions: Row=4 (t), Col=3 (t-1), Col=2 (t-2)
    val_t_minus_1 = bias[0, 4, 3].item() 
    val_t_minus_2 = bias[0, 4, 2].item() 
    
    print(f"\nPenalty at Dist 1 (t-1): {val_t_minus_1:.4f}")
    print(f"Penalty at Dist 2 (t-2): {val_t_minus_2:.4f}")
    
    if val_t_minus_2 < val_t_minus_1 and val_t_minus_1 <= 0:
        print("✅ SUCCESS: ALiBi penalty increases with distance.")
    else:
        print("❌ FAILURE: ALiBi penalty logic seems wrong.")

def test_overfitting_capability():
    print("\n=== TEST 3: Sanity Check (Overfitting a Tiny Batch) ===")
    
    # Data: y = mean of x features
    x = torch.randn(4, 10, 2)
    y = x.mean(dim=1).mean(dim=1) # Simple target
    
    model = VolTransformer(n_features=2, d_model=32, embedding_type='sinusoidal')
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("Training on tiny batch...")
    for i in range(50):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Iter {i}: Loss {loss.item():.6f}")
            
    if loss.item() < 0.05:
        print("✅ SUCCESS: Model can overfit small data (Architecture is valid).")
    else:
        print("❌ FAILURE: Model failed to converge on trivial data.")

if __name__ == "__main__":
    test_time2vec_gradients()
    test_alibi_structure()
    test_overfitting_capability()