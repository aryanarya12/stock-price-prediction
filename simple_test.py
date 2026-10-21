"""
Simple test to verify the model can forward pass without errors
"""

import torch
import numpy as np
from model import RCDGAT
from config import TRAINING_CONFIG

print("Testing RC-DGAT model...")

# Create model
model = RCDGAT()
print("✓ Model created successfully")

# Create sample batch data
# The model expects:
# - price: (batch_size, num_stocks, feature_dim)
# - returns: (batch_size, num_stocks)
# - volatility: (batch_size, num_stocks)

batch_size = 4
num_stocks = 10
feature_dim = 31

# Create proper shapes
price = torch.randn(batch_size, num_stocks, feature_dim)
returns = torch.randn(batch_size, num_stocks)
volatility = torch.randn(batch_size, num_stocks)

print(f"Input shapes:")
print(f"  price: {price.shape}")
print(f"  returns: {returns.shape}")
print(f"  volatility: {volatility.shape}")

# Fit HMM on sufficient historical data
# Need at least 100+ samples for stable HMM fitting
np.random.seed(42)
T = 200  # Time periods
returns_hist = np.random.randn(T, num_stocks)
volatility_hist = np.abs(np.random.randn(T, num_stocks)) * 0.02

model.hmm_detector.fit(returns_hist, volatility_hist)
print("✓ HMM fitted successfully")

# Forward pass
try:
    predictions, regime_logits, regime_states = model(price, returns, volatility)
    print(f"✓ Forward pass successful")
    print(f"  - Predictions shape: {predictions.shape}")
    print(f"  - Regime logits shape: {regime_logits.shape}")
    print(f"  - Regime states shape: {regime_states.shape}")
    print("\n✅ All tests passed!")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
