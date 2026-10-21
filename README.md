# RC-DGAT: Regime-Conditioned Dynamic Graph Attention Network

A comprehensive implementation of **RC-DGAT** (Regime-Conditioned Dynamic Graph Attention Network) for stock return forecasting using graph neural networks and hidden Markov models.

## Project Overview

This project implements a state-of-the-art deep learning model for predicting stock returns by:

1. **Detecting Market Regimes** using Hidden Markov Models (HMM)
   - Identifies Bull, Bear, and High-Volatility market states
   - Uses return-volatility observations as input signals
   - Dynamically updates regime detection in real-time

2. **Constructing Dynamic Graphs** based on stock correlations
   - Builds correlation-based graphs between stocks
   - Regime-aware edge weighting and topology adjustment
   - Tighter graphs in volatile/bear markets

3. **Applying Graph Attention Networks (GAT)**
   - Multi-head attention mechanism for cross-sectional dependencies
   - Learns weighted aggregation of stock relationships
   - Captures complex multi-stock interactions

4. **Temporal Encoding with Regime-Aware GRU**
   - Gated Recurrent Unit with regime modulation
   - Regime information influences memory gates
   - Adaptive temporal dynamics based on market state

5. **Return Forecasting via MLP Head**
   - Predicts next-day stock returns
   - Multi-layer perceptron prediction head
   - Joint optimization with regime classification

## Key Results (from Paper)

| Metric | Value |
|--------|-------|
| Directional Accuracy | 52.80% |
| RMSE | 0.0231 |
| Sharpe Ratio | 0.168 |
| CAGR | 8.4% |
| Max Drawdown | 14.2% |

## Architecture

```
Input: OHLCV Data (5 features) + Technical Indicators (26 features) = 31 features/stock
   ↓
[HMM Regime Detector] → Detect Bull/Bear/HighVol states
   ↓
[Dynamic Graph Constructor] → Build correlation graph with regime modulation
   ↓
[Multi-Head GAT] → Cross-sectional feature aggregation (8 heads)
   ↓
[Regime-Aware GRU] → Temporal encoding with regime gating (2 layers)
   ↓
[MLP Head] → Stock return prediction
   ↓
Output: Next-day return predictions + Regime classification
```

## File Structure

```
stock regime/
├── config.py                 # All configuration parameters
├── model.py                  # RC-DGAT model architecture
├── data_pipeline.py          # Data loading, preprocessing, and utilities
├── train.py                  # Main training script
├── evaluation.py             # Metrics, backtesting, and visualization
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download and Prepare Data

The `DataPreprocessor` class automatically:
- Downloads OHLCV data from Yahoo Finance
- Computes 26 technical indicators
- Aligns dates across multiple stocks
- Creates walk-forward validation splits

### 3. Train Model

```bash
python train.py
```

Key features:
- **Walk-forward validation**: 5 independent splits with 3-year train / 6-month val / 1-year test windows
- **Automatic checkpointing**: Best models saved based on validation directional accuracy
- **Early stopping**: Stops after 20 epochs without improvement
- **Regime loss ramp-up**: Gradually increases regime classification loss over first 20 epochs
- **Mixed precision**: FP16 training on NVIDIA GPUs for efficiency

### 4. Evaluate Results

The evaluation includes:
- **Regression metrics**: RMSE, MAE, MAPE
- **Classification metrics**: Directional Accuracy, Precision, Recall, F1
- **Portfolio metrics**: Sharpe Ratio, Sortino Ratio, Max Drawdown, CAGR
- **Statistical tests**: Diebold-Mariano test for forecast comparison

## Configuration

Edit `config.py` to customize:

### HMM Configuration
```python
HMM_CONFIG = {
    'num_states': 3,              # Bull, Bear, High-Volatility
    'observation_window': 252,    # 1 year of data for HMM
    'n_iter': 1000               # EM iterations
}
```

### Graph Configuration
```python
GRAPH_CONFIG = {
    'method': 'correlation',
    'window_size': 20,            # 20-day rolling correlation
    'threshold': 0.3,             # Minimum edge weight
    'regime_weight_dict': {
        'Bull': {'correlation_scale': 1.0},
        'Bear': {'correlation_scale': 1.2},
        'HighVolatility': {'correlation_scale': 1.3}
    }
}
```

### GAT Configuration
```python
GAT_CONFIG = {
    'input_dim': 31,              # Number of input features
    'num_heads': 8,               # Attention heads
    'hidden_dim': 64,
    'output_dim': 32,
    'dropout': 0.3
}
```

### Training Configuration
```python
TRAINING_CONFIG = {
    'batch_size': 32,
    'num_epochs': 200,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'early_stopping_patience': 20,
    'regime_loss_ramp': {
        'ramp_end_epoch': 20,
        'initial_weight': 0.0,
        'final_weight': 0.2
    }
}
```

## Model Components

### 1. HMM Regime Detector
- Fits a Gaussian HMM on return-volatility observations
- Detects 3 market regimes
- Provides regime state and probability distributions

### 2. Dynamic Graph Constructor
- Computes rolling correlation matrices
- Applies regime-specific weighting
- Creates sparse adjacency with minimum correlation threshold
- Falls back to top-K neighbors if no edges meet threshold

### 3. Multi-Head Graph Attention Layer
- Implements graph attention with learnable weights
- Supports 8 parallel attention heads
- Aggregates neighborhood features with learned importance weights

### 4. Regime-Aware GRU
- Standard GRU for temporal encoding
- Regime embedding concatenated with hidden state
- Learned sigmoid gating to modulate hidden state contributions
- 2-layer stacking

### 5. MLP Prediction Head
- 3 fully connected layers with ReLU activations
- Dropout for regularization
- Single output for return magnitude prediction

## Training Details

### Loss Function
```
Total Loss = MSE(prediction, target) + weight * CrossEntropy(regime_logits, regime_label)

where weight linearly increases from 0 to 0.2 over first 20 epochs
```

### Optimizer
- **Adam** with default betas (0.9, 0.999)
- Learning rate: 0.001
- Weight decay: 1e-5
- Gradient clipping: 1.0

### Learning Rate Scheduler
- Cosine annealing with T_max=100 epochs
- Minimum LR: 1e-6

## Data

### Supported Datasets

**NIFTY 50** (50 largest Indian stocks)
- Trading days: 2,080 (2016-2023)
- Total observations: 104,000
- Graph density: 2.3%

**BSE Sensex** (30 largest Indian stocks)
- Trading days: 2,080 (2016-2023)
- Total observations: 62,400
- Graph density: 3.1%

### Features

**OHLCV (5 features)**
- Open, High, Low, Close, Volume

**Technical Indicators (26 features)**
- Returns: 1d, 5d, 20d
- Volatility: 5d, 20d
- Volume: log-volume, 20-day MA
- Momentum: RSI, MACD, MACD Signal, MACD Histogram
- Moving Averages: SMA 5/20/50, EMA 12/26
- Bollinger Bands: Upper, Lower, Middle
- ATR, ADX, CCI

**Total: 31 features per stock**

## Validation Strategy

### Walk-Forward Validation
- 5 independent folds
- Each fold:
  - **Train**: 3 years (756 days)
  - **Validation**: 6 months (126 days)
  - **Test**: 1 year (252 days)
  - Stepped by 1 year to avoid overlap

### Evaluation Metrics

**Return Prediction (Regression)**
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)

**Direction Prediction (Classification)**
- Directional Accuracy
- Precision / Recall / F1 for up-direction

**Portfolio Performance**
- Sharpe Ratio (annualized)
- Sortino Ratio (downside risk)
- Maximum Drawdown
- Compound Annual Growth Rate (CAGR)
- Calmar Ratio

**Statistical Significance**
- Binomial test (DA > 50%?)
- Diebold-Mariano test (vs. GRU baseline)

## Backtesting

### Long-Short Strategy
- Long positions when prediction > 0
- Short positions when prediction < 0
- Equal-weight allocation
- Daily rebalancing

### Transaction Costs
- Base: 0.00% (gross returns)
- Realistic: 0.05% per trade
- Conservative: 0.10% per trade

### Results
```
Transaction Cost  |  CAGR  |  Sharpe  |  Max DD
    0.00%        |  8.40% |  0.168   | 14.2%
    0.05%        |  5.70% |  0.113   | 15.8%
    0.10%        |  3.10% |  0.074   | 17.3%
```

## Cross-Market Generalization

**Transfer Learning Test:**
- Train on NIFTY 50
- Test on BSE Sensex (different index composition)
- No retraining required

**Results:**
- Directional Accuracy drops only 0.5 percentage points (52.8% → 52.3%)
- Sharpe Ratio: 0.141 (vs 0.168 on NIFTY 50)
- Suggests model captures genuine structural patterns

## Comparison with Baselines

| Model | RMSE | DA (%) | Sharpe |
|-------|------|--------|--------|
| ARIMA-GARCH | 0.0268 | 48.20 | -0.091 |
| VAR | 0.0261 | 48.70 | -0.074 |
| Random Forest | 0.0252 | 49.10 | -0.051 |
| XGBoost | 0.0243 | 48.64 | -0.041 |
| SVR | 0.0255 | 48.90 | -0.062 |
| MLP | 0.0249 | 49.30 | -0.044 |
| LSTM | 0.0251 | 49.21 | -0.032 |
| GRU | 0.0247 | 49.58 | -0.029 |
| TCN | 0.0245 | 49.70 | -0.031 |
| Transformer | 0.0244 | 49.80 | -0.028 |
| Static GCN | 0.0241 | 49.80 | -0.021 |
| Static GAT | 0.0239 | 49.97 | -0.017 |
| Dynamic GCN | 0.0236 | 51.10 | +0.089 |
| **RC-DGAT (Ours)** | **0.0231** | **52.80** | **+0.168** |

## Ablation Study

Impact of each component (on NIFTY 50):

| Component | RMSE | DA Increase |
|-----------|------|-----------|
| GRU Baseline | 0.0247 | 49.58% |
| + Static Graph | 0.0239 | +0.39pp |
| + Dynamic Graph | 0.0235 | +1.45pp |
| + Regime Conditioning | 0.0231 | +1.38pp |
| **Total Improvement** | - | **+3.22pp** |

Each component contributes independently—not redundant.

## Computational Efficiency

| Model | Train/Epoch | Inference |
|-------|-------------|-----------|
| GRU | 1.9s | 1.2ms |
| Static GAT | 4.8s | 3.6ms |
| Dynamic GCN | 6.9s | 5.1ms |
| **RC-DGAT** | **9.4s** | **8.3ms** |

- Training: ~2.1x slower than vanilla GRU
- Inference: 8.3ms per batch (operationally irrelevant for daily strategies)
- Full training: <4 hours on NVIDIA RTX 3080 GPU with FP16 precision

## Economic Insights

### Regime Detection
The HMM automatically identified:
- **Feb-May 2020**: High-Volatility state (COVID-19 crash)
- **Sep 2018**: Bear state (NBFC liquidity crisis)
- **2022-2023**: Bear/High-Vol state (inflation environment)

All detected WITHOUT manual labeling — emerged from return-volatility patterns.

### Model Advantage
- Gap between RC-DGAT and baselines **widens during drawdowns**
- Largest advantage during 2020 COVID crash
- Largest advantage during 2022 sell-off
- Model correctly **tightens graph and raises regime influence** during stress periods

## Future Work

1. **Feature Enrichment**
   - Fundamental data (P/E, Book Value)
   - Earnings transcripts sentiment
   - Macroeconomic indicators

2. **Richer Graph Structure**
   - Supply-chain networks
   - Institutional ownership graphs
   - Granger causality edges

3. **Online Regime Updating**
   - Real-time HMM updates with forgetting factor
   - Faster response to regime transitions

4. **Architectural Improvements**
   - Transformer temporal encoders
   - End-to-end RL optimization of risk-adjusted returns
   - Heterogeneous graphs (different node/edge types)

5. **Broader Validation**
   - Other asset classes (bonds, commodities)
   - Different time horizons
   - Global equity markets

## References

The full academic paper cites:
- Hamilton (1989): HMM for economic analysis
- Velickovic et al. (2017): Graph Attention Networks
- Fischer & Krauss (2018): Deep learning for market prediction
- Multiple papers on graph neural networks, temporal forecasting, and financial time series

See references section in original paper for complete citations.

## License

MIT License

## Contact

For questions or collaboration opportunities, please open an issue on the GitHub repository.

## Disclaimer

This implementation is for educational and research purposes. Past performance does not guarantee future results. Use at your own risk in live trading.
