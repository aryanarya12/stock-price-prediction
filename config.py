"""
Configuration and hyperparameters for RC-DGAT model
Based on: RC-DGAT: Regime-Conditioned Dynamic Graph Attention for Stock Return Forecasting
"""

# ===================== DATASET CONFIGURATION =====================
DATASET_CONFIG = {
    'nifty50': {
        'num_stocks': 50,
        'trading_days': 2080,
        'total_observations': 104000,
        'features_per_stock': 31,
        'avg_daily_return': 0.12,
        'return_volatility': 2.47,
        'correlation_range': [-0.65, 0.89],
        'avg_graph_density': 2.3
    },
    'bse_sensex': {
        'num_stocks': 30,
        'trading_days': 2080,
        'total_observations': 62400,
        'features_per_stock': 31,
        'avg_daily_return': 0.13,
        'return_volatility': 2.51,
        'correlation_range': [-0.61, 0.91],
        'avg_graph_density': 3.1
    }
}

# ===================== INPUT FEATURES =====================
OHLCV_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']

TECHNICAL_FEATURES = [
    'returns_1d', 'returns_5d', 'returns_20d',  # Returns at different horizons
    'volatility_5d', 'volatility_20d',  # Historical volatility
    'log_volume', 'log_volume_ma_20',  # Volume features
    'high_low_ratio', 'close_open_ratio',  # Intraday features
    'rsi_14', 'macd', 'macd_signal', 'macd_hist',  # Momentum indicators
    'sma_5', 'sma_20', 'sma_50',  # Moving averages
    'ema_12', 'ema_26',  # Exponential moving averages
    'bb_upper', 'bb_lower', 'bb_middle',  # Bollinger bands
    'atr_14',  # Average true range
    'adx_14',  # Average directional index
    'cci_20'  # Commodity channel index
]

TOTAL_FEATURES = len(OHLCV_FEATURES) + len(TECHNICAL_FEATURES)  # 31 features total

# ===================== HMM CONFIGURATION =====================
HMM_CONFIG = {
    'num_states': 3,  # Bull, Bear, High-Volatility
    'state_names': ['Bull', 'Bear', 'HighVolatility'],
    'state_colors': ['green', 'red', 'yellow'],  # For visualization
    'n_iter': 1000,  # EM iterations
    'random_state': 42,
    'covariance_type': 'full',  # full, tied, diag, spherical
    'observation_window': 252,  # Rolling window for HMM fitting (1 trading year)
    'update_frequency': 1  # Update HMM state every N days
}

# ===================== GRAPH CONSTRUCTION =====================
GRAPH_CONFIG = {
    'method': 'correlation',  # Correlation-based graph
    'correlation_type': 'pearson',  # pearson, spearman, or kendall
    'window_size': 20,  # Days for rolling correlation
    'threshold': 0.3,  # Minimum correlation to include edge
    'self_loop': False,  # Include self-loops
    'direction': 'undirected',  # Graph is undirected
    'dynamic_update': True,  # Update graph at each timestep based on regime
    'regime_weight_dict': {
        'Bull': {'correlation_scale': 1.0, 'density': 'normal'},
        'Bear': {'correlation_scale': 1.2, 'density': 'tight'},  # Tighter in bear markets
        'HighVolatility': {'correlation_scale': 1.3, 'density': 'tight'}  # Tightest in high vol
    }
}

# ===================== REGIME-AWARE COMPONENTS =====================
REGIME_CONFIG = {
    'embedding_dim': 16,  # Regime embedding dimension
    'embedding_initialization': 'he_normal',
    'regime_influence_start_epoch': 0,  # Start influencing from epoch 0
    'regime_loss_weight': 0.2,  # Weight for regime classification loss
    'regime_loss_scale_per_epoch': {
        'epoch_0_to_20': 0.0,  # Linear increase from 0 to 1 over first 20 epochs
        'after_epoch_20': 1.0
    }
}

# ===================== GRAPH ATTENTION NETWORK (GAT) =====================
GAT_CONFIG = {
    'input_dim': 31,  # Features per node
    'hidden_dim': 64,  # Hidden dimension
    'output_dim': 32,  # Output dimension after GAT
    'num_heads': 8,  # Number of attention heads
    'dropout': 0.3,
    'activation': 'elu',
    'use_bias': True,
    'add_self_loops': False,
    'improved': False,  # Use improved attention (GAT v2)
    'concat_heads': True  # Concatenate or average heads
}

# ===================== TEMPORAL ENCODER (GRU) =====================
GRU_CONFIG = {
    'input_dim': 32,  # From GAT output
    'hidden_dim': 64,  # GRU hidden dimension
    'num_layers': 2,
    'dropout': 0.3,
    'bidirectional': False,
    'batch_first': True,
    'regime_gating': True,  # Use regime to modulate GRU gate
    'regime_gate_weight': 0.5  # How much regime affects gates
}

# ===================== PREDICTION HEAD =====================
PREDICTION_HEAD_CONFIG = {
    'input_dim': 64,  # From GRU
    'hidden_dims': [32, 16],  # MLP layers
    'output_dim': 1,  # Predict single return value
    'dropout': [0.2, 0.1],
    'activations': ['relu', 'relu']
}

# ===================== TRAINING CONFIGURATION =====================
TRAINING_CONFIG = {
    # Data split
    'walk_forward_splits': 5,
    'train_window_years': 3,  # 3 years
    'val_window_months': 6,  # 6 months
    'test_window_years': 1,  # 1 year
    'step_size_months': 12,  # Move forward by 1 year each split
    
    # Training hyperparameters
    'batch_size': 32,
    'num_epochs': 200,
    'learning_rate': 0.001,
    'optimizer': 'adam',  # adam, sgd, rmsprop
    'optimizer_config': {
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 1e-5
    },
    
    # Loss functions
    'loss_function': 'mse',  # Primary: MSE for return magnitude
    'regime_loss_function': 'crossentropy',  # Secondary: cross-entropy for regime classification
    'regime_loss_weight': 0.2,  # Balance between primary and regime loss
    'regime_loss_ramp': {
        'ramp_start_epoch': 0,
        'ramp_end_epoch': 20,
        'initial_weight': 0.0,
        'final_weight': 0.2
    },
    
    # Early stopping
    'early_stopping_patience': 20,
    'early_stopping_metric': 'val_directional_accuracy',
    'early_stopping_mode': 'max',
    'early_stopping_min_delta': 0.001,
    
    # Learning rate scheduling
    'use_scheduler': True,
    'scheduler_type': 'cosine_annealing',  # cosine_annealing, exponential_decay, step
    'scheduler_config': {
        'T_max': 100,  # Number of iterations for cosine annealing
        'eta_min': 1e-6
    },
    
    # Gradient clipping
    'gradient_clip_value': 1.0,
    'gradient_clip_norm': None
}

# ===================== EVALUATION METRICS =====================
EVAL_METRICS = {
    'regression': ['rmse', 'mae', 'mape'],
    'classification': ['directional_accuracy', 'precision', 'recall', 'f1'],
    'portfolio': ['sharpe_ratio', 'max_drawdown', 'cagr', 'sortino_ratio'],
    'statistical': ['diebold_mariano_test']
}

# ===================== BACKTESTING CONFIGURATION =====================
BACKTEST_CONFIG = {
    'initial_capital': 1000000,  # $1M starting capital
    'transaction_costs': [0.00, 0.05, 0.10],  # Percentage per trade
    'slippage': 0.01,  # Basis points
    'position_size': 0.05,  # Use 5% of capital per position
    'rebalance_frequency': 'daily',  # Rebalance daily
    'long_short_strategy': True,  # Long-short portfolio
    'min_holding_period': 1  # Minimum days to hold position
}

# ===================== BASELINE MODELS =====================
BASELINE_MODELS = [
    'ARIMA_GARCH',
    'VAR',
    'RandomForest',
    'XGBoost',
    'SVR',
    'MLP',
    'LSTM',
    'GRU',
    'TCN',
    'Transformer',
    'StaticGCN',
    'StaticGAT',
    'DynamicGCN',
    'RCDGAT'  # Our model
]

# ===================== EXPECTED RESULTS (FROM PAPER) =====================
EXPECTED_RESULTS = {
    'rmse': 0.0231,
    'directional_accuracy': 52.80,
    'sharpe_ratio': 0.168,
    'max_drawdown': 14.2,
    'cagr': 8.4
}

# ===================== DEVICE AND COMPUTATION =====================
COMPUTE_CONFIG = {
    'device': 'cuda',  # cuda or cpu
    'mixed_precision': True,  # FP16 mixed precision
    'num_workers': 4,  # DataLoader workers
    'pin_memory': True,
    'benchmark': True,  # torch.backends.cudnn.benchmark
    'deterministic': True,  # For reproducibility
    'seed': 42
}

# ===================== PATHS AND LOGGING =====================
IO_CONFIG = {
    'data_dir': './data/',
    'model_dir': './models/',
    'results_dir': './results/',
    'logs_dir': './logs/',
    'checkpoint_dir': './checkpoints/',
    'log_level': 'INFO',
    'save_frequency': 5  # Save checkpoint every N epochs
}
