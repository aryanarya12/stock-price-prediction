"""
Quick validation script - trains on a subset of stocks for fast testing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path

from config import (
    DATASET_CONFIG, TRAINING_CONFIG, COMPUTE_CONFIG, IO_CONFIG,
    EXPECTED_RESULTS
)
from model import RCDGAT
from data_pipeline import StockDataset, DataPreprocessor, TrainingPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Setup paths
for path_key in ['data_dir', 'model_dir', 'results_dir', 'logs_dir', 'checkpoint_dir']:
    Path(IO_CONFIG[path_key]).mkdir(parents=True, exist_ok=True)


def quick_validation():
    """
    Quick validation using only 10 stocks and a single short split.
    Useful for checking if the code works before full training.
    """
    
    logger.info("="*60)
    logger.info("Quick Validation Script (10 stocks, 1 split)")
    logger.info("="*60)
    
    # Use only 10 stocks for quick validation
    quick_tickers = [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'WIPRO.NS', 'MARUTI.NS', 'BAJAJFINSV.NS', 'AXISBANK.NS'
    ]
    
    # Download data
    logger.info("\nDownloading data for 10 stocks...")
    preprocessor = DataPreprocessor(
        ticker_list=quick_tickers,
        start_date='2020-01-01',  # Shorter date range for quick testing
        end_date='2023-12-31'
    )
    
    try:
        preprocessor.download_data()
        preprocessor.align_dates()
        
        # Prepare arrays
        price_array, returns_array, volatility_array = preprocessor.prepare_arrays()
        logger.info(f"Prepared data: {price_array.shape}")
        
        # Create just 1 split for quick testing
        # Use last 750 days: 500 train + 125 val + 125 test
        T = len(price_array)
        train_end = T - 250
        val_end = train_end + 125
        test_end = T
        
        split_data = {
            'train': (price_array[:train_end], returns_array[:train_end], 
                     volatility_array[:train_end]),
            'val': (price_array[train_end:val_end], returns_array[train_end:val_end],
                   volatility_array[train_end:val_end]),
            'test': (price_array[val_end:test_end], returns_array[val_end:test_end],
                    volatility_array[val_end:test_end])
        }
        
        # Create dataloaders
        train_price, train_returns, train_vol = split_data['train']
        val_price, val_returns, val_vol = split_data['val']
        test_price, test_returns, test_vol = split_data['test']
        
        train_labels = np.diff(train_returns, axis=0, prepend=train_returns[0:1])
        val_labels = np.diff(val_returns, axis=0, prepend=val_returns[0:1])
        test_labels = np.diff(test_returns, axis=0, prepend=test_returns[0:1])
        
        train_dataset = StockDataset(train_price, train_returns, train_vol, 
                                    train_labels, sequence_length=20)
        val_dataset = StockDataset(val_price, val_returns, val_vol, 
                                  val_labels, sequence_length=20)
        test_dataset = StockDataset(test_price, test_returns, test_vol, 
                                   test_labels, sequence_length=20)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RCDGAT()
        model.to(device)
        
        # Fit HMM before training
        logger.info("Fitting HMM detector...")
        train_returns_np = train_returns
        train_vol_np = train_vol
        model.hmm_detector.fit(train_returns_np, train_vol_np)
        logger.info("HMM fitted successfully")
        
        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        pipeline = TrainingPipeline(model, device)
        
        logger.info("\n" + "="*60)
        logger.info("Starting Quick Training (20 epochs)")
        logger.info("="*60)
        
        for epoch in range(20):
            train_metrics = pipeline.train_epoch(train_loader, optimizer, epoch, 20)
            val_metrics = pipeline.validate(val_loader)
            
            logger.info(f"Epoch {epoch+1}/20 | "
                       f"Train Loss: {train_metrics['total_loss']:.6f} | "
                       f"Val Loss: {val_metrics['loss']:.6f} | "
                       f"Val Acc: {val_metrics['directional_accuracy']:.2f}%")
        
        # Test
        logger.info("\n" + "="*60)
        logger.info("Testing")
        logger.info("="*60)
        
        test_metrics = pipeline.test(test_loader)
        logger.info(f"Test RMSE: {test_metrics['rmse']:.6f}")
        logger.info(f"Test Directional Accuracy: {test_metrics['directional_accuracy']:.2f}%")
        
        logger.info("\n" + "="*60)
        logger.info("✅ Quick Validation Successful!")
        logger.info("Code is working correctly. Ready for full training.")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error during validation: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = quick_validation()
    if success:
        logger.info("\nNext step: Run full training with: python train.py")
    else:
        logger.error("Quick validation failed. Fix errors above.")
