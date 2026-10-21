"""
Fast training script - optimized for CPU execution
Trains RC-DGAT on reduced dataset for quick validation before full training
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
import pandas as pd
import yfinance as yf

from config import (
    DATASET_CONFIG, TRAINING_CONFIG, COMPUTE_CONFIG, IO_CONFIG,
    EXPECTED_RESULTS, HMM_CONFIG, GAT_CONFIG, GRU_CONFIG
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


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if COMPUTE_CONFIG['deterministic']:
        torch.backends.cudnn.deterministic = True


def fast_train():
    """Fast training on 15 top stocks with optimized settings"""
    
    logger.info("="*70)
    logger.info("RC-DGAT FAST TRAINING (15 stocks, optimized for CPU)")
    logger.info("="*70)
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Use only top 15 liquid stocks for speed
    fast_tickers = [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'WIPRO.NS', 'MARUTI.NS', 'BAJAJFINSV.NS', 'AXISBANK.NS',
        'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 'ASIANPAINT.NS', 'CIPLA.NS'
    ]
    
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Download data")
    logger.info("="*70)
    
    preprocessor = DataPreprocessor(
        ticker_list=fast_tickers,
        start_date='2019-01-01',  # Start from 2019 for faster processing
        end_date='2023-12-31'
    )
    
    # Download and align
    preprocessor.download_data()
    preprocessor.align_dates()
    
    # Prepare arrays
    price_array, returns_array, volatility_array = preprocessor.prepare_arrays()
    logger.info(f"✓ Prepared arrays: price {price_array.shape}, returns {returns_array.shape}")
    
    # Create just 2 splits for fast validation
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Create walk-forward splits (2 splits for fast validation)")
    logger.info("="*70)
    
    T = len(price_array)
    
    # Split 1: train[0:756], val[756:882], test[882:1134]
    # Split 2: train[252:1008], val[1008:1134], test[1134:1386]
    
    splits = []
    
    # Split 1
    train1_price = price_array[0:756]
    train1_returns = returns_array[0:756]
    train1_vol = volatility_array[0:756]
    
    val1_price = price_array[756:882]
    val1_returns = returns_array[756:882]
    val1_vol = volatility_array[756:882]
    
    test1_price = price_array[882:min(1134, T)]
    test1_returns = returns_array[882:min(1134, T)]
    test1_vol = volatility_array[882:min(1134, T)]
    
    splits.append({
        'train': (train1_price, train1_returns, train1_vol),
        'val': (val1_price, val1_returns, val1_vol),
        'test': (test1_price, test1_returns, test1_vol)
    })
    
    # Split 2
    if T > 1008:
        train2_price = price_array[252:1008]
        train2_returns = returns_array[252:1008]
        train2_vol = volatility_array[252:1008]
        
        val2_price = price_array[1008:min(1134, T)]
        val2_returns = returns_array[1008:min(1134, T)]
        val2_vol = volatility_array[1008:min(1134, T)]
        
        test2_price = price_array[min(1134, T):min(1386, T)]
        test2_returns = returns_array[min(1134, T):min(1386, T)]
        test2_vol = volatility_array[min(1134, T):min(1386, T)]
        
        if len(test2_price) > 0:
            splits.append({
                'train': (train2_price, train2_returns, train2_vol),
                'val': (val2_price, val2_returns, val2_vol),
                'test': (test2_price, test2_returns, test2_vol)
            })
    
    logger.info(f"✓ Created {len(splits)} walk-forward splits")
    
    # Training
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Train RC-DGAT models")
    logger.info("="*70)
    
    all_results = []
    
    for split_idx, split_data in enumerate(splits):
        logger.info(f"\n--- Split {split_idx + 1}/{len(splits)} ---")
        
        try:
            # Create model
            model = RCDGAT()
            model.to(device)
            
            # Fit HMM
            train_returns, train_vol = split_data['train'][1], split_data['train'][2]
            model.hmm_detector.fit(train_returns, train_vol)
            logger.info("✓ HMM fitted")
            
            # Create datasets
            train_price, train_returns, train_vol = split_data['train']
            val_price, val_returns, val_vol = split_data['val']
            test_price, test_returns, test_vol = split_data['test']
            
            # Labels (next-day returns)
            train_labels = np.diff(train_returns, axis=0, prepend=train_returns[0:1])
            val_labels = np.diff(val_returns, axis=0, prepend=val_returns[0:1])
            test_labels = np.diff(test_returns, axis=0, prepend=test_returns[0:1])
            
            # Datasets
            train_dataset = StockDataset(train_price, train_returns, train_vol, 
                                        train_labels, sequence_length=20)
            val_dataset = StockDataset(val_price, val_returns, val_vol, 
                                      val_labels, sequence_length=20)
            test_dataset = StockDataset(test_price, test_returns, test_vol, 
                                       test_labels, sequence_length=20)
            
            # Dataloaders
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            
            logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
            
            # Training setup
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            pipeline = TrainingPipeline(model, device)
            
            best_val_acc = 0
            patience_counter = 0
            num_epochs = 50  # Reduced for speed
            
            # Training loop
            for epoch in range(num_epochs):
                train_metrics = pipeline.train_epoch(train_loader, optimizer, epoch, num_epochs)
                val_metrics = pipeline.validate(val_loader)
                
                logger.info(f"Epoch {epoch+1}/{num_epochs} | "
                           f"Train Loss: {train_metrics['total_loss']:.6f} | "
                           f"Val Loss: {val_metrics['loss']:.6f} | "
                           f"Val Acc: {val_metrics['directional_accuracy']:.2f}%")
                
                # Early stopping
                if val_metrics['directional_accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['directional_accuracy']
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Test
            test_metrics = pipeline.test(test_loader)
            logger.info(f"Test RMSE: {test_metrics['rmse']:.6f}")
            logger.info(f"Test Directional Accuracy: {test_metrics['directional_accuracy']:.2f}%")
            
            all_results.append({
                'split': split_idx + 1,
                'rmse': test_metrics['rmse'],
                'directional_accuracy': test_metrics['directional_accuracy'],
                'best_val_acc': best_val_acc
            })
            
        except Exception as e:
            logger.error(f"Error in split {split_idx}: {e}", exc_info=True)
            continue
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("FINAL RESULTS")
    logger.info("="*70)
    
    if all_results:
        rmses = [r['rmse'] for r in all_results]
        accs = [r['directional_accuracy'] for r in all_results]
        
        logger.info(f"Average RMSE: {np.mean(rmses):.6f} ± {np.std(rmses):.6f}")
        logger.info(f"Average Directional Accuracy: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
        
        # Save results
        results_file = Path(IO_CONFIG['results_dir']) / f'fast_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'n_stocks': len(fast_tickers),
                'n_splits': len(splits),
                'results': all_results,
                'summary': {
                    'mean_rmse': float(np.mean(rmses)),
                    'std_rmse': float(np.std(rmses)),
                    'mean_accuracy': float(np.mean(accs)),
                    'std_accuracy': float(np.std(accs))
                }
            }, f, indent=2)
        logger.info(f"✓ Results saved to {results_file}")
        
        logger.info("\n" + "="*70)
        logger.info("✅ FAST TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        return True
    else:
        logger.error("❌ No results generated")
        return False


if __name__ == "__main__":
    success = fast_train()
    if not success:
        exit(1)
