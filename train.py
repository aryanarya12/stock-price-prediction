"""
Main training script for RC-DGAT model
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


def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if COMPUTE_CONFIG['deterministic']:
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = COMPUTE_CONFIG['benchmark']


def prepare_data(dataset_type='nifty50'):
    """
    Download, preprocess, and prepare data.
    
    Args:
        dataset_type: 'nifty50' or 'bse_sensex'
    """
    logger.info(f"Preparing {dataset_type} dataset...")
    
    # NIFTY 50 tickers (NSE format with .NS suffix) - CORRECTED
    nifty50_tickers = [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'WIPRO.NS', 'MARUTI.NS', 'BAJAJFINSV.NS', 'AXISBANK.NS',
        'LTTS.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'INDIGO.NS', 'SUNPHARMA.NS',
        'KOTAKBANK.NS', 'TATASTEEL.NS', 'HEROMOTOCO.NS', 'ADANIPORTS.NS', 'POWERGRID.NS',
        'TECHM.NS', 'NTPC.NS', 'COALINDIA.NS', 'TITAN.NS', 'BAJAJ-AUTO.NS',
        'ASIANPAINT.NS', 'CIPLA.NS', 'GAIL.NS', 'ONGC.NS', 'BPCL.NS',
        'IOC.NS', 'TATACONSUM.NS', 'HCLTECH.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS',
        'DRREDDY.NS', 'BIOCON.NS', 'LUPIN.NS', 'EICHERMOT.NS', 'BOSCHLTD.NS',
        'PIDILITIND.NS', 'UPL.NS', 'JSWSTEEL.NS', 'INDUSTOWER.NS', 'HAVELLS.NS',
        'IDFCFIRSTB.NS', 'AMBUJACEM.NS', 'GODREJCP.NS', 'M&MFIN.NS', 'SHRIRAMFIN.NS'
    ]
    
    # BSE Sensex tickers - CORRECTED
    sensex_tickers = [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'WIPRO.NS', 'MARUTI.NS', 'BAJAJFINSV.NS', 'AXISBANK.NS',
        'SBIN.NS', 'BHARTIARTL.NS', 'INDIGO.NS', 'SUNPHARMA.NS', 'KOTAKBANK.NS',
        'TATASTEEL.NS', 'HEROMOTOCO.NS', 'ADANIPORTS.NS', 'POWERGRID.NS', 'TECHM.NS',
        'NTPC.NS', 'COALINDIA.NS', 'TITAN.NS', 'BAJAJ-AUTO.NS', 'ASIANPAINT.NS',
        'CIPLA.NS', 'GAIL.NS', 'ONGC.NS', 'BPCL.NS', 'IOC.NS'
    ]
    
    ticker_list = nifty50_tickers if dataset_type == 'nifty50' else sensex_tickers
    
    # Download data
    preprocessor = DataPreprocessor(
        ticker_list=ticker_list,
        start_date='2018-01-01',
        end_date='2025-12-31'
    )
    
    preprocessor.download_data()
    preprocessor.align_dates()
    
    # Prepare arrays
    price_array, returns_array, volatility_array = preprocessor.prepare_arrays()
    
    # Create splits
    splits = preprocessor.create_splits(price_array, returns_array, volatility_array)
    
    return splits


def create_dataloaders(split_data, batch_size=32, sequence_length=20):
    """Create DataLoaders from split data"""
    
    train_price, train_returns, train_vol = split_data['train']
    val_price, val_returns, val_vol = split_data['val']
    test_price, test_returns, test_vol = split_data['test']
    
    # Compute labels (next day returns)
    train_labels = np.diff(train_returns, axis=0, prepend=train_returns[0:1])
    val_labels = np.diff(val_returns, axis=0, prepend=val_returns[0:1])
    test_labels = np.diff(test_returns, axis=0, prepend=test_returns[0:1])
    
    # Create datasets
    train_dataset = StockDataset(train_price, train_returns, train_vol, 
                                train_labels, sequence_length)
    val_dataset = StockDataset(val_price, val_returns, val_vol, 
                              val_labels, sequence_length)
    test_dataset = StockDataset(test_price, test_returns, test_vol, 
                               test_labels, sequence_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=COMPUTE_CONFIG['num_workers'],
                             pin_memory=COMPUTE_CONFIG['pin_memory'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=COMPUTE_CONFIG['num_workers'],
                           pin_memory=COMPUTE_CONFIG['pin_memory'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=COMPUTE_CONFIG['num_workers'],
                            pin_memory=COMPUTE_CONFIG['pin_memory'])
    
    return train_loader, val_loader, test_loader


def create_optimizer(model, config=TRAINING_CONFIG):
    """Create optimizer"""
    
    optimizer_type = config['optimizer'].lower()
    lr = config['learning_rate']
    
    if optimizer_type == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=lr,
            betas=config['optimizer_config'].get('betas', (0.9, 0.999)),
            eps=config['optimizer_config'].get('eps', 1e-8),
            weight_decay=config['optimizer_config'].get('weight_decay', 0)
        )
    elif optimizer_type == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config['optimizer_config'].get('momentum', 0.9),
            weight_decay=config['optimizer_config'].get('weight_decay', 0)
        )
    elif optimizer_type == 'rmsprop':
        return optim.RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=config['optimizer_config'].get('weight_decay', 0)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(optimizer, config=TRAINING_CONFIG):
    """Create learning rate scheduler"""
    
    if not config.get('use_scheduler'):
        return None
    
    scheduler_type = config['scheduler_type'].lower()
    
    if scheduler_type == 'cosine_annealing':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['scheduler_config'].get('T_max', 100),
            eta_min=config['scheduler_config'].get('eta_min', 0)
        )
    elif scheduler_type == 'exponential_decay':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config['scheduler_config'].get('gamma', 0.99)
        )
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['scheduler_config'].get('step_size', 10),
            gamma=config['scheduler_config'].get('gamma', 0.1)
        )
    else:
        return None


def train_single_split(model, train_loader, val_loader, split_idx, device):
    """Train model on single walk-forward split"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Split {split_idx + 1}/{TRAINING_CONFIG['walk_forward_splits']}")
    logger.info(f"{'='*60}")
    
    # Setup training
    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer)
    pipeline = TrainingPipeline(model, device)
    
    best_val_acc = 0
    patience_counter = 0
    split_results = {
        'train_losses': [],
        'val_losses': [],
        'val_accuracies': [],
        'best_epoch': 0,
        'best_val_acc': 0
    }
    
    num_epochs = TRAINING_CONFIG['num_epochs']
    
    for epoch in range(num_epochs):
        # Training
        train_metrics = pipeline.train_epoch(train_loader, optimizer, epoch, num_epochs)
        
        # Validation
        val_metrics = pipeline.validate(val_loader)
        
        # Scheduler step
        if scheduler:
            scheduler.step()
        
        # Log results
        logger.info(f"Epoch {epoch+1}/{num_epochs} | "
                   f"Train Loss: {train_metrics['total_loss']:.6f} | "
                   f"Val Loss: {val_metrics['loss']:.6f} | "
                   f"Val Acc: {val_metrics['directional_accuracy']:.2f}%")
        
        # Track results
        split_results['train_losses'].append(train_metrics['total_loss'])
        split_results['val_losses'].append(val_metrics['loss'])
        split_results['val_accuracies'].append(val_metrics['directional_accuracy'])
        
        # Early stopping
        if val_metrics['directional_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['directional_accuracy']
            patience_counter = 0
            split_results['best_epoch'] = epoch
            split_results['best_val_acc'] = best_val_acc
            
            # Save checkpoint
            checkpoint_path = Path(IO_CONFIG['checkpoint_dir']) / f'split_{split_idx}_best.pt'
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= TRAINING_CONFIG['early_stopping_patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    return split_results


def train_all_splits(data_splits, device):
    """Train model on all walk-forward splits"""
    
    all_results = []
    
    for split_idx, split_data in enumerate(data_splits):
        try:
            # Create fresh model for each split
            model = RCDGAT()
            model.to(device)
            
            # Fit HMM on training data before training
            train_price, train_returns, train_vol = split_data['train']
            logger.info(f"Fitting HMM on training data ({train_returns.shape[0]} samples)...")
            model.hmm_detector.fit(train_returns, train_vol)
            logger.info("HMM fitted successfully")
            
            # Create dataloaders
            train_loader, val_loader, test_loader = create_dataloaders(
                split_data,
                batch_size=TRAINING_CONFIG['batch_size']
            )
            
            # Train
            split_results = train_single_split(model, train_loader, val_loader, split_idx, device)
            all_results.append(split_results)
            
            # Test
            logger.info(f"\nTesting Split {split_idx + 1}...")
            pipeline = TrainingPipeline(model, device)
            test_metrics = pipeline.test(test_loader)
            logger.info(f"Test RMSE: {test_metrics['rmse']:.6f} | "
                       f"Test Acc: {test_metrics['directional_accuracy']:.2f}%")
        except Exception as e:
            logger.error(f"Error training split {split_idx}: {e}", exc_info=True)
            continue
    
    return all_results


def main():
    """Main training script"""
    
    logger.info("="*60)
    logger.info("RC-DGAT Training Pipeline")
    logger.info("="*60)
    
    # Setup
    set_seed(COMPUTE_CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() and 
                         COMPUTE_CONFIG['device'] == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Prepare data
    logger.info("\nPreparing data...")
    try:
        data_splits = prepare_data('nifty50')
        
        if len(data_splits) == 0:
            logger.error("No valid data splits created!")
            return
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return
    
    # Train model
    logger.info("\nStarting training...")
    try:
        all_results = train_all_splits(data_splits, device)
        
        if len(all_results) == 0:
            logger.error("No training results generated!")
            return
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return
    
    # Aggregate results
    logger.info("\n" + "="*60)
    logger.info("Final Results Summary")
    logger.info("="*60)
    
    final_val_accs = [r['best_val_acc'] for r in all_results if 'best_val_acc' in r]
    
    if len(final_val_accs) > 0:
        avg_val_acc = np.mean(final_val_accs)
        std_val_acc = np.std(final_val_accs)
        
        logger.info(f"Average Validation Directional Accuracy: {avg_val_acc:.2f}% ± {std_val_acc:.2f}%")
        logger.info(f"Expected Result (from paper): {EXPECTED_RESULTS['directional_accuracy']:.2f}%")
        
        # Save results
        results_file = Path(IO_CONFIG['results_dir']) / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'all_splits': all_results,
                'summary': {
                    'avg_val_acc': float(avg_val_acc),
                    'std_val_acc': float(std_val_acc),
                    'expected_acc': EXPECTED_RESULTS['directional_accuracy'],
                    'num_splits': len(all_results)
                }
            }, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    else:
        logger.error("No valid results to summarize!")


if __name__ == "__main__":
    main()
