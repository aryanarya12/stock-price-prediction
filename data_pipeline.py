"""
Data loading, preprocessing, and training pipeline for RC-DGAT
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from config import (
    DATASET_CONFIG, TRAINING_CONFIG, COMPUTE_CONFIG, 
    TECHNICAL_FEATURES, OHLCV_FEATURES
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataset(Dataset):
    """
    PyTorch Dataset for multi-stock data.
    """
    
    def __init__(self, price_data, returns_data, volatility_data, 
                 labels=None, sequence_length=20):
        """
        Args:
            price_data: (T, num_stocks, num_features) normalized prices
            returns_data: (T, num_stocks) daily returns
            volatility_data: (T, num_stocks) rolling volatilities
            labels: (T, num_stocks) target returns (next day)
            sequence_length: Lookback window
        """
        self.price_data = torch.from_numpy(price_data).float()
        self.returns_data = torch.from_numpy(returns_data).float()
        self.volatility_data = torch.from_numpy(volatility_data).float()
        self.labels = torch.from_numpy(labels).float() if labels is not None else None
        self.seq_len = sequence_length
        
    def __len__(self):
        return len(self.price_data) - self.seq_len
    
    def __getitem__(self, idx):
        price_seq = self.price_data[idx:idx + self.seq_len]
        returns_seq = self.returns_data[idx:idx + self.seq_len]
        volatility_seq = self.volatility_data[idx:idx + self.seq_len]
        
        if self.labels is not None:
            label = self.labels[idx + self.seq_len]
        else:
            label = None
        
        return {
            'price': price_seq,
            'returns': returns_seq,
            'volatility': volatility_seq,
            'label': label
        }


class DataPreprocessor:
    """
    Handles data loading, feature engineering, and preprocessing.
    """
    
    def __init__(self, ticker_list, start_date, end_date):
        """
        Args:
            ticker_list: List of stock tickers (e.g., ['RELIANCE.NS', 'TCS.NS', ...])
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        self.ticker_list = ticker_list
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.scaler = StandardScaler()
        
    def download_data(self):
        """Download OHLCV data from Yahoo Finance"""
        logger.info(f"Downloading data for {len(self.ticker_list)} stocks...")
        
        data = {}
        for ticker in self.ticker_list:
            try:
                df = yf.download(ticker, start=self.start_date, end=self.end_date, 
                               progress=False)
                if len(df) > 0:
                    data[ticker] = df
                    logger.info(f"Downloaded {ticker}: {len(df)} days")
            except Exception as e:
                logger.warning(f"Failed to download {ticker}: {e}")
        
        self.data = data
        return self
    
    def align_dates(self):
        """Align all stocks to common trading dates"""
        if not self.data:
            raise ValueError("No data loaded. Call download_data() first.")
        
        # Get common dates
        all_dates = [pd.to_datetime(df.index) for df in self.data.values()]
        common_dates = all_dates[0]
        for dates in all_dates[1:]:
            common_dates = common_dates.intersection(dates)
        
        # Align all dataframes
        aligned_data = {}
        for ticker, df in self.data.items():
            aligned_data[ticker] = df.loc[common_dates]
        
        self.data = aligned_data
        logger.info(f"Aligned to {len(common_dates)} common trading dates")
        return self
    
    def compute_technical_features(self, df):
        """Compute technical indicators"""
        features = pd.DataFrame(index=df.index)
        
        # Ensure we have clean data
        df = df.copy()
        df = df.fillna(df.shift())
        df = df.fillna(df.bfill())
        if df.isna().sum().sum() > 0:
            df = df.fillna(df.mean())
        
        # Returns
        features['returns_1d'] = df['Close'].pct_change(1)
        features['returns_5d'] = df['Close'].pct_change(5)
        features['returns_20d'] = df['Close'].pct_change(20)
        
        # Volatility
        features['volatility_5d'] = features['returns_1d'].rolling(5).std()
        features['volatility_20d'] = features['returns_1d'].rolling(20).std()
        
        # Volume
        features['log_volume'] = np.log1p(df['Volume'].fillna(0))
        features['log_volume_ma_20'] = features['log_volume'].rolling(20).mean().fillna(0)
        
        # Intraday
        features['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
        features['close_open_ratio'] = (df['Close'] - df['Open']) / df['Open']
        
        # Momentum
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Moving averages
        features['sma_5'] = df['Close'].rolling(5).mean()
        features['sma_20'] = df['Close'].rolling(20).mean()
        features['sma_50'] = df['Close'].rolling(50).mean()
        
        features['ema_12'] = df['Close'].ewm(span=12).mean()
        features['ema_26'] = df['Close'].ewm(span=26).mean()
        
        # Bollinger Bands
        bb_mid = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        features['bb_upper'] = bb_mid + (bb_std * 2)
        features['bb_lower'] = bb_mid - (bb_std * 2)
        features['bb_middle'] = bb_mid
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        features['atr_14'] = true_range.rolling(14).mean()
        
        # ADX - Simplified calculation
        try:
            plus_dm = df['High'].diff()
            minus_dm = -df['Low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            atr_val = features['atr_14'].copy()
            atr_val[atr_val == 0] = 1  # Avoid division by zero
            
            plus_di = (100 * plus_dm.rolling(14).mean() / atr_val).astype(float)
            minus_di = (100 * minus_dm.rolling(14).mean() / atr_val).astype(float)
            
            di_diff = (plus_di - minus_di).abs()
            di_sum = plus_di + minus_di
            di_sum[di_sum == 0] = 1  # Avoid division by zero
            
            dx = (100 * di_diff / di_sum).astype(float)
            features['adx_14'] = dx.rolling(14).mean().astype(float)
        except Exception as e:
            features['adx_14'] = np.nan
        
        # CCI - With error handling
        try:
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            cci_ma = tp.rolling(20).mean()
            cci_mad = tp.rolling(20).apply(lambda x: (x - x.mean()).abs().mean(), raw=False)
            cci_mad[cci_mad == 0] = 1  # Avoid division by zero
            features['cci_20'] = (tp - cci_ma) / (0.015 * cci_mad)
        except Exception as e:
            features['cci_20'] = np.nan
        
        # Include OHLCV as features
        features['open'] = df['Open']
        features['high'] = df['High']
        features['low'] = df['Low']
        features['close'] = df['Close']
        features['volume'] = df['Volume']
        
        # Fill any remaining NaN values
        features = features.ffill().bfill()
        if features.isna().sum().sum() > 0:
            features = features.fillna(0)
        
        # Ensure all values are numeric and finite
        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
            features[col] = features[col].replace([np.inf, -np.inf], 0)
        
        return features
    
    def prepare_arrays(self):
        """Convert preprocessed data to numpy arrays"""
        if not self.data:
            raise ValueError("No aligned data. Call download_data() and align_dates() first.")
        
        # Compute all features
        all_features = {}
        for ticker, df in self.data.items():
            try:
                features = self.compute_technical_features(df)
                all_features[ticker] = features
            except Exception as e:
                logger.warning(f"Error computing features for {ticker}: {e}")
                continue
        
        # Remove tickers that failed
        available_tickers = list(all_features.keys())
        if len(available_tickers) < 10:
            raise ValueError(f"Not enough valid stocks. Only {len(available_tickers)} out of {len(self.ticker_list)}")
        
        logger.info(f"Using {len(available_tickers)} stocks out of {len(self.ticker_list)}")
        
        # Convert to arrays (T, num_stocks, num_features)
        dates = self.data[available_tickers[0]].index
        T = len(dates)
        num_stocks = len(available_tickers)
        num_features = 31  # As per config
        
        price_array = np.zeros((T, num_stocks, num_features))
        returns_array = np.zeros((T, num_stocks))
        volatility_array = np.zeros((T, num_stocks))
        
        feature_columns = OHLCV_FEATURES + TECHNICAL_FEATURES
        
        for stock_idx, ticker in enumerate(available_tickers):
            features_df = all_features[ticker]
            
            # Price features
            for feat_idx, feat_name in enumerate(feature_columns[:num_features]):
                if feat_name in features_df.columns:
                    try:
                        price_array[:, stock_idx, feat_idx] = features_df[feat_name].fillna(0).values
                    except Exception as e:
                        logger.warning(f"Could not assign {feat_name} for {ticker}")
                        price_array[:, stock_idx, feat_idx] = 0
            
            # Returns and volatility
            try:
                returns_array[:, stock_idx] = features_df['returns_1d'].fillna(0).values
                volatility_array[:, stock_idx] = features_df['volatility_20d'].fillna(0).values
            except Exception as e:
                logger.warning(f"Could not assign returns/volatility for {ticker}")
                returns_array[:, stock_idx] = 0
                volatility_array[:, stock_idx] = 0.01
        
        # Normalize
        price_array = self.scaler.fit_transform(price_array.reshape(-1, num_features)).reshape(T, num_stocks, num_features)
        
        logger.info(f"Prepared arrays: price {price_array.shape}, returns {returns_array.shape}")
        
        return price_array, returns_array, volatility_array
    
    def create_splits(self, price_array, returns_array, volatility_array):
        """Create walk-forward training/validation/test splits"""
        
        T = len(price_array)
        num_stocks = price_array.shape[1]
        
        splits = []
        
        train_days = int(TRAINING_CONFIG['train_window_years'] * 252)  # ~252 trading days/year
        val_days = int(TRAINING_CONFIG['val_window_months'] / 12 * 252)
        test_days = int(TRAINING_CONFIG['test_window_years'] * 252)
        step_days = int(TRAINING_CONFIG['step_size_months'] / 12 * 252)
        
        logger.info(f"Creating splits: train={train_days} days, val={val_days} days, test={test_days} days")
        
        for i in range(TRAINING_CONFIG['walk_forward_splits']):
            train_end = train_days + i * step_days
            val_end = train_end + val_days
            test_end = min(val_end + test_days, T)
            
            train_start = max(0, train_end - train_days)
            
            if test_end > val_end and train_end > 0 and test_end - val_end >= 50:  # Minimum test set
                train_slice = slice(train_start, train_end)
                val_slice = slice(train_end, val_end)
                test_slice = slice(val_end, test_end)
                
                try:
                    splits.append({
                        'train': (price_array[train_slice], returns_array[train_slice], 
                                 volatility_array[train_slice]),
                        'val': (price_array[val_slice], returns_array[val_slice],
                               volatility_array[val_slice]),
                        'test': (price_array[test_slice], returns_array[test_slice],
                                volatility_array[test_slice])
                    })
                    
                    logger.info(f"Split {i}: train[{train_start}:{train_end}], "
                              f"val[{train_end}:{val_end}], test[{val_end}:{test_end}]")
                except Exception as e:
                    logger.warning(f"Could not create split {i}: {e}")
        
        if len(splits) == 0:
            raise ValueError("Could not create any valid splits. Dataset may be too small.")
        
        logger.info(f"Created {len(splits)} walk-forward splits")
        return splits


class TrainingPipeline:
    """
    Main training loop for RC-DGAT.
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
    def train_epoch(self, train_loader, optimizer, epoch, total_epochs):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_return_loss = 0
        total_regime_loss = 0
        
        criterion_return = nn.MSELoss()
        criterion_regime = nn.CrossEntropyLoss()
        
        for batch_idx, batch in enumerate(train_loader):
            price_data = batch['price'].to(self.device)
            returns = batch['returns'].to(self.device)
            volatility = batch['volatility'].to(self.device)
            label = batch['label'].to(self.device) if batch['label'] is not None else None
            
            # Forward pass
            predictions, regime_logits, regime_states = self.model(
                price_data, returns, volatility
            )
            
            # Loss computation
            return_loss = criterion_return(predictions, label.unsqueeze(-1))
            
            # Ramp up regime loss over epochs
            regime_loss_weight = self._get_regime_loss_weight(epoch, total_epochs)
            regime_loss = criterion_regime(regime_logits, regime_states) * regime_loss_weight
            
            total_loss_batch = return_loss + regime_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss_batch.backward()
            
            # Gradient clipping
            if TRAINING_CONFIG.get('gradient_clip_value'):
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(),
                    TRAINING_CONFIG['gradient_clip_value']
                )
            
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_return_loss += return_loss.item()
            total_regime_loss += regime_loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Epoch {epoch}/{total_epochs}, Batch {batch_idx + 1}, "
                          f"Loss: {total_loss_batch.item():.6f}")
        
        avg_loss = total_loss / len(train_loader)
        avg_return_loss = total_return_loss / len(train_loader)
        avg_regime_loss = total_regime_loss / len(train_loader)
        
        return {
            'total_loss': avg_loss,
            'return_loss': avg_return_loss,
            'regime_loss': avg_regime_loss
        }
    
    @staticmethod
    def _get_regime_loss_weight(epoch, total_epochs):
        """Compute regime loss weight with ramp-up"""
        ramp_config = TRAINING_CONFIG['regime_loss_ramp']
        ramp_start = ramp_config['ramp_start_epoch']
        ramp_end = ramp_config['ramp_end_epoch']
        initial_w = ramp_config['initial_weight']
        final_w = ramp_config['final_weight']
        
        if epoch < ramp_start:
            return initial_w
        elif epoch < ramp_end:
            progress = (epoch - ramp_start) / (ramp_end - ramp_start)
            return initial_w + (final_w - initial_w) * progress
        else:
            return final_w
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for batch in val_loader:
                price_data = batch['price'].to(self.device)
                returns = batch['returns'].to(self.device)
                volatility = batch['volatility'].to(self.device)
                label = batch['label'].to(self.device)
                
                predictions, _, _ = self.model(price_data, returns, volatility)
                
                loss = criterion(predictions, label.unsqueeze(-1))
                total_loss += loss.item()
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(label.cpu().numpy())
        
        predictions_array = np.concatenate(all_predictions)  # (batch*seq, num_stocks, 1)
        labels_array = np.concatenate(all_labels)  # (batch*seq, num_stocks)
        
        # Squeeze predictions to match labels shape
        predictions_array = predictions_array.squeeze(-1)  # (batch*seq, num_stocks)
        
        # Compute directional accuracy
        pred_direction = np.sign(predictions_array)
        label_direction = np.sign(labels_array)
        directional_accuracy = np.mean(pred_direction == label_direction) * 100
        
        avg_loss = total_loss / len(val_loader)
        
        return {
            'loss': avg_loss,
            'directional_accuracy': directional_accuracy
        }
    
    def test(self, test_loader):
        """Test the model"""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                price_data = batch['price'].to(self.device)
                returns = batch['returns'].to(self.device)
                volatility = batch['volatility'].to(self.device)
                label = batch['label'].to(self.device)
                
                predictions, _, _ = self.model(price_data, returns, volatility)
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(label.cpu().numpy())
        
        predictions_array = np.concatenate(all_predictions)  # (batch*seq, num_stocks, 1)
        labels_array = np.concatenate(all_labels)  # (batch*seq, num_stocks)
        
        # Squeeze predictions to match labels shape
        predictions_array = predictions_array.squeeze(-1)  # (batch*seq, num_stocks)
        
        # Check for NaN values
        if np.isnan(predictions_array).any():
            # Return dummy metrics if NaN present
            return {
                'rmse': np.nan,
                'mae': np.nan,
                'directional_accuracy': 50.0,
                'predictions': predictions_array,
                'labels': labels_array
            }
        
        # Compute metrics
        rmse = np.sqrt(np.mean((predictions_array - labels_array) ** 2))
        mae = np.mean(np.abs(predictions_array - labels_array))
        
        pred_direction = np.sign(predictions_array)
        label_direction = np.sign(labels_array)
        directional_accuracy = np.mean(pred_direction == label_direction) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'directional_accuracy': directional_accuracy,
            'predictions': predictions_array,
            'labels': labels_array
        }


if __name__ == "__main__":
    # Example usage
    logger.info("Starting data pipeline...")
