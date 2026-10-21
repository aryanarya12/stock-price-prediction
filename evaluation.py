"""
Evaluation metrics, backtesting, and visualization utilities for RC-DGAT
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
import logging

logger = logging.getLogger(__name__)


class MetricsComputer:
    """Compute evaluation metrics"""
    
    @staticmethod
    def rmse(y_true, y_pred):
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true, y_pred):
        """Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def mape(y_true, y_pred):
        """Mean Absolute Percentage Error"""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def directional_accuracy(y_true, y_pred):
        """
        Directional Accuracy: Percentage of correct direction predictions.
        """
        pred_direction = np.sign(y_pred)
        true_direction = np.sign(y_true)
        return np.mean(pred_direction == true_direction) * 100
    
    @staticmethod
    def precision(y_true, y_pred):
        """Precision for direction prediction"""
        pred_direction = (y_pred > 0).astype(int)
        true_direction = (y_true > 0).astype(int)
        return precision_score(true_direction, pred_direction, zero_division=0)
    
    @staticmethod
    def recall(y_true, y_pred):
        """Recall for direction prediction"""
        pred_direction = (y_pred > 0).astype(int)
        true_direction = (y_true > 0).astype(int)
        return recall_score(true_direction, pred_direction, zero_division=0)
    
    @staticmethod
    def f1(y_true, y_pred):
        """F1 score for direction prediction"""
        pred_direction = (y_pred > 0).astype(int)
        true_direction = (y_true > 0).astype(int)
        return f1_score(true_direction, pred_direction, zero_division=0)
    
    @staticmethod
    def compute_all_metrics(y_true, y_pred):
        """Compute all regression and classification metrics"""
        return {
            'rmse': MetricsComputer.rmse(y_true, y_pred),
            'mae': MetricsComputer.mae(y_true, y_pred),
            'mape': MetricsComputer.mape(y_true, y_pred),
            'directional_accuracy': MetricsComputer.directional_accuracy(y_true, y_pred),
            'precision': MetricsComputer.precision(y_true, y_pred),
            'recall': MetricsComputer.recall(y_true, y_pred),
            'f1': MetricsComputer.f1(y_true, y_pred)
        }


class PortfolioMetrics:
    """Compute portfolio-level metrics"""
    
    @staticmethod
    def sharpe_ratio(returns, risk_free_rate=0.0):
        """
        Sharpe Ratio: (mean_return - risk_free_rate) / std_dev
        Assumes daily returns.
        """
        excess_returns = returns - risk_free_rate / 252  # Annualize risk-free rate
        if np.std(excess_returns) == 0:
            return 0
        return (np.mean(excess_returns) * 252) / (np.std(excess_returns) * np.sqrt(252))
    
    @staticmethod
    def sortino_ratio(returns, risk_free_rate=0.0, target_return=0.0):
        """Sortino Ratio: Uses downside deviation instead of std dev"""
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return 0
        
        return (np.mean(excess_returns) * 252) / (downside_std * np.sqrt(252))
    
    @staticmethod
    def max_drawdown(returns):
        """Maximum Drawdown: Largest peak-to-trough decline"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown) * 100
    
    @staticmethod
    def cagr(returns, periods_per_year=252):
        """Compound Annual Growth Rate"""
        cumulative = np.prod(1 + returns)
        num_years = len(returns) / periods_per_year
        if num_years == 0:
            return 0
        return (cumulative ** (1 / num_years) - 1) * 100
    
    @staticmethod
    def calmar_ratio(returns, periods_per_year=252):
        """Calmar Ratio: CAGR / Max Drawdown"""
        cagr = PortfolioMetrics.cagr(returns, periods_per_year)
        max_dd = abs(PortfolioMetrics.max_drawdown(returns))
        if max_dd == 0:
            return 0
        return cagr / max_dd
    
    @staticmethod
    def compute_all(returns):
        """Compute all portfolio metrics"""
        return {
            'sharpe_ratio': PortfolioMetrics.sharpe_ratio(returns),
            'sortino_ratio': PortfolioMetrics.sortino_ratio(returns),
            'max_drawdown': PortfolioMetrics.max_drawdown(returns),
            'cagr': PortfolioMetrics.cagr(returns),
            'calmar_ratio': PortfolioMetrics.calmar_ratio(returns)
        }


class StatisticalTests:
    """Statistical tests for model comparison"""
    
    @staticmethod
    def diebold_mariano_test(actual, pred1, pred2, horizon=1):
        """
        Diebold-Mariano test for forecast comparison.
        Tests if pred1 and pred2 have significantly different forecast accuracy.
        
        Returns:
            p_value: Two-tailed p-value
            dm_stat: Test statistic
        """
        error1 = actual - pred1
        error2 = actual - pred2
        
        loss_diff = error1**2 - error2**2
        
        mean_diff = np.mean(loss_diff)
        var_diff = np.var(loss_diff, ddof=1)
        
        if var_diff == 0:
            return 1.0, 0.0
        
        dm_stat = mean_diff / np.sqrt(var_diff / len(loss_diff))
        p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=len(loss_diff) - 1))
        
        return p_value, dm_stat
    
    @staticmethod
    def directional_accuracy_binomial_test(predictions, actuals):
        """
        Binomial test for directional accuracy.
        Tests if directional accuracy is significantly better than 50% (random).
        """
        pred_dir = np.sign(predictions)
        actual_dir = np.sign(actuals)
        
        num_correct = np.sum(pred_dir == actual_dir)
        num_total = len(predictions)
        
        # Binomial test (null: p=0.5)
        p_value = stats.binom_test(num_correct, num_total, 0.5, alternative='greater')
        
        return p_value, num_correct, num_total


class BacktestEngine:
    """Simple backtesting engine"""
    
    def __init__(self, initial_capital=1000000, transaction_cost=0.0005, 
                 slippage=0.0001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
    
    def backtest_long_short(self, predictions, returns, rebalance_freq='daily'):
        """
        Backtest a long-short strategy based on predictions.
        
        Args:
            predictions: (T, num_stocks) predicted returns
            returns: (T, num_stocks) actual returns
            rebalance_freq: 'daily' or 'weekly'
            
        Returns:
            strategy_returns: Daily strategy returns
            equity_curve: Cumulative equity curve
        """
        T, num_stocks = predictions.shape
        
        # Generate signals: long if prediction > 0, short if < 0
        signals = np.sign(predictions)
        
        # Normalize: equal weight long/short
        position_size = 1.0 / num_stocks
        
        # Portfolio returns
        portfolio_returns = np.zeros(T)
        
        for t in range(T):
            # Daily return from positions
            daily_return = np.sum(signals[t] * returns[t]) * position_size
            
            # Transaction costs
            if t > 0:
                position_change = np.abs(signals[t] - signals[t-1]) / 2  # 0.5 * turnover
                transaction_cost_cost = np.sum(position_change) * self.transaction_cost * position_size
            else:
                transaction_cost_cost = 0
            
            # Apply costs
            portfolio_returns[t] = daily_return - transaction_cost_cost
        
        # Equity curve
        equity_curve = self.initial_capital * np.cumprod(1 + portfolio_returns)
        
        return portfolio_returns, equity_curve
    
    def backtest_long_only(self, predictions, returns):
        """
        Backtest a long-only strategy.
        Only take positions when prediction > 0.
        """
        T, num_stocks = predictions.shape
        
        # Generate signals: long if prediction > 0, neutral otherwise
        signals = (predictions > 0).astype(float)
        
        # Portfolio returns
        portfolio_returns = np.zeros(T)
        
        for t in range(T):
            num_positions = np.sum(signals[t])
            if num_positions > 0:
                position_size = 1.0 / num_positions
                daily_return = np.sum(signals[t] * returns[t]) * position_size / num_stocks
            else:
                daily_return = 0
            
            # Transaction costs
            if t > 0:
                turnover = np.sum(np.abs(signals[t] - signals[t-1])) / 2
                transaction_cost_cost = turnover * self.transaction_cost / num_stocks
            else:
                transaction_cost_cost = 0
            
            portfolio_returns[t] = daily_return - transaction_cost_cost
        
        equity_curve = self.initial_capital * np.cumprod(1 + portfolio_returns)
        
        return portfolio_returns, equity_curve


class Visualizer:
    """Visualization utilities"""
    
    @staticmethod
    def plot_predictions_vs_actual(y_true, y_pred, title="Predictions vs Actual"):
        """Plot predictions vs actual values"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(y_true, label='Actual', alpha=0.7)
        ax.plot(y_pred, label='Predicted', alpha=0.7)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Return')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    @staticmethod
    def plot_equity_curves(equity_curves_dict, title="Equity Curves"):
        """Plot multiple equity curves"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for label, equity_curve in equity_curves_dict.items():
            ax.plot(equity_curve, label=label, alpha=0.7)
        
        ax.set_xlabel('Days')
        ax.set_ylabel('Equity ($)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    @staticmethod
    def plot_drawdown(returns, title="Drawdown"):
        """Plot cumulative drawdown"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        ax.plot(drawdown, color='red', alpha=0.7)
        
        ax.set_xlabel('Days')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    @staticmethod
    def plot_returns_distribution(returns, title="Returns Distribution"):
        """Plot histogram of returns"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(returns, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(returns), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(returns):.4f}')
        ax.axvline(np.median(returns), color='green', linestyle='--',
                  label=f'Median: {np.median(returns):.4f}')
        
        ax.set_xlabel('Daily Return')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        return fig, ax


if __name__ == "__main__":
    # Example usage
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1
    
    metrics = MetricsComputer.compute_all_metrics(y_true, y_pred)
    print("Metrics:", metrics)
    
    portfolio_returns = np.random.randn(252) * 0.01
    portfolio_metrics = PortfolioMetrics.compute_all(portfolio_returns)
    print("Portfolio Metrics:", portfolio_metrics)
