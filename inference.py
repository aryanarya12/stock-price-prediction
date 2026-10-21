import numpy as np
import pandas as pd

def run_inference(ticker_list, num_shares, date):
    """
    Standardized inference output:
    - 'table' → row-wise structured data (safe for DataFrame)
    - other keys → time series / vectors for charts
    """

    # -------------------------------
    # MOCK MODEL LOGIC (replace later)
    # -------------------------------
    np.random.seed(42)

    n = len(ticker_list)

    prices = np.random.uniform(100, 3000, n)
    signals = np.random.choice(["BUY", "SELL", "HOLD"], n)
    regimes = np.random.choice(["Bull", "Bear", "Sideways"], n)

    # Allocation (normalized weights)
    raw_alloc = np.random.rand(n)
    allocation = raw_alloc / raw_alloc.sum()

    # Equity curve (time series)
    equity_curve = np.cumsum(np.random.randn(100)) + 10000

    # Regime probabilities (time × regime matrix)
    regime_probs = np.random.dirichlet([1, 1, 1], size=100)

    # -------------------------------
    # BUILD SAFE TABULAR OUTPUT
    # -------------------------------
    table = []

    for i in range(n):
        row = {
            "ticker": ticker_list[i],
            "price": float(prices[i]),
            "signal": signals[i],
            "regime": regimes[i],
            "allocation": float(allocation[i]),
            "shares": int(num_shares)
        }
        table.append(row)

    # -------------------------------
    # FINAL STRUCTURED OUTPUT
    # -------------------------------
    results = {
        "table": table,                        # ✅ ALWAYS SAFE
        "allocation": allocation.tolist(),     # for bar chart
        "equity_curve": equity_curve.tolist(), # for line chart
        "regime_probs": pd.DataFrame(
            regime_probs,
            columns=["Bull", "Bear", "Sideways"]
        )
    }

    return results