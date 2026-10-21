import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from inference import run_inference

st.set_page_config(page_title="RC-DGAT Stock Regime GUI", layout="wide")
st.title("RC-DGAT Advanced Stock Regime Prediction GUI")


# Sidebar
st.sidebar.header("User Input")

num_shares = st.sidebar.number_input(
    "Number of Shares", min_value=1, max_value=1000, value=10
)

# Predefined ticker options (can be expanded)
available_tickers = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "HINDUNILVR.NS",
    "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS", "ITC.NS", "ASIANPAINT.NS",
    "AXISBANK.NS", "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS"
]

ticker_list = st.sidebar.multiselect(
    "Select Stock Tickers",
    options=available_tickers,
    default=["RELIANCE.NS", "TCS.NS", "INFY.NS"]
)

date = st.sidebar.date_input("Prediction Date")

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if st.sidebar.button("Run Prediction"):
    with st.spinner("Running model inference..."):
        results = run_inference(ticker_list, num_shares, date)

    st.success("Inference complete!")
    st.subheader("Prediction Results")

    # -------------------------------
    # SAFE TABLE DISPLAY
    # -------------------------------
    if isinstance(results, dict):
        st.markdown("### 📊 Prediction Table")
        # Case 1: Already row-wise → ideal
        if "table" in results and isinstance(results["table"], list):
            df = pd.DataFrame(results["table"])
            # Add recommendation column
            def get_recommendation(signal):
                if signal == "BUY":
                    return "Buy"
                elif signal == "HOLD":
                    return "Hold"
                else:
                    return "Sell"
            df["Recommendation"] = df["signal"].apply(get_recommendation)
            st.dataframe(df[["ticker", "price", "signal", "Recommendation", "regime", "allocation", "shares"]])
        else:
            # Extract only equal-length arrays
            valid_cols = {}
            lengths = []
            for k, v in results.items():
                if isinstance(v, (list, np.ndarray)):
                    valid_cols[k] = list(v)
                    lengths.append(len(v))
            # Check if all lengths match
            if len(lengths) > 0 and len(set(lengths)) == 1:
                df = pd.DataFrame(valid_cols)
                if "signal" in df.columns:
                    def get_recommendation(signal):
                        if signal == "BUY":
                            return "Buy"
                        elif signal == "HOLD":
                            return "Hold"
                        else:
                            return "Sell"
                    df["Recommendation"] = df["signal"].apply(get_recommendation)
                st.dataframe(df)
            else:
                st.warning("Results are not in tabular format. Showing raw output.")
                st.json(results)
    else:
        st.write(results)


    # -------------------------------
    # VISUALIZATION BLOCK
    # -------------------------------
    if isinstance(results, dict):
        # Portfolio Allocation
        if "allocation" in results:
            st.markdown("### 🧮 Portfolio Allocation")
            st.caption("Shows the proportion of each stock in the recommended portfolio.")
            try:
                alloc = pd.Series(results["allocation"])
                fig = px.bar(alloc, x=alloc.index, y=alloc.values, labels={"x": "Stock Ticker", "y": "Allocation (%)"}, title="Portfolio Allocation")
                fig.update_layout(xaxis_title="Stock Ticker", yaxis_title="Allocation (%)")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.write(results["allocation"])
                st.error(f"Plot error: {e}")

        # Equity Curve
        if "equity_curve" in results:
            st.markdown("### 📈 Equity Curve")
            st.caption("Displays the growth of portfolio value over time.")
            try:
                eq = pd.Series(results["equity_curve"])
                fig = px.line(eq, x=eq.index, y=eq.values, labels={"x": "Time Step", "y": "Portfolio Value"}, title="Equity Curve")
                fig.update_layout(xaxis_title="Time Step", yaxis_title="Portfolio Value")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.write(results["equity_curve"])
                st.error(f"Plot error: {e}")

        # Regime Probabilities
        if "regime_probs" in results:
            st.markdown("### 🔄 Regime Probabilities Over Time")
            st.caption("Probability of each market regime at each time step.")
            try:
                probs = pd.DataFrame(results["regime_probs"])
                fig = go.Figure()
                for col in probs.columns:
                    fig.add_trace(go.Scatter(x=probs.index, y=probs[col], mode='lines', name=str(col)))
                fig.update_layout(title="Regime Probabilities", xaxis_title="Time Step", yaxis_title="Probability", legend_title="Regime")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.write(results["regime_probs"])
                st.error(f"Plot error: {e}")

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.info(
    "Advanced GUI for RC-DGAT: Enter stocks, shares, and date to generate predictions and analytics."
)