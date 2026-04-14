# -*- coding: utf-8 -*-
"""
Portfolio Risk Metrics Dashboard
Interactive VaR/ES backtesting with Basel-compliant validation
Now with Sharpe-optimized portfolio backtesting
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm, chi2
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

if 'run_clicked' not in st.session_state:
    st.session_state.run_clicked = False

# ============================================================================
# Core Risk Functions
# ============================================================================

def fetch_data(tickers, start_date, end_date):
    """Download adjusted close prices and compute returns"""
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)
    adj_close = data['Adj Close'].dropna()
    returns = adj_close.pct_change().dropna()
    return adj_close, returns

def compute_portfolio_metrics(returns, weights=None):
    """Compute portfolio returns, volatility, and diversification benefit"""
    if weights is None:
        weights = np.ones(len(returns.columns)) / len(returns.columns)

    portfolio_returns = returns @ weights

    # Annualised metrics
    ann_return = portfolio_returns.mean() * 252
    ann_vol = portfolio_returns.std() * np.sqrt(252)

    # Individual volatilities
    ind_vols = returns.std() * np.sqrt(252)
    avg_ind_vol = ind_vols.mean()
    div_benefit = (avg_ind_vol - ann_vol) / avg_ind_vol

    return portfolio_returns, ann_return, ann_vol, div_benefit

def historical_var_es(returns, alpha_var=0.99, alpha_es=0.975, holding_days=10):
    """Historical VaR and Expected Shortfall"""
    var_1d = np.percentile(returns, (1 - alpha_var) * 100)
    es_threshold = np.percentile(returns, (1 - alpha_es) * 100)
    tail_returns = returns[returns <= es_threshold]
    es_1d = tail_returns.mean() if len(tail_returns) > 0 else var_1d

    var_10d = -var_1d * np.sqrt(holding_days)
    es_10d = -es_1d * np.sqrt(holding_days)

    return var_10d, es_10d

def parametric_var_es(returns, alpha_var=0.99, alpha_es=0.975, holding_days=10):
    """Delta-Normal VaR and Expected Shortfall"""
    mu = returns.mean()
    sigma = returns.std()

    z_var = norm.ppf(1 - alpha_var)
    z_es = norm.ppf(1 - alpha_es)

    var_10d = -(mu + sigma * z_var) * np.sqrt(holding_days)
    es_10d = abs((mu + sigma * norm.pdf(z_es) / (1 - alpha_es)) * np.sqrt(holding_days))

    return var_10d, es_10d

def kupiec_test(returns, var_threshold, alpha=0.01):
    """Kupiec Proportion of Failures test"""
    n = len(returns)
    failures = int(np.sum(returns < -var_threshold))
    failure_rate = failures / n
    expected = n * alpha

    if failures == 0:
        LR = -2 * np.log((1 - alpha) ** n)
    elif failures == n:
        LR = -2 * np.log(alpha ** n)
    else:
        LR = -2 * np.log(
            ((1 - alpha) ** (n - failures) * alpha ** failures) /
            ((1 - failure_rate) ** (n - failures) * failure_rate ** failures)
        )

    p_value = 1 - chi2.cdf(LR, df=1)
    reject = p_value < 0.01

    return {
        'failures': failures,
        'expected': round(expected, 1),
        'failure_rate': f"{failure_rate:.2%}",
        'p_value': round(p_value, 4),
        'reject': reject,
        'passed': not reject
    }

def traffic_light(returns, var_threshold, alpha=0.01, window=250):
    """Basel Traffic Light test"""
    n = len(returns)
    failures = int(np.sum(returns < -var_threshold))
    scale = n / window
    green_max = int(4 * scale)
    yellow_max = int(9 * scale)

    if failures <= green_max:
        zone = "🟢 GREEN"
        status = "Accept"
    elif failures <= yellow_max:
        zone = "🟡 YELLOW"
        status = "Scrutinize"
    else:
        zone = "🔴 RED"
        status = "Reject"

    return {'failures': failures, 'zone': zone, 'status': status}

def rolling_var(returns, window=250, alpha=0.99, holding_days=10):
    """Rolling window VaR"""
    rolling_values = []
    for i in range(window, len(returns)):
        window_returns = returns.iloc[i - window:i]
        var = np.percentile(window_returns, (1 - alpha) * 100)
        rolling_values.append(-var * np.sqrt(holding_days))

    return pd.Series(rolling_values, index=returns.index[window:])

def compute_betas(returns, market_ticker='SPY'):
    """Compute CAPM betas against market"""
    if market_ticker not in returns.columns:
        return {}

    market_ret = returns[market_ticker]
    betas = {}
    for ticker in returns.columns:
        if ticker != market_ticker:
            cov_im = np.cov(returns[ticker], market_ret)[0, 1]
            beta = cov_im / np.var(market_ret)
            betas[ticker] = beta

    return betas

def optimize_portfolio(returns, method='max_sharpe'):
    """Mean-variance portfolio optimization"""
    import scipy.optimize as sco
    
    def portfolio_stats(weights):
        pret = np.sum(returns.mean() * weights) * 252
        pvol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe = pret / pvol
        return pret, pvol, sharpe
    
    def neg_sharpe(weights):
        return -portfolio_stats(weights)[2]
    
    def port_variance(weights):
        return portfolio_stats(weights)[1] ** 2
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(returns.columns)))
    init_weights = np.array([1.0 / len(returns.columns)] * len(returns.columns))
    
    if method == 'max_sharpe':
        result = sco.minimize(neg_sharpe, init_weights, method='SLSQP', 
                              bounds=bounds, constraints=constraints)
    else:
        result = sco.minimize(port_variance, init_weights, method='SLSQP', 
                              bounds=bounds, constraints=constraints)
    
    optimal_weights = result['x']
    stats = portfolio_stats(optimal_weights)
    return optimal_weights, stats

# ============================================================================
# Streamlit UI
# ============================================================================

st.set_page_config(page_title="Risk Metrics Dashboard", page_icon="📊", layout="wide")
st.title("📊 Portfolio Risk Metrics Dashboard")
st.markdown("Basel-compliant VaR/ES with backtesting validation (Equal-weight & Optimized)")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    default_tickers = "AAPL, MSFT, SPY, RELIANCE.NS, SAIL.NS"
    ticker_input = st.text_input("Tickers (comma-separated)", default_tickers)
    tickers = [t.strip() for t in ticker_input.split(",")]

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=pd.to_datetime("2016-01-01"))
    with col2:
        end_date = st.date_input("End date", value=pd.to_datetime("2025-12-31"))

    st.subheader("Risk Parameters")
    alpha_var = st.selectbox("VaR Confidence Level", [0.95, 0.99, 0.999], index=1)
    alpha_es = st.selectbox("ES Confidence Level", [0.95, 0.975, 0.99], index=1)
    holding_days = st.number_input("Holding Period (days)", min_value=1, max_value=20, value=10)

    with st.expander("Advanced Options"):
        rolling_window = st.number_input("Rolling Window (days)", min_value=50, max_value=500, value=250)
        train_split = st.date_input("Train/Test Split", value=pd.to_datetime("2022-01-01"))
        market_ticker = st.text_input("Market Benchmark", "SPY")

    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Main content
if run_btn:
    st.session_state.run_clicked = True
if st.session_state.run_clicked:
   with st.spinner("Fetching data..."):
        try:
            prices, returns = fetch_data(tickers, start_date, end_date)
            if len(returns) == 0:
                st.error("No data retrieved. Check tickers and date range.")
                st.stop()

            # Equal-weight portfolio
            eq_returns, eq_ann_ret, eq_ann_vol, eq_div = compute_portfolio_metrics(returns)
            
            # Optimized portfolios
            opt_weights_sharpe, opt_stats_sharpe = optimize_portfolio(returns, 'max_sharpe')
            opt_weights_minvar, opt_stats_minvar = optimize_portfolio(returns, 'min_variance')
            sharpe_returns = returns @ opt_weights_sharpe
            minvar_returns = returns @ opt_weights_minvar

            split_date = pd.to_datetime(train_split)
            
            # Function to run full backtest suite
            def backtest_portfolio(port_returns, name):
                train = port_returns[port_returns.index < split_date]
                test = port_returns[port_returns.index >= split_date]
                var_1d = -np.percentile(train, (1 - alpha_var) * 100)
                kupiec = kupiec_test(test, var_1d, alpha=1 - alpha_var)
                tl = traffic_light(test, var_1d, alpha=1 - alpha_var)
                hist_var, hist_es = historical_var_es(port_returns, alpha_var, alpha_es, holding_days)
                return {
                    'Name': name,
                    'Train Returns': train,
                    'Test Returns': test,
                    'VaR 1d': var_1d,
                    'Kupiec': kupiec,
                    'Traffic Light': tl,
                    'Hist VaR 10d': hist_var,
                    'Hist ES 10d': hist_es
                }

            eq_bt = backtest_portfolio(eq_returns, "Equal-Weight")
            sharpe_bt = backtest_portfolio(sharpe_returns, "Max Sharpe")
            minvar_bt = backtest_portfolio(minvar_returns, "Min Variance")

            # Key metrics row (Equal-Weight)
            st.subheader("📈 Key Risk Metrics (Equal-Weight)")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Portfolio Volatility (Ann.)", f"{eq_ann_vol:.2%}")
            col2.metric("Portfolio Return (Ann.)", f"{eq_ann_ret:.2%}")
            col3.metric("Diversification Benefit", f"{eq_div:.1%}")
            col4.metric("10-day 99% VaR (Hist)", f"{eq_bt['Hist VaR 10d']:.2%}")
            col5.metric("10-day 97.5% ES (Hist)", f"{eq_bt['Hist ES 10d']:.2%}")

            # Portfolio Optimization Section
            st.subheader("📊 Portfolio Optimization (Mean-Variance)")
            opt_df = pd.DataFrame({
                'Asset': returns.columns,
                'Max Sharpe': opt_weights_sharpe.round(4),
                'Min Variance': opt_weights_minvar.round(4),
                'Equal Weight': [1.0/len(returns.columns)] * len(returns.columns)
            })
            st.dataframe(opt_df.style.format({
                'Max Sharpe': '{:.2%}', 
                'Min Variance': '{:.2%}', 
                'Equal Weight': '{:.2%}'
            }), use_container_width=True, hide_index=True)

            # Comparison Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Max Sharpe Return", f"{opt_stats_sharpe[0]:.2%}")
                st.metric("Max Sharpe Volatility", f"{opt_stats_sharpe[1]:.2%}")
                st.metric("Max Sharpe Ratio", f"{opt_stats_sharpe[2]:.2f}")
            with col2:
                st.metric("Min Variance Return", f"{opt_stats_minvar[0]:.2%}")
                st.metric("Min Variance Volatility", f"{opt_stats_minvar[1]:.2%}")
                st.metric("Min Variance Sharpe", f"{opt_stats_minvar[2]:.2f}")
            with col3:
                st.metric("Equal-Weight Return", f"{eq_ann_ret:.2%}")
                st.metric("Equal-Weight Volatility", f"{eq_ann_vol:.2%}")
                st.metric("Equal-Weight Sharpe", f"{eq_ann_ret/eq_ann_vol:.2f}")

            # VaR/ES Comparison
            st.subheader("📊 VaR & Expected Shortfall Comparison")
            var_es_df = pd.DataFrame({
                'Portfolio': ['Equal-Weight', 'Max Sharpe', 'Min Variance'],
                '99% VaR (10d)': [eq_bt['Hist VaR 10d'], sharpe_bt['Hist VaR 10d'], minvar_bt['Hist VaR 10d']],
                '97.5% ES (10d)': [eq_bt['Hist ES 10d'], sharpe_bt['Hist ES 10d'], minvar_bt['Hist ES 10d']]
            })
            st.dataframe(var_es_df.style.format({
                '99% VaR (10d)': '{:.2%}', 
                '97.5% ES (10d)': '{:.2%}'
            }), use_container_width=True, hide_index=True)

            # Backtesting Results (Three-Way)
            st.subheader("🔍 Backtesting Results (Out-of-Sample)")
            col1, col2, col3 = st.columns(3)
            
            for col, bt in zip([col1, col2, col3], [eq_bt, sharpe_bt, minvar_bt]):
                with col:
                    st.markdown(f"**{bt['Name']}**")
                    st.metric("Failures", f"{bt['Kupiec']['failures']} / {len(bt['Test Returns'])}")
                    st.metric("Expected", bt['Kupiec']['expected'])
                    st.metric("p-value", bt['Kupiec']['p_value'])
                    st.markdown(f"**Verdict:** {'✅ PASS' if bt['Kupiec']['passed'] else '❌ FAIL'}")
                    st.markdown(f"**Traffic Light:** {bt['Traffic Light']['zone']}")

            # Rolling VaR Chart (Equal-Weight + Max Sharpe overlay)
            st.subheader("📉 Rolling 10-day 99% VaR")
            roll_eq = rolling_var(eq_returns, rolling_window, alpha_var, holding_days)
            roll_sharpe = rolling_var(sharpe_returns, rolling_window, alpha_var, holding_days)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=roll_eq.index, y=roll_eq, mode='lines', 
                                     name='Equal-Weight', line=dict(color='red', width=1.5)))
            fig.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe, mode='lines', 
                                     name='Max Sharpe', line=dict(color='blue', width=1.5, dash='dot')))
            fig.update_layout(title=f"Rolling {int(alpha_var*100)}% VaR",
                              xaxis_title="Date", yaxis_title="Potential Loss",
                              yaxis_tickformat=".0%", height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Betas
            betas = compute_betas(returns, market_ticker)
            if betas:
                st.subheader("📐 CAPM Betas")
                beta_df = pd.DataFrame([{'Asset': k, 'Beta': f"{v:.3f}"} for k, v in betas.items()])
                st.dataframe(beta_df, use_container_width=True, hide_index=True)

            # Summary Statistics
            st.subheader("📊 Descriptive Statistics")
            stats_df = pd.DataFrame({
                'Ticker': returns.columns,
                'Ann. Return': returns.mean() * 252,
                'Ann. Vol': returns.std() * np.sqrt(252),
                'Skewness': returns.skew(),
                'Excess Kurtosis': returns.kurtosis()
            }).round(4)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

            # Download
            st.subheader("💾 Download Results")
            results_dict = {
                'Equal_Weight_VaR': eq_bt['Hist VaR 10d'],
                'Max_Sharpe_VaR': sharpe_bt['Hist VaR 10d'],
                'Min_Variance_VaR': minvar_bt['Hist VaR 10d'],
                'Equal_Weight_Kupiec_p': eq_bt['Kupiec']['p_value'],
                'Max_Sharpe_Kupiec_p': sharpe_bt['Kupiec']['p_value'],
                'Min_Variance_Kupiec_p': minvar_bt['Kupiec']['p_value']
            }
            results_df = pd.DataFrame([results_dict])
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results CSV", csv, "risk_results.csv", "text/csv")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Try: fewer tickers, different date range, or check internet connection")

else:
    st.info("👈 Configure parameters in the sidebar and click 'Run Analysis'")
    st.markdown("""
    ### Features
    - **Multiple VaR Methods**: Historical, Parametric, and Rolling Window
    - **Regulatory Backtesting**: Kupiec POF test, Basel Traffic Light framework
    - **Portfolio Optimization**: Max Sharpe & Min Variance (with backtesting)
    - **Interactive Visualizations**: Rolling risk metrics and exception timelines
    """)
