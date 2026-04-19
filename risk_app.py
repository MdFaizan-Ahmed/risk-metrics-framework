# -*- coding: utf-8 -*-
"""
Portfolio Risk Metrics Dashboard
Interactive VaR/ES backtesting with Basel-compliant validation
Now with Sharpe-optimized portfolio backtesting + Custom Weights
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
    div_benefit = (avg_ind_vol - ann_vol) / avg_ind_vol if avg_ind_vol > 0 else 0

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

def kupiec_test(returns, var_threshold, alpha=0.05):
    """Kupiec Proportion of Failures test"""
    n = len(returns)
    failures = int(np.sum(returns < -var_threshold))
    failure_rate = failures / n if n > 0 else 0
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

def traffic_light(returns, var_threshold, alpha=0.05, window=250):
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
            beta = cov_im / np.var(market_ret) if np.var(market_ret) > 0 else 0
            betas[ticker] = beta

    return betas

def optimize_portfolio(returns, method='max_sharpe'):
    """Mean-variance portfolio optimization"""
    import scipy.optimize as sco
    
    def portfolio_stats(weights):
        pret = np.sum(returns.mean() * weights) * 252
        pvol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe = pret / pvol if pvol > 0 else 0
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
st.markdown("Basel-compliant VaR/ES with backtesting validation | Compare vs. Sharpe-Optimized Portfolio")

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

    # Portfolio Selection
    st.subheader("📊 Portfolio Selection")
    portfolio_type = st.radio(
        "Choose portfolio type",
        ["Custom Weights", "Min Variance (Benchmark)"],
        help="Custom: Enter your own weights. Benchmark: Minimum Variance portfolio (lowest possible volatility)"
    )
    
    custom_weights_dict = None
    if portfolio_type == "Custom Weights":
        st.markdown("**Enter your portfolio weights (must sum to 100%):**")
        custom_weights_dict = {}
        total = 0
        for ticker in tickers:
            weight = st.number_input(
                f"{ticker}", 
                min_value=0.0, 
                max_value=1.0, 
                value=1.0/len(tickers), 
                step=0.01,
                format="%.2f"
            )
            custom_weights_dict[ticker] = weight
            total += weight
        
        if abs(total - 1.0) > 0.01:
            st.warning(f"⚠️ Weights sum to {total:.1%}, not 100%. Will normalize.")
            # Normalize
            for ticker in custom_weights_dict:
                custom_weights_dict[ticker] /= total
            st.success(f"✅ Normalized weights (now sum to 100%)")
        
        # Optional: Show Sharpe-optimized for comparison
        show_sharpe_benchmark = st.checkbox("📈 Also compare with Sharpe-Optimized portfolio", value=True)
    else:
        show_sharpe_benchmark = False

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

            # Always compute Sharpe-optimized and Min Variance for benchmarking
            sharpe_weights, sharpe_stats = optimize_portfolio(returns, 'max_sharpe')
            minvar_weights, minvar_stats = optimize_portfolio(returns, 'min_variance')
            
            # Calculate portfolio returns for benchmarks
            sharpe_returns, sharpe_ann_ret, sharpe_ann_vol, sharpe_div = compute_portfolio_metrics(returns, sharpe_weights)
            minvar_returns, minvar_ann_ret, minvar_ann_vol, minvar_div = compute_portfolio_metrics(returns, minvar_weights)
            
            # ========== PORTFOLIO DEFINITION ==========
            portfolios_to_analyze = []
            
            if portfolio_type == "Custom Weights":
                # Create custom weight array
                custom_weights = np.array([custom_weights_dict[t] for t in tickers])
                custom_returns, custom_ann_ret, custom_ann_vol, custom_div = compute_portfolio_metrics(returns, custom_weights)
                portfolios_to_analyze.append({
                    'name': "🎯 Your Custom Portfolio",
                    'returns': custom_returns,
                    'ann_ret': custom_ann_ret,
                    'ann_vol': custom_ann_vol,
                    'div_benefit': custom_div,
                    'weights': custom_weights,
                    'type': 'custom'
                })
                
                # Show custom weights table
                st.subheader("📊 Your Custom Portfolio Weights")
                weights_df = pd.DataFrame({
                    'Asset': tickers,
                    'Weight': custom_weights
                })
                st.dataframe(weights_df.style.format({'Weight': '{:.2%}'}), 
                            use_container_width=True, hide_index=True)
                
                # Add Sharpe-optimized benchmark if requested
                if show_sharpe_benchmark:
                    portfolios_to_analyze.append({
                        'name': "📈 Sharpe-Optimized (Benchmark)",
                        'returns': sharpe_returns,
                        'ann_ret': sharpe_ann_ret,
                        'ann_vol': sharpe_ann_vol,
                        'div_benefit': sharpe_div,
                        'weights': sharpe_weights,
                        'type': 'sharpe',
                        'sharpe_ratio': sharpe_stats[2]
                    })
                
                # Always add Min Variance as second benchmark (lowest risk)
                portfolios_to_analyze.append({
                    'name': "🛡️ Min Variance (Lowest Risk)",
                    'returns': minvar_returns,
                    'ann_ret': minvar_ann_ret,
                    'ann_vol': minvar_ann_vol,
                    'div_benefit': minvar_div,
                    'weights': minvar_weights,
                    'type': 'minvar',
                    'sharpe_ratio': minvar_stats[2]
                })
                
            else:  # Min Variance Benchmark mode
                portfolios_to_analyze.append({
                    'name': "🛡️ Min Variance Portfolio",
                    'returns': minvar_returns,
                    'ann_ret': minvar_ann_ret,
                    'ann_vol': minvar_ann_vol,
                    'div_benefit': minvar_div,
                    'weights': minvar_weights,
                    'type': 'minvar',
                    'sharpe_ratio': minvar_stats[2]
                })
                
                # Also show Sharpe-optimized for context
                portfolios_to_analyze.append({
                    'name': "📈 Sharpe-Optimized (For Context)",
                    'returns': sharpe_returns,
                    'ann_ret': sharpe_ann_ret,
                    'ann_vol': sharpe_ann_vol,
                    'div_benefit': sharpe_div,
                    'weights': sharpe_weights,
                    'type': 'sharpe',
                    'sharpe_ratio': sharpe_stats[2]
                })

            # Show optimization weights table for transparency
            if len(portfolios_to_analyze) > 1:
                st.subheader("📊 Benchmark Portfolio Weights")
                bench_df = pd.DataFrame({'Asset': returns.columns})
                for port in portfolios_to_analyze:
                    if port['type'] in ['sharpe', 'minvar']:
                        bench_df[port['name']] = port['weights']
                if len(bench_df.columns) > 1:
                    st.dataframe(bench_df.style.format({col: '{:.2%}' for col in bench_df.columns if col != 'Asset'}), 
                                use_container_width=True, hide_index=True)

            split_date = pd.to_datetime(train_split)
            
            # Function to run full backtest suite
            def backtest_portfolio(port_returns, name, weights=None, port_type=None):
                train = port_returns[port_returns.index < split_date]
                test = port_returns[port_returns.index >= split_date]
                
                if len(train) == 0 or len(test) == 0:
                    return None
                
                var_1d = -np.percentile(train, (1 - alpha_var) * 100)
                kupiec = kupiec_test(test, var_1d, alpha=1 - alpha_var)
                tl = traffic_light(test, var_1d, alpha=1 - alpha_var)
                hist_var, hist_es = historical_var_es(port_returns, alpha_var, alpha_es, holding_days)
                param_var, param_es = parametric_var_es(port_returns, alpha_var, alpha_es, holding_days)
                
                return {
                    'Name': name,
                    'Train Returns': train,
                    'Test Returns': test,
                    'VaR 1d': var_1d,
                    'Kupiec': kupiec,
                    'Traffic Light': tl,
                    'Hist VaR 10d': hist_var,
                    'Hist ES 10d': hist_es,
                    'Param VaR 10d': param_var,
                    'Param ES 10d': param_es,
                    'Weights': weights,
                    'Type': port_type
                }

            # Run backtest for all portfolios
            backtest_results = []
            for port in portfolios_to_analyze:
                bt = backtest_portfolio(port['returns'], port['name'], port.get('weights'), port.get('type'))
                if bt:
                    bt.update(port)
                    backtest_results.append(bt)

            if not backtest_results:
                st.error("No valid portfolios to analyze. Check train/test split dates.")
                st.stop()

            # ========== DISPLAY RESULTS ==========
            
            # Key metrics row (custom portfolio if exists, otherwise min variance)
            primary = backtest_results[0]
            st.subheader(f"📈 Key Risk Metrics ({primary['Name']})")
            cols = st.columns(5)
            cols[0].metric("Portfolio Volatility (Ann.)", f"{primary['ann_vol']:.2%}")
            cols[1].metric("Portfolio Return (Ann.)", f"{primary['ann_ret']:.2%}")
            cols[2].metric("Sharpe Ratio", f"{primary['ann_ret']/primary['ann_vol']:.2f}" if primary['ann_vol'] > 0 else "N/A")
            cols[3].metric("10-day 99% VaR (Hist)", f"{primary['Hist VaR 10d']:.2%}")
            cols[4].metric("10-day 97.5% ES (Hist)", f"{primary['Hist ES 10d']:.2%}")

            # Comparison Metrics Table
            st.subheader("📊 Portfolio Comparison vs. Benchmarks")
            comparison_data = []
            for bt in backtest_results:
                sharpe_ratio = bt['ann_ret']/bt['ann_vol'] if bt['ann_vol'] > 0 else 0
                # Calculate outperformance vs Min Variance (if not itself)
                if bt['type'] != 'minvar' and len([b for b in backtest_results if b.get('type') == 'minvar']) > 0:
                    minvar_port = [b for b in backtest_results if b.get('type') == 'minvar'][0]
                    vol_diff = (bt['ann_vol'] - minvar_port['ann_vol']) / minvar_port['ann_vol']
                    ret_diff = bt['ann_ret'] - minvar_port['ann_ret']
                else:
                    vol_diff = None
                    ret_diff = None
                
                row = {
                    'Portfolio': bt['Name'],
                    'Ann. Return': f"{bt['ann_ret']:.2%}",
                    'Ann. Vol': f"{bt['ann_vol']:.2%}",
                    'Sharpe': f"{sharpe_ratio:.2f}",
                    '99% VaR (10d)': f"{bt['Hist VaR 10d']:.2%}",
                    '97.5% ES (10d)': f"{bt['Hist ES 10d']:.2%}",
                    'Backtest Result': '✅ PASS' if bt['Kupiec']['passed'] else '❌ FAIL'
                }
                comparison_data.append(row)
            
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
            
            # If custom portfolio, show how it compares to benchmarks
            if portfolio_type == "Custom Weights" and show_sharpe_benchmark:
                custom_port = backtest_results[0]
                sharpe_port = [b for b in backtest_results if b.get('type') == 'sharpe'][0]
                minvar_port = [b for b in backtest_results if b.get('type') == 'minvar'][0]
                
                st.subheader("🎯 Your Portfolio vs. Benchmarks")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**vs. Sharpe-Optimized**")
                    ret_diff = custom_port['ann_ret'] - sharpe_port['ann_ret']
                    vol_diff = custom_port['ann_vol'] - sharpe_port['ann_vol']
                    st.metric("Return Difference", f"{ret_diff:+.2%}", 
                             delta_color="normal" if ret_diff > 0 else "inverse")
                    st.metric("Volatility Difference", f"{vol_diff:+.2%}",
                             delta_color="inverse" if vol_diff < 0 else "normal")
                with col2:
                    st.markdown("**vs. Min Variance**")
                    ret_diff = custom_port['ann_ret'] - minvar_port['ann_ret']
                    vol_diff = custom_port['ann_vol'] - minvar_port['ann_vol']
                    st.metric("Return Difference", f"{ret_diff:+.2%}",
                             delta_color="normal" if ret_diff > 0 else "inverse")
                    st.metric("Volatility Difference", f"{vol_diff:+.2%}",
                             delta_color="inverse" if vol_diff < 0 else "normal")
                with col3:
                    st.markdown("**Risk-Adjusted**")
                    custom_sharpe = custom_port['ann_ret']/custom_port['ann_vol'] if custom_port['ann_vol'] > 0 else 0
                    sharpe_sharpe = sharpe_port['ann_ret']/sharpe_port['ann_vol'] if sharpe_port['ann_vol'] > 0 else 0
                    st.metric("Your Sharpe Ratio", f"{custom_sharpe:.2f}")
                    st.metric("Sharpe-Optimized", f"{sharpe_sharpe:.2f}")

            # VaR/ES Comparison Table
            st.subheader("📊 VaR & Expected Shortfall Comparison")
            var_es_data = []
            for bt in backtest_results:
                var_es_data.append({
                    'Portfolio': bt['Name'],
                    'Historical 99% VaR (10d)': bt['Hist VaR 10d'],
                    'Historical 97.5% ES (10d)': bt['Hist ES 10d'],
                    'Parametric 99% VaR (10d)': bt['Param VaR 10d'],
                    'Parametric 97.5% ES (10d)': bt['Param ES 10d']
                })
            
            var_es_df = pd.DataFrame(var_es_data)
            st.dataframe(var_es_df.style.format({
                'Historical 99% VaR (10d)': '{:.2%}',
                'Historical 97.5% ES (10d)': '{:.2%}',
                'Parametric 99% VaR (10d)': '{:.2%}',
                'Parametric 97.5% ES (10d)': '{:.2%}'
            }), use_container_width=True, hide_index=True)

            # Backtesting Results
            st.subheader("🔍 Regulatory Backtesting Results (Out-of-Sample)")
            st.caption("Kupiec POF Test: Checks if failure rate matches 99% confidence level")
            
            # Dynamic columns based on number of portfolios
            backtest_cols = st.columns(len(backtest_results))
            for col, bt in zip(backtest_cols, backtest_results):
                with col:
                    st.markdown(f"**{bt['Name']}**")
                    st.metric("VaR Exceptions", f"{bt['Kupiec']['failures']} / {len(bt['Test Returns'])}")
                    st.metric("Expected", bt['Kupiec']['expected'])
                    st.metric("p-value", bt['Kupiec']['p_value'])
                    st.markdown(f"**Verdict:** {'✅ PASS' if bt['Kupiec']['passed'] else '❌ FAIL'}")
                    st.markdown(f"**Basel Zone:** {bt['Traffic Light']['zone']}")

            # Rolling VaR Chart (all portfolios)
            st.subheader("📉 Rolling 10-day 99% VaR Comparison")
            fig = go.Figure()
            colors = {'custom': '#FF6B6B', 'sharpe': '#4ECDC4', 'minvar': '#45B7D1'}
            line_styles = {'custom': 'solid', 'sharpe': 'dash', 'minvar': 'dot'}
            
            for bt in backtest_results:
                roll = rolling_var(bt['returns'], rolling_window, alpha_var, holding_days)
                fig.add_trace(go.Scatter(
                    x=roll.index, y=roll, mode='lines', 
                    name=bt['Name'], 
                    line=dict(color=colors.get(bt.get('type', 'custom'), '#888888'), 
                             width=2 if bt.get('type') == 'custom' else 1.5,
                             dash=line_styles.get(bt.get('type', 'custom'), 'solid'))
                ))
            
            fig.update_layout(
                title=f"Rolling {int(alpha_var*100)}% VaR ({holding_days}-day) - Lower is Better",
                xaxis_title="Date", 
                yaxis_title="Expected Loss",
                yaxis_tickformat=".0%", 
                height=450,
                legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
            )
            st.plotly_chart(fig, use_container_width=True)

            # Betas
            betas = compute_betas(returns, market_ticker)
            if betas:
                st.subheader("📐 CAPM Betas (Market Sensitivity)")
                beta_df = pd.DataFrame([{'Asset': k, 'Beta': f"{v:.3f}", 
                                        'Interpretation': 'High risk' if v > 1.2 else ('Low risk' if v < 0.8 else 'Market-like')} 
                                       for k, v in betas.items()])
                st.dataframe(beta_df, use_container_width=True, hide_index=True)

            # Correlation Heatmap
            st.subheader("📊 Asset Correlation Matrix")
            corr = returns.corr()
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr.values, 
                x=corr.columns, 
                y=corr.columns, 
                colorscale='RdBu', 
                zmid=0,
                text=corr.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)

            # Summary Statistics
            st.subheader("📊 Individual Asset Statistics")
            stats_df = pd.DataFrame({
                'Ticker': returns.columns,
                'Ann. Return': returns.mean() * 252,
                'Ann. Vol': returns.std() * np.sqrt(252),
                'Sharpe': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
                'Skewness': returns.skew(),
                'Excess Kurtosis': returns.kurtosis()
            }).round(4)
            st.dataframe(stats_df.style.format({
                'Ann. Return': '{:.2%}',
                'Ann. Vol': '{:.2%}',
                'Sharpe': '{:.2f}'
            }), use_container_width=True, hide_index=True)

            # Download Results
            st.subheader("💾 Download Results")
            
            # Prepare download data
            download_data = []
            for bt in backtest_results:
                download_data.append({
                    'Portfolio': bt['Name'],
                    'Annual_Return': bt['ann_ret'],
                    'Annual_Volatility': bt['ann_vol'],
                    'Sharpe_Ratio': bt['ann_ret']/bt['ann_vol'] if bt['ann_vol'] > 0 else np.nan,
                    'Historical_VaR_10d': bt['Hist VaR 10d'],
                    'Historical_ES_10d': bt['Hist ES 10d'],
                    'Parametric_VaR_10d': bt['Param VaR 10d'],
                    'Parametric_ES_10d': bt['Param ES 10d'],
                    'Kupiec_Failures': bt['Kupiec']['failures'],
                    'Kupiec_p_value': bt['Kupiec']['p_value'],
                    'Kupiec_Passed': bt['Kupiec']['passed'],
                    'Basel_Traffic_Light': bt['Traffic Light']['zone']
                })
            
            results_df = pd.DataFrame(download_data)
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results CSV", csv, "risk_results.csv", "text/csv")
            
            # If custom weights, also download the weights
            if portfolio_type == "Custom Weights" and custom_weights_dict:
                weights_df = pd.DataFrame([custom_weights_dict])
                weights_csv = weights_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Custom Weights", weights_csv, "portfolio_weights.csv", "text/csv")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("💡 Tips: Try fewer tickers, different date range, or check your internet connection")

else:
    st.info("👈 Configure parameters in the sidebar and click 'Run Analysis'")
    st.markdown("""
    ### 🎯 Why Sharpe-Optimized is the Right Benchmark
    
    **Equal-weight is arbitrary** - it assumes you have no view on asset returns or risk.
    
    **Sharpe-Optimized (Tangency Portfolio)** is theoretically optimal because:
    - Maximizes return per unit of risk
    - Represents the efficient frontier's best risk-adjusted return
    - What mean-variance optimization actually solves for
    
    ### 📊 What This Dashboard Does
    
    - **Custom Portfolio**: Enter your own weights to test your strategy
    - **Sharpe-Optimized Benchmark**: Compare against the theoretical optimum
    - **Min Variance Benchmark**: See the lowest possible volatility portfolio
    - **Regulatory Validation**: Kupiec tests & Basel Traffic Light zones
    - **Rolling Risk Analysis**: See how VaR changes through market cycles
    
    ### 🚀 Quick Start
    
    1. Enter your tickers (e.g., `AAPL, MSFT, GOOGL`)
    2. Choose "Custom Weights" and enter your allocation
    3. Compare against Sharpe-Optimized and Min Variance benchmarks
    4. See if your strategy beats the theoretical optimum
    """)

    # Example comparison
    with st.expander("📖 Understanding the Benchmarks"):
        st.markdown("""
        **Sharpe-Optimized Portfolio**
        - Solves for weights that maximize (Return - RiskFree)/Volatility
        - Theoretically the best risk-adjusted return
        - Often concentrated in best-performing assets
        - Can have high volatility despite high Sharpe
        
        **Min Variance Portfolio**
        - Solves for lowest possible volatility regardless of return
        - Often diversified across uncorrelated assets
        - Lower risk but potentially lower returns
        - Good for risk-averse investors
        
        **Your Custom Portfolio**
        - Your actual or proposed allocation
        - Compare Sharpe ratio to see if you're beating the market
        - Compare VaR to see if you're taking more tail risk
        - Regulatory backtest shows if your VaR model is accurate
        """)
