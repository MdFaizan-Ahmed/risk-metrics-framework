# Portfolio Risk Metrics & Backtesting Framework

Market risk modelling and validation framework implementing VaR/ES across Historical, Parametric (Delta-Normal), and Monte Carlo approaches, with full backtesting and out-of-sample validation.

## Overview

This project implements a complete market risk framework for a multi-asset portfolio, with a focus on model validation, backtesting, and distributional risk analysis.

It evaluates model performance under:

- In-sample and out-of-sample conditions
- Different distributional assumptions
- Regulatory backtesting standards

The objective is not only to compute VaR, but to assess whether risk models are reliable under real market conditions.

## Portfolio & Data
- Assets: AAPL, MSFT, RELIANCE.NS, SAIL.NS, SPY<br>
- Period: Jan 2016 – Jan 2026<br>
- Split:
   - Training: 2016–2021<br>
   - Testing: 2022–2026<br>
- Data source: yfinance

## Risk Models Implemented
- Historical VaR & Expected Shortfall
    - Empirical distribution-based
- Parametric (Delta-Normal) VaR & ES
    - Gaussian assumption with covariance estimation
- Monte Carlo VaR
    - Bootstrap resampling from empirical returns

## Portfolio Construction
Variance via: wᵀΣw
Covariance matrix constructed from asset correlations and volatilities
CAPM beta estimation
Diversification effects explicitly modelled

## Backtesting & Validation
- Kupiec POF Test (exception rate)
- Christoffersen Conditional Coverage Test (independence + clustering)
- Basel Traffic Light Framework
- Strict train/test separation
- Out-of-sample validation across 2,300+ trading days

## Key Findings
- Parametric VaR fails in real-world data
- Due to violation of normality (fat tails, skewness)
- Fails Kupiec test even in-sample
- Exhibits clustering of violations during stress periods
- Historical VaR is more robust
- Better captures empirical distribution characteristics
- More stable across validation frameworks
- Monte Carlo (bootstrap) performs comparably to Historical VaR
- Preserves fat-tail behaviour
- Produces consistent thresholds and passes OOS tests
- Model risk is highly regime-dependent
- Performance degrades significantly out-of-sample
- Highlights importance of validation beyond calibration
- Synthetic normal data validation confirms implementation correctness
- Parametric model passes when assumptions are satisfied
- Confirms failures are due to data properties, not coding errors

## Key Insight

- All models fail out-of-sample but for fundamentally different reasons.
- This framework isolates why models fail, not just whether they fail.

## Methodological Notes

- Simple vs Log Returns
    - Simple returns used for portfolio aggregation
    - Log returns used for distributional analysis
- Scaling Assumptions
    - √T scaling only holds under i.i.d. assumptions
    - ES exhibits slightly faster scaling due to tail sensitivity
- Monte Carlo Approach
    - Bootstrap resampling preserves empirical distribution
    - Extends naturally to path-dependent risk measures

## Future Work
- Stress testing & scenario analysis
- GARCH-based volatility modelling
- Filtered Historical Simulation (FHS)
- Factor models & PCA-based risk decomposition
- FRTB (SA/IMA) implementation
- P&L attribution & model validation frameworks (SR 11-7)

## Tech Stack
Python: pandas, NumPy, SciPy, matplotlib
Data: yfinance
```
