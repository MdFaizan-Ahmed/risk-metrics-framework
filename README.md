# Portfolio Risk Metrics & Backtesting Framework

**Live Dashboard:** https://risk-metrics-framework-backtesting-validation.streamlit.app/

**Market risk modelling and validation framework implementing VaR/ES across Historical, Parametric (Delta-Normal), and Monte Carlo methods, with full in-sample and out-of-sample backtesting.**

---

## Overview

This project implements a multi-method market risk framework for a diversified equity portfolio and evaluates model performance using **regulatory backtesting standards and out-of-sample validation**.

The focus is on **model behaviour under different data regimes**, rather than purely in-sample fit.

---

## Portfolio & Data

- Assets: AAPL, MSFT, RELIANCE.NS, SAIL.NS, SPY  
- Period: Jan 2016 – Jan 2026  
- Split:
  - In-sample: 2016–2021  
  - Out-of-sample: 2022–2026  
- Data source: yfinance  

---

## Risk Models Implemented

- Historical VaR & Expected Shortfall  
  - Empirical distribution-based approach  

- Parametric (Delta-Normal) VaR & ES  
  - Gaussian assumption with covariance estimation  

- Monte Carlo VaR (Historical Simulation)  
  - Bootstrap resampling from empirical returns  

---

## Portfolio Framework

- Portfolio variance via: wᵀΣw  
- Covariance matrix constructed from asset correlations and volatilities  
- CAPM beta estimation  
- Diversification effects explicitly captured  

---

## Backtesting & Validation

- Kupiec Proportion of Failures (POF) test  
- Christoffersen Conditional Coverage test  
- Basel Traffic Light framework  
- Strict train/test separation  
- Out-of-sample validation across 2,300+ trading days  

---

## Key Findings

- Parametric VaR shows mixed performance across regimes  
  - Fails in-sample (2016–2021)  
  - Passes out-of-sample (2022–2026)  
  - Reflects estimation window effects and regime dependence rather than consistent superiority  

- Historical VaR remains broadly robust  
  - Passes both in-sample and out-of-sample tests  
  - Better captures empirical distribution characteristics  

- Monte Carlo (bootstrap) aligns closely with Historical VaR  
  - Produces similar thresholds  
  - Passes out-of-sample validation  
  - Inherits empirical distribution properties  

- Model performance is regime-dependent  
  - Backtesting results vary significantly across time periods  
  - Highlights the impact of non-stationarity and changing volatility regimes  

- Backtesting results alone are insufficient to rank model quality  
  - Outcomes are sensitive to:
    - Data regime  
    - Estimation window  
    - Sample-specific characteristics  

---

## Key Insight

- Model performance is conditional on the data regime  
- A model may fail under one distributional environment and pass under another without implying a structural improvement  

This highlights the importance of **robust validation across multiple regimes**, not just in-sample calibration.

---

## Methodological Notes

- Simple vs Log Returns  
  - Simple returns used for portfolio aggregation  
  - Log returns used for distributional analysis  

- Scaling Assumptions  
  - √T scaling assumes i.i.d. returns  
  - ES scaling exhibits slightly faster growth due to tail sensitivity  

- Monte Carlo Approach  
  - Bootstrap resampling from empirical distribution  
  - Not independent of historical VaR, but useful for simulation-based extension  

---

## Development Approach

- Initial exploration of returns, covariance, and portfolio construction  
- Introduction of VaR/ES modelling under multiple assumptions  
- Implementation of regulatory backtesting frameworks  
- Addition of out-of-sample validation  
- Refinement through structured model comparison  

This iterative approach mirrors how risk models are developed and validated in practice.

---

## Future Work

- Stress testing & scenario analysis  
- GARCH-based volatility modelling  
- Filtered Historical Simulation (FHS)  
- Factor models & PCA-based risk decomposition  
- FRTB (SA/IMA) framework  
- P&L attribution & model validation (SR 11-7)  

---

## Tech Stack

- Python: pandas, NumPy, SciPy, matplotlib  
- Data: yfinance  
