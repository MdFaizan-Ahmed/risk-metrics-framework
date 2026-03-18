# risk-metrics-framework
A comprehensive VaR/ES backtesting framework comparing Historical, Parametric (Delta-Normal), and Monte Carlo methods. Implements Kupiec POF, Christoffersen Conditional Coverage, and Basel Traffic Light tests with train/test split validation. Reveals why all models fail out-of-sample, but for fundamentally different reasons.

# Portfolio Risk Metrics: VaR, ES & Backtesting Validation

A ground-up implementation of market risk models built iteratively across six notebook versions — from basic descriptive statistics to regulatory-grade backtesting validation. The progression is intentional: each version adds a layer of rigour, and the earlier notebooks document the reasoning process rather than just the final answer.

**Portfolio:** 5-asset equal-weighted portfolio — AAPL, MSFT, RELIANCE.NS, SAIL.NS, SPY  
**Period:** January 2016 – January 2026 | **Train/Test split:** 2016–2021 / 2022–2026  
**Stack:** Python — pandas, NumPy, SciPy, matplotlib, yfinance

---

## What's implemented

**Risk metrics**
- Annualised mean, volatility, skewness, excess kurtosis
- Correlation and covariance matrix construction (Σ = Corr ⊙ σᵢσⱼ)
- Beta and CAPM expected return
- Portfolio variance via **wᵀΣw** with diversification benefit
- Historical VaR & ES (empirical percentile)
- Parametric (Delta-Normal) VaR & ES under Gaussian assumptions
- Historical Monte Carlo VaR (bootstrap resampling from empirical distribution)
- 10-day scaling via square-root-of-time rule

**Backtesting frameworks**
- Kupiec's Proportion of Failures (POF) test: likelihood ratio test on exception rate
- Christoffersen's Conditional Coverage test: extends Kupiec by testing independence of exceptions
- Basel Traffic Light test:  Green / Yellow / Red zone classification
- In-sample and out-of-sample validation with strict train/test split
- Synthetic normal data sanity check: confirms parametric failure is distributional, not a bug

---

## The notebooks

The repo preserves the full development history. This was a deliberate choice as the learning process is part of the work.

| Notebook | Description |
|----------|-------------|
| `v01` | Foundation. Descriptive statistics, correlation, covariance matrix, Beta/CAPM, portfolio construction, Historical and Parametric VaR/ES. Both simple and log returns computed in parallel for early exploration of their behaviour. |
| `v02` | Same as v01 but cleaner. Redundant plots removed, comments tightened. The log returns parallel track remains as I was still building intuition at this stage. |
| `v03` | Backtesting added. All three frameworks i.e., Kupiec POF, Christoffersen Conditional Coverage, and Basel Traffic Light are implemented. First serious engagement with the question: *does the model actually work?* |
| `v04` | Out-of-sample validation and Monte Carlo VaR added. Train/test split introduced properly. Rolling VaR/ES. Three-way model comparison: Historical vs Parametric vs MC. |
| `v05A` *(TrainRets)* | Full refactor. Config block, clean function signatures, structured markdown narrative. **Strict version:** in-sample thresholds derived from `train_returns` only, ensuring no data leakage anywhere. |
| `v05B` *(Full_Returns)* | Full refactor, same structure as v05A. **Looser version:** in-sample thresholds derived from the full dataset, which is closer to how in-sample fit is typically presented in practice. Explicitly acknowledged in the notebook. |

The purpose of v05A and v05B is not that one is "wrong" but that they answer slightly different questions. v05A is more rigorous for research purposes; v05B reflects common industry presentation.

---

## Key findings

Historical VaR consistently outperforms Parametric across all backtesting frameworks. The parametric model's normality assumption breaks down on real equity return distributions, which exhibit negative skewness and excess kurtosis. Accordingly, it fails Kupiec even in-sample, with failures clustering during stress periods (Christoffersen test).

Historical Monte Carlo, by bootstrapping from the empirical distribution, preserving fat tails and avoiding the Gaussian assumption, produces thresholds close to Historical VaR and passes out-of-sample backtesting.

The synthetic normal data check is worth noting: the parametric model passes Kupiec when tested on data that is genuinely normally distributed. This confirms the implementation is correct and the real-data failure is entirely distributional.

```
── Final Model Comparison ──
Model               IS failures  OOS failures  IS p-value  OOS p-value  IS verdict  OOS verdict
Historical VaR               15             3      0.8683       0.0129    ✅ Pass      ✅ Pass
Parametric VaR               26            13      0.0056       0.2860    ❌ Fail      ✅ Pass
Historical MC VaR           N/A             3         N/A       0.0129       N/A      ✅ Pass
```
*(Results from v05A: train-only thresholds)*

---

## A note on methodology

**Why SPY instead of ^GSPC?** SPY is actually tradeable. Beta computed against an untradeable index creates a replication problem; SPY is the cleaner benchmark.

**Why simple returns for portfolio arithmetic?** Log returns are additive over time but not across assets. Portfolio returns require simple return aggregation (weighted sum), so simple returns are used for portfolio construction and parametric VaR. Log returns are included in earlier versions for comparison.

**On √T scaling:** The square-root-of-time rule holds exactly only for i.i.d. zero-mean returns. Parametric ES scales slightly faster than √T because the φ(z)/(1−α) term in ES_T = T·μ + √T·σ·φ(z_α)/(1−α) amplifies tail sensitivity. This is documented explicitly in v05A/B.

**On the Monte Carlo method:** Historical MC resamples (with replacement) from the empirical training distribution. It is not independent of Historical VaR as both draw from the same empirical distribution. As a result, similar thresholds are expected and intended. The value is in the simulation framework itself, which extends naturally to path-dependent risk measures.

---

## What's next

This is the first notebook in a planned series building toward a full market risk framework:

- [ ] Stress testing & scenario analysis
- [ ] Regression & time series fundamentals
- [ ] Greeks & sensitivity-based approaches (Taylor series / Duration-Convexity)
- [ ] GARCH — time-varying volatility
- [ ] Filtered Historical Simulation (FHS)
- [ ] Portfolio mapping — Systematic VaR, Specific VaR, Factor Models & PCA
- [ ] Risk aggregation
- [ ] FRTB SA/IMA
- [ ] P&L Attribution & backtesting
- [ ] Model validation — SR 11-7

---

## Dependencies

```
pip install pandas numpy scipy matplotlib yfinance
```
