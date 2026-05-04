# Unsupervised Regime Decomposition

Unsupervised market regime detection on daily multi-asset data using PCA/SVD, clustering, Gaussian mixtures, Hidden Markov Models, and walk-forward risk-overlay backtests.

The project studies whether liquid market data can be decomposed into persistent, interpretable regimes without using supervised labels such as "bull" or "bear". The final output is not a return-forecasting model. It is a regime-aware risk overlay that changes SPY exposure when the inferred market state becomes more fragile.

## Research Goal

The goal is to build a compact quant research pipeline that connects three pieces:

- matrix decomposition of cross-asset returns
- unsupervised regime modeling
- out-of-sample portfolio behavior under regime-aware exposure rules

The main question is:

> Can unsupervised models identify market states that are useful for risk management, even if they do not produce standalone alpha?

## Data

Default universe:

`SPY`, `QQQ`, `IWM`, `TLT`, `GLD`, `^VIX`

Daily Yahoo Finance data are used from 2000 onward. The shared multi-asset history starts later because GLD begins trading in 2004.

Main features:

- daily log returns
- PCA scores from the standardized multi-asset return matrix
- SPY realized volatility
- VIX level and VIX changes
- drawdown and crisis-window diagnostics

## Methods

The project is built in layers.

First, SVD/PCA is applied to the standardized return matrix. This gives a low-dimensional representation of the cross-asset market state and makes it possible to inspect broad equity-risk, rates, and defensive-market components.

Second, clustering models are fit in the PCA/volatility/VIX feature space:

- K-Means
- Ward hierarchical clustering
- full-covariance Gaussian Mixture Models

These models are useful for comparing regime geometry, cluster uncertainty, and the effect of soft versus hard assignments.

Third, Gaussian Hidden Markov Models are used to model regime persistence. The HMM transition matrix gives expected state durations and makes the output more natural for risk monitoring than independent daily clustering.

The final inference layer is walk-forward. Each fold fits the scaler and HMM only on historical data, then filters the next out-of-sample block. Trading exposure is lagged by one day to avoid using same-day close-derived information.

## Main Results

The PCA/SVD layer finds that the first three principal components explain most of the standardized cross-asset variation:

| Component | Explained Variance |
|---|---:|
| PC1 | 57.7% |
| PC2 | 22.5% |
| PC3 | 15.1% |

The walk-forward HMM sample runs from `2010-02-18` to `2026-05-01`.

| Strategy | Total Return | Ann. Return | Ann. Vol | Sharpe | Max Drawdown | Avg Exposure |
|---|---:|---:|---:|---:|---:|---:|
| Buy and hold | 775.0% | 14.35% | 17.15% | 0.87 | -33.72% | 1.00 |
| Hard regime filter | 247.0% | 8.00% | 10.28% | 0.80 | -17.35% | 0.79 |
| Probability scaled | 255.8% | 8.16% | 10.19% | 0.82 | -18.01% | 0.79 |

The regime overlay does not beat buy-and-hold on absolute return over a strong equity sample. Its value is risk management: lower realized volatility, materially smaller drawdowns, and better behavior in stress windows.

Selected stress-window behavior:

| Period | Buy and Hold | Regime Overlay |
|---|---:|---:|
| COVID crash 2020 | -33.4% | about -8% |
| 2022 rates/inflation bear market | -24.1% | about -16% |
| 2011 sovereign-risk stress | -17.8% | about -7% |
| Q4 2018 selloff | -18.7% | about -11.5% |

The economic interpretation is deliberately conservative: the evidence supports dynamic risk budgeting, not a claim that unsupervised regimes are a standalone alpha signal.

## Reproducibility

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

The numbered scripts in `scripts/` reproduce the research stages. Intermediate datasets are written to `data/processed/`; tables and figures are written to `results/`.

## Limitations

Yahoo Finance is suitable for a public research project, but not for production trading research. Full-sample PCA, clustering, GMM, and HMM outputs are exploratory. The stricter layer is the walk-forward HMM and backtest with signal lag, transaction costs, and stress-period checks.

Natural extensions include rolling-window HMM refits, richer macro and credit features, cash returns, volatility targeting, and stronger regime-stability diagnostics.
