from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def compute_drawdown(equity: pd.Series) -> pd.Series:
    """Compute drawdown from a positive equity curve."""

    return equity.div(equity.cummax()).sub(1.0)


def summarize_return_series(
    returns: pd.Series,
    exposure: pd.Series,
    turnover: pd.Series,
    transaction_cost: pd.Series,
) -> dict[str, float]:
    """Summarize daily simple returns with standard trading metrics."""

    clean_returns = returns.dropna()
    if clean_returns.empty:
        return {
            "observations": 0,
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
            "hit_rate": np.nan,
            "avg_exposure": np.nan,
            "max_exposure": np.nan,
            "avg_daily_turnover": np.nan,
            "total_turnover": np.nan,
            "total_transaction_cost": np.nan,
        }

    equity = (1.0 + clean_returns).cumprod()
    drawdown = compute_drawdown(equity)
    years = len(clean_returns) / TRADING_DAYS_PER_YEAR
    return_std = clean_returns.std(ddof=1)
    total_return = float(equity.iloc[-1] - 1.0)
    annualized_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if equity.iloc[-1] > 0 else np.nan
    annualized_vol = float(return_std * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = float(clean_returns.mean() / return_std * np.sqrt(TRADING_DAYS_PER_YEAR)) if return_std else np.nan
    max_drawdown = float(drawdown.min())
    calmar = float(annualized_return / abs(max_drawdown)) if max_drawdown < 0 else np.nan

    aligned_exposure = exposure.reindex(clean_returns.index).dropna()
    aligned_turnover = turnover.reindex(clean_returns.index).dropna()
    aligned_transaction_cost = transaction_cost.reindex(clean_returns.index).dropna()

    return {
        "observations": int(len(clean_returns)),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "hit_rate": float((clean_returns > 0).mean()),
        "avg_exposure": float(aligned_exposure.mean()),
        "max_exposure": float(aligned_exposure.max()),
        "avg_daily_turnover": float(aligned_turnover.mean()),
        "total_turnover": float(aligned_turnover.sum()),
        "total_transaction_cost": float(aligned_transaction_cost.sum()),
    }
