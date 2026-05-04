from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regime_decomposition.metrics import compute_drawdown, summarize_return_series


@dataclass(frozen=True)
class BacktestResult:
    daily_results: pd.DataFrame
    summary: pd.DataFrame
    yearly_returns: pd.DataFrame
    exposure_map: pd.DataFrame


def build_default_exposure_map(regime_summary: pd.DataFrame | None = None, n_regimes: int | None = None) -> pd.Series:
    """Map calm-to-stress regime IDs into SPY target exposures."""

    if regime_summary is not None:
        regimes = sorted(regime_summary["regime"].astype(int).unique())
    elif n_regimes is not None:
        regimes = list(range(n_regimes))
    else:
        raise ValueError("Provide regime_summary or n_regimes.")

    if len(regimes) == 1:
        exposures = [1.0]
    elif len(regimes) == 2:
        exposures = [1.0, 0.0]
    elif len(regimes) == 3:
        exposures = [1.0, 0.5, 0.0]
    elif len(regimes) == 4:
        exposures = [1.0, 1.0, 0.5, 0.0]
    else:
        exposures = np.linspace(1.0, 0.0, len(regimes)).round(4).tolist()

    exposure_map = pd.Series(exposures, index=regimes, name="target_exposure")
    exposure_map.index.name = "regime"
    return exposure_map


def run_regime_backtests(
    panel: pd.DataFrame,
    probabilities: pd.DataFrame,
    exposure_map: pd.Series,
    transaction_cost_bps: float = 1.0,
    signal_lag: int = 1,
) -> BacktestResult:
    """Run benchmark, hard-label, and probability-scaled regime strategies."""

    if "spy_ret" not in panel.columns:
        raise KeyError("panel must contain 'spy_ret'.")
    if "walk_forward_hmm_state" not in panel.columns:
        raise KeyError("panel must contain 'walk_forward_hmm_state'.")

    data = panel.join(probabilities, how="inner").copy()
    spy_log_returns = data["spy_ret"].replace([np.inf, -np.inf], np.nan).dropna()
    data = data.reindex(spy_log_returns.index)
    probabilities = probabilities.reindex(spy_log_returns.index)

    hard_target = data["walk_forward_hmm_state"].astype(int).map(exposure_map).rename("hard_regime_filter")
    prob_target = probability_scaled_exposure(probabilities=probabilities, exposure_map=exposure_map)

    strategy_results = [
        run_buy_and_hold(spy_log_returns),
        run_strategy_backtest(
            spy_log_returns=spy_log_returns,
            target_exposure=hard_target,
            strategy_name="hard_regime_filter",
            transaction_cost_bps=transaction_cost_bps,
            signal_lag=signal_lag,
        ),
        run_strategy_backtest(
            spy_log_returns=spy_log_returns,
            target_exposure=prob_target,
            strategy_name="probability_scaled",
            transaction_cost_bps=transaction_cost_bps,
            signal_lag=signal_lag,
        ),
    ]

    daily_results = pd.concat(strategy_results, axis=1)
    strategies = ["buy_and_hold", "hard_regime_filter", "probability_scaled"]
    summary = summarize_backtest(daily_results, strategies)
    yearly_returns = compute_yearly_returns(daily_results, strategies)
    exposure_map_df = exposure_map.rename("target_exposure").reset_index()
    return BacktestResult(
        daily_results=daily_results,
        summary=summary,
        yearly_returns=yearly_returns,
        exposure_map=exposure_map_df,
    )


def probability_scaled_exposure(probabilities: pd.DataFrame, exposure_map: pd.Series) -> pd.Series:
    """Convert regime posterior probabilities into expected SPY exposure."""

    aligned = pd.DataFrame(index=probabilities.index)
    for regime, exposure in exposure_map.items():
        column = f"regime_{int(regime)}"
        if column not in probabilities.columns:
            raise KeyError(f"Missing probability column: {column}")
        aligned[column] = probabilities[column] * float(exposure)
    return aligned.sum(axis=1).rename("probability_scaled")


def apply_signal_lag(target_exposure: pd.Series, signal_lag: int = 1, fill_value: float = 0.0) -> pd.Series:
    """Shift target exposure so today's return uses information known before today."""

    if signal_lag < 0:
        raise ValueError("signal_lag must be non-negative.")
    if signal_lag == 0:
        return target_exposure.astype(float)
    return target_exposure.shift(signal_lag).fillna(fill_value).astype(float)


def run_buy_and_hold(spy_log_returns: pd.Series) -> pd.DataFrame:
    spy_simple_returns = np.expm1(spy_log_returns).rename("spy_simple_return")
    equity = (1.0 + spy_simple_returns).cumprod()
    return pd.DataFrame(
        {
            "spy_simple_return": spy_simple_returns,
            "buy_and_hold_target_exposure": 1.0,
            "buy_and_hold_exposure": 1.0,
            "buy_and_hold_turnover": 0.0,
            "buy_and_hold_transaction_cost": 0.0,
            "buy_and_hold_return": spy_simple_returns,
            "buy_and_hold_equity": equity,
            "buy_and_hold_drawdown": compute_drawdown(equity),
        }
    )


def run_strategy_backtest(
    spy_log_returns: pd.Series,
    target_exposure: pd.Series,
    strategy_name: str,
    transaction_cost_bps: float = 1.0,
    signal_lag: int = 1,
) -> pd.DataFrame:
    """Backtest a long/cash SPY allocation strategy with turnover costs."""

    target = target_exposure.reindex(spy_log_returns.index).astype(float).clip(0.0, 1.0)
    exposure = apply_signal_lag(target, signal_lag=signal_lag, fill_value=0.0).clip(0.0, 1.0)
    turnover = exposure.diff().abs()
    turnover.iloc[0] = abs(exposure.iloc[0])
    transaction_cost = turnover * (transaction_cost_bps / 10_000.0)

    spy_simple_returns = np.expm1(spy_log_returns)
    strategy_returns = exposure * spy_simple_returns - transaction_cost
    equity = (1.0 + strategy_returns).cumprod()

    return pd.DataFrame(
        {
            f"{strategy_name}_target_exposure": target,
            f"{strategy_name}_exposure": exposure,
            f"{strategy_name}_turnover": turnover,
            f"{strategy_name}_transaction_cost": transaction_cost,
            f"{strategy_name}_return": strategy_returns,
            f"{strategy_name}_equity": equity,
            f"{strategy_name}_drawdown": compute_drawdown(equity),
        }
    )


def summarize_backtest(daily_results: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    rows = []

    for strategy in strategies:
        metrics = summarize_return_series(
            returns=daily_results[f"{strategy}_return"],
            exposure=daily_results[f"{strategy}_exposure"],
            turnover=daily_results[f"{strategy}_turnover"],
            transaction_cost=daily_results[f"{strategy}_transaction_cost"],
        )
        metrics.update(
            {
                "strategy": strategy,
                "start": daily_results.index.min().date(),
                "end": daily_results.index.max().date(),
            }
        )
        rows.append(metrics)

    columns = [
        "strategy",
        "start",
        "end",
        "observations",
        "total_return",
        "annualized_return",
        "annualized_vol",
        "sharpe",
        "max_drawdown",
        "calmar",
        "hit_rate",
        "avg_exposure",
        "max_exposure",
        "avg_daily_turnover",
        "total_turnover",
        "total_transaction_cost",
    ]
    return pd.DataFrame(rows).loc[:, columns]


def compute_yearly_returns(daily_results: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    rows = []
    for year, group in daily_results.groupby(daily_results.index.year):
        row = {"year": int(year)}
        for strategy in strategies:
            row[strategy] = float((1.0 + group[f"{strategy}_return"]).prod() - 1.0)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_equity_curves(daily_results: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    _plot_strategy_columns(daily_results, "equity", ax, loc="upper left")
    ax.set_title("Regime-Aware Strategy Equity Curves")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_drawdowns(daily_results: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    _plot_strategy_columns(daily_results, "drawdown", ax, loc="lower left")
    ax.set_title("Regime-Aware Strategy Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_exposures(daily_results: pd.DataFrame, output_path: Path) -> None:
    columns = ["hard_regime_filter_exposure", "probability_scaled_exposure"]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    daily_results[columns].plot(ax=ax, linewidth=1.0)
    ax.set_title("Regime-Aware SPY Exposure")
    ax.set_ylabel("Exposure")
    ax.set_xlabel("")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(["Hard regime filter", "Probability-scaled"], loc="lower left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_strategy_columns(daily_results: pd.DataFrame, suffix: str, ax: plt.Axes, loc: str) -> None:
    columns = [column for column in daily_results.columns if column.endswith(f"_{suffix}")]
    labels = [column.removesuffix(f"_{suffix}").replace("_", " ").title() for column in columns]
    daily_results[columns].plot(ax=ax, linewidth=1.4)
    ax.legend(labels, loc=loc)
