from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from regime_decomposition.backtest import TRADING_DAYS_PER_YEAR, run_regime_backtests
from regime_decomposition.eda import CrisisWindow


@dataclass(frozen=True)
class RobustnessResult:
    grid_summary: pd.DataFrame
    stress_period_summary: pd.DataFrame
    exposure_scenarios: pd.DataFrame


def build_exposure_scenarios(n_regimes: int) -> dict[str, pd.Series]:
    """Create conservative, balanced, and aggressive long/cash maps."""

    if n_regimes < 1:
        raise ValueError("n_regimes must be positive.")

    if n_regimes == 1:
        scenario_values = {
            "balanced": [1.0],
            "defensive": [1.0],
            "aggressive": [1.0],
        }
    elif n_regimes == 4:
        scenario_values = {
            "balanced": [1.0, 1.0, 0.5, 0.0],
            "defensive": [1.0, 0.75, 0.25, 0.0],
            "aggressive": [1.0, 1.0, 0.75, 0.25],
            "crisis_cut": [1.0, 0.75, 0.0, 0.0],
        }
    else:
        balanced = np.linspace(1.0, 0.0, n_regimes)
        scenario_values = {
            "balanced": balanced,
            "defensive": np.sqrt(balanced),
            "aggressive": np.maximum(balanced, 0.25),
        }

    scenarios: dict[str, pd.Series] = {}
    for name, values in scenario_values.items():
        scenarios[name] = pd.Series(values, index=range(n_regimes), name="target_exposure").clip(0.0, 1.0)
        scenarios[name].index.name = "regime"
    return scenarios


def exposure_scenarios_to_frame(scenarios: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for scenario, exposure_map in scenarios.items():
        for regime, exposure in exposure_map.items():
            rows.append({"scenario": scenario, "regime": int(regime), "target_exposure": float(exposure)})
    return pd.DataFrame(rows)


def run_backtest_robustness_grid(
    panel: pd.DataFrame,
    probabilities: pd.DataFrame,
    exposure_scenarios: dict[str, pd.Series],
    transaction_cost_bps_values: tuple[float, ...] = (0.0, 1.0, 5.0, 10.0),
    signal_lag_values: tuple[int, ...] = (1,),
) -> pd.DataFrame:
    """Run Step 7 backtests across exposure and cost assumptions."""

    rows = []
    for scenario_name, exposure_map in exposure_scenarios.items():
        for transaction_cost_bps in transaction_cost_bps_values:
            for signal_lag in signal_lag_values:
                result = run_regime_backtests(
                    panel=panel,
                    probabilities=probabilities,
                    exposure_map=exposure_map,
                    transaction_cost_bps=transaction_cost_bps,
                    signal_lag=signal_lag,
                )
                summary = result.summary.copy()
                summary.insert(0, "scenario", scenario_name)
                summary.insert(1, "transaction_cost_bps", float(transaction_cost_bps))
                summary.insert(2, "signal_lag", int(signal_lag))
                rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def summarize_stress_period_performance(
    daily_results: pd.DataFrame,
    windows: tuple[CrisisWindow, ...],
    strategies: tuple[str, ...] = ("buy_and_hold", "hard_regime_filter", "probability_scaled"),
) -> pd.DataFrame:
    """Recompute performance metrics inside each named date window."""

    rows = []
    for window in windows:
        window_data = daily_results.loc[window.start : window.end]
        for strategy in strategies:
            if window_data.empty:
                rows.append(_empty_window_row(window, strategy))
                continue

            returns = window_data[f"{strategy}_return"].dropna()
            exposure = window_data[f"{strategy}_exposure"].dropna()
            turnover = window_data[f"{strategy}_turnover"].dropna()
            transaction_cost = window_data[f"{strategy}_transaction_cost"].dropna()
            equity = (1.0 + returns).cumprod()
            drawdown = equity.div(equity.cummax()).sub(1.0)
            years = len(returns) / TRADING_DAYS_PER_YEAR

            total_return = float(equity.iloc[-1] - 1.0)
            annualized_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else np.nan
            annualized_vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
            sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if returns.std(ddof=1) else np.nan

            rows.append(
                {
                    "window": window.name,
                    "label": window.label,
                    "window_start": window.start,
                    "window_end": window.end,
                    "observed_start": window_data.index.min().date(),
                    "observed_end": window_data.index.max().date(),
                    "strategy": strategy,
                    "observations": int(len(returns)),
                    "total_return": total_return,
                    "annualized_return": annualized_return,
                    "annualized_vol": annualized_vol,
                    "sharpe": sharpe,
                    "max_drawdown": float(drawdown.min()),
                    "avg_exposure": float(exposure.mean()),
                    "total_turnover": float(turnover.sum()),
                    "total_transaction_cost": float(transaction_cost.sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_robustness_sharpe(grid_summary: pd.DataFrame, output_path: Path) -> None:
    data = grid_summary.query("strategy in ['hard_regime_filter', 'probability_scaled']").copy()
    grid = sns.FacetGrid(data, col="strategy", hue="scenario", height=4.2, aspect=1.25, sharey=True)
    grid.map_dataframe(sns.lineplot, x="transaction_cost_bps", y="sharpe", marker="o")
    grid.add_legend(title="Scenario")
    grid.set_axis_labels("Transaction cost, bps", "Sharpe")
    grid.set_titles("{col_name}")
    grid.fig.suptitle("Backtest Robustness: Sharpe vs Transaction Costs", y=1.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.fig.savefig(output_path, bbox_inches="tight")
    plt.close(grid.fig)


def plot_robustness_drawdown(grid_summary: pd.DataFrame, output_path: Path) -> None:
    data = grid_summary.query("strategy in ['hard_regime_filter', 'probability_scaled']").copy()
    grid = sns.FacetGrid(data, col="strategy", hue="scenario", height=4.2, aspect=1.25, sharey=True)
    grid.map_dataframe(sns.lineplot, x="transaction_cost_bps", y="max_drawdown", marker="o")
    grid.add_legend(title="Scenario")
    grid.set_axis_labels("Transaction cost, bps", "Max drawdown")
    grid.set_titles("{col_name}")
    grid.fig.suptitle("Backtest Robustness: Drawdown vs Transaction Costs", y=1.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.fig.savefig(output_path, bbox_inches="tight")
    plt.close(grid.fig)


def plot_stress_period_returns(stress_summary: pd.DataFrame, output_path: Path) -> None:
    data = stress_summary[stress_summary["observations"] > 0].copy()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.barplot(data=data, x="window", y="total_return", hue="strategy", ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Stress-Period Strategy Returns")
    ax.set_xlabel("")
    ax.set_ylabel("Total return inside window")
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _empty_window_row(window: CrisisWindow, strategy: str) -> dict[str, object]:
    return {
        "window": window.name,
        "label": window.label,
        "window_start": window.start,
        "window_end": window.end,
        "observed_start": pd.NaT,
        "observed_end": pd.NaT,
        "strategy": strategy,
        "observations": 0,
        "total_return": np.nan,
        "annualized_return": np.nan,
        "annualized_vol": np.nan,
        "sharpe": np.nan,
        "max_drawdown": np.nan,
        "avg_exposure": np.nan,
        "total_turnover": np.nan,
        "total_transaction_cost": np.nan,
    }
