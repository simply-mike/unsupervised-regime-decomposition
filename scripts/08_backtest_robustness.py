from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

from regime_decomposition.backtest import build_default_exposure_map, run_regime_backtests  # noqa: E402
from regime_decomposition.config import DATA_PROCESSED_DIR, FIGURES_DIR, TABLES_DIR  # noqa: E402
from regime_decomposition.eda import CrisisWindow, DEFAULT_CRISIS_WINDOWS  # noqa: E402
from regime_decomposition.robustness import (  # noqa: E402
    build_exposure_scenarios,
    exposure_scenarios_to_frame,
    plot_robustness_drawdown,
    plot_robustness_sharpe,
    plot_stress_period_returns,
    run_backtest_robustness_grid,
    summarize_stress_period_performance,
)
from regime_decomposition.visualization import set_plot_style  # noqa: E402


OOS_STRESS_WINDOWS: tuple[CrisisWindow, ...] = (
    *DEFAULT_CRISIS_WINDOWS,
    CrisisWindow("euro_debt_2011", "2011-07-22", "2011-10-03", "US Downgrade / Euro Debt Stress"),
    CrisisWindow("q4_2018", "2018-09-20", "2018-12-24", "Q4 2018 Selloff"),
    CrisisWindow("vol_shock_2020", "2020-02-19", "2020-04-08", "COVID Crash and Initial Rebound"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 8: robustness checks for regime-aware backtests.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DATA_PROCESSED_DIR,
        help="Directory containing Step 6 walk-forward HMM outputs.",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        nargs="+",
        type=float,
        default=[0.0, 1.0, 5.0, 10.0],
        help="Transaction-cost values in bps for robustness grid.",
    )
    parser.add_argument("--signal-lag", type=int, default=1, help="Trading-day signal lag for all robustness runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_plot_style()

    panel_path = args.processed_dir / "walk_forward_hmm_labeled_panel.parquet"
    probabilities_path = args.processed_dir / "walk_forward_hmm_state_probabilities.parquet"
    regime_summary_path = TABLES_DIR / "step6_walk_forward_hmm_state_interpretation.csv"
    if not panel_path.exists() or not probabilities_path.exists() or not regime_summary_path.exists():
        raise FileNotFoundError("Missing Step 6 outputs. Run scripts/06_walk_forward_hmm.py first.")

    panel = pd.read_parquet(panel_path)
    probabilities = pd.read_parquet(probabilities_path)
    regime_summary = pd.read_csv(regime_summary_path)
    n_regimes = int(regime_summary["regime"].nunique())
    exposure_scenarios = build_exposure_scenarios(n_regimes=n_regimes)
    baseline_exposure_map = build_default_exposure_map(regime_summary=regime_summary)
    baseline_backtest = run_regime_backtests(
        panel=panel,
        probabilities=probabilities,
        exposure_map=baseline_exposure_map,
        transaction_cost_bps=1.0,
        signal_lag=args.signal_lag,
    )

    grid_summary = run_backtest_robustness_grid(
        panel=panel,
        probabilities=probabilities,
        exposure_scenarios=exposure_scenarios,
        transaction_cost_bps_values=tuple(args.transaction_cost_bps),
        signal_lag_values=(args.signal_lag,),
    )
    stress_period_summary = summarize_stress_period_performance(
        daily_results=baseline_backtest.daily_results,
        windows=OOS_STRESS_WINDOWS,
    )
    exposure_scenarios_df = exposure_scenarios_to_frame(exposure_scenarios)

    table_dir = TABLES_DIR
    figure_dir = FIGURES_DIR / "step8"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    grid_summary.to_csv(table_dir / "step8_robustness_grid.csv", index=False)
    stress_period_summary.to_csv(table_dir / "step8_stress_period_performance.csv", index=False)
    exposure_scenarios_df.to_csv(table_dir / "step8_exposure_scenarios.csv", index=False)

    plot_robustness_sharpe(grid_summary, figure_dir / "robustness_sharpe_by_cost.png")
    plot_robustness_drawdown(grid_summary, figure_dir / "robustness_drawdown_by_cost.png")
    plot_stress_period_returns(stress_period_summary, figure_dir / "stress_period_returns.png")

    display_columns = [
        "scenario",
        "transaction_cost_bps",
        "strategy",
        "annualized_return",
        "annualized_vol",
        "sharpe",
        "max_drawdown",
        "avg_exposure",
        "total_turnover",
    ]
    strategy_rows = grid_summary.query("strategy != 'buy_and_hold'").sort_values(
        ["strategy", "transaction_cost_bps", "scenario"]
    )
    stress_display = stress_period_summary.query("observations > 0")[
        ["window", "strategy", "observations", "total_return", "max_drawdown", "avg_exposure"]
    ]

    print("Step 8 complete.")
    print(f"Signal lag: {args.signal_lag} trading day(s)")
    print(f"Transaction-cost grid: {', '.join(f'{x:.1f}' for x in args.transaction_cost_bps)} bps")
    print("\nRobustness grid, non-benchmark strategies:")
    print(strategy_rows[display_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nStress-period baseline performance:")
    print(stress_display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nFigures saved to: {figure_dir}")


if __name__ == "__main__":
    main()
