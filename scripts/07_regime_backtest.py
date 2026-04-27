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

from regime_decomposition.backtest import (  # noqa: E402
    build_default_exposure_map,
    plot_drawdowns,
    plot_equity_curves,
    plot_exposures,
    run_regime_backtests,
)
from regime_decomposition.config import DATA_PROCESSED_DIR, FIGURES_DIR, TABLES_DIR  # noqa: E402
from regime_decomposition.visualization import set_plot_style  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 7: regime-aware walk-forward HMM backtest.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DATA_PROCESSED_DIR,
        help="Directory containing Step 6 walk-forward HMM outputs.",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=1.0,
        help="One-way transaction cost in basis points per unit of exposure turnover.",
    )
    parser.add_argument(
        "--signal-lag",
        type=int,
        default=1,
        help="Number of trading days between regime signal and executable exposure.",
    )
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
    exposure_map = build_default_exposure_map(regime_summary=regime_summary)

    result = run_regime_backtests(
        panel=panel,
        probabilities=probabilities,
        exposure_map=exposure_map,
        transaction_cost_bps=args.transaction_cost_bps,
        signal_lag=args.signal_lag,
    )

    table_dir = TABLES_DIR
    figure_dir = FIGURES_DIR / "step7"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    result.daily_results.to_parquet(args.processed_dir / "regime_backtest_daily_results.parquet")
    result.summary.to_csv(table_dir / "step7_backtest_summary.csv", index=False)
    result.yearly_returns.to_csv(table_dir / "step7_backtest_yearly_returns.csv", index=False)
    result.exposure_map.to_csv(table_dir / "step7_regime_exposure_map.csv", index=False)

    plot_equity_curves(result.daily_results, figure_dir / "equity_curves.png")
    plot_drawdowns(result.daily_results, figure_dir / "drawdowns.png")
    plot_exposures(result.daily_results, figure_dir / "exposures.png")

    display_columns = [
        "strategy",
        "total_return",
        "annualized_return",
        "annualized_vol",
        "sharpe",
        "max_drawdown",
        "avg_exposure",
        "total_turnover",
        "total_transaction_cost",
    ]
    print("Step 7 complete.")
    print(f"Signal lag: {args.signal_lag} trading day(s)")
    print(f"Transaction cost: {args.transaction_cost_bps:.2f} bps per unit turnover")
    print("\nRegime exposure map:")
    exposure_display = result.exposure_map.merge(
        regime_summary[["regime", "regime_name"]],
        on="regime",
        how="left",
    )[["regime", "regime_name", "target_exposure"]]
    print(exposure_display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nBacktest summary:")
    print(result.summary[display_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nFigures saved to: {figure_dir}")


if __name__ == "__main__":
    main()
