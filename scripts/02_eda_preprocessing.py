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

from regime_decomposition.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, FIGURES_DIR, TABLES_DIR  # noqa: E402
from regime_decomposition.eda import (  # noqa: E402
    DEFAULT_CRISIS_WINDOWS,
    build_eda_panel,
    plot_crisis_window_dashboard,
    plot_stress_overview,
    summarize_crisis_windows,
    summarize_data_quality,
    summarize_feature_missingness,
)
from regime_decomposition.visualization import set_plot_style  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 2: EDA and preprocessing sanity checks.")
    parser.add_argument(
        "--raw-data-path",
        type=Path,
        default=None,
        help="Path to raw market parquet. Defaults to the newest data/raw/market_data_*.parquet.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DATA_PROCESSED_DIR,
        help="Directory containing Step 1 processed parquet files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_plot_style()

    raw_data_path = args.raw_data_path or _latest_raw_market_data_path()
    market_data = pd.read_parquet(raw_data_path)
    returns = pd.read_parquet(args.processed_dir / "returns_matrix.parquet")
    features = pd.read_parquet(args.processed_dir / "feature_matrix.parquet")
    pca_scores = pd.read_parquet(args.processed_dir / "pca_component_scores.parquet")

    panel = build_eda_panel(market_data=market_data, returns=returns, features=features, pca_scores=pca_scores)
    data_quality = summarize_data_quality(
        market_data=market_data,
        returns=returns,
        features=features,
        pca_scores=pca_scores,
    )
    missingness = summarize_feature_missingness(panel)
    crisis_summary = summarize_crisis_windows(panel=panel, windows=DEFAULT_CRISIS_WINDOWS)

    table_dir = TABLES_DIR
    figure_dir = FIGURES_DIR / "step2"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    panel.to_parquet(args.processed_dir / "eda_panel.parquet")
    data_quality.to_csv(table_dir / "step2_data_quality_summary.csv", index=False)
    missingness.to_csv(table_dir / "step2_eda_panel_missingness.csv")
    crisis_summary.to_csv(table_dir / "step2_crisis_window_summary.csv", index=False)

    plot_stress_overview(panel=panel, windows=DEFAULT_CRISIS_WINDOWS, output_path=figure_dir / "stress_overview.png")
    for window in DEFAULT_CRISIS_WINDOWS:
        plot_crisis_window_dashboard(
            panel=panel,
            window=window,
            output_path=figure_dir / f"{window.name}_dashboard.png",
        )

    print("Step 2 complete.")
    print(f"Raw data: {raw_data_path}")
    print(f"EDA panel shape: {panel.shape}")
    print("\nData quality:")
    print(data_quality.to_string(index=False))
    print("\nCrisis summary:")
    print(
        crisis_summary[
            [
                "window",
                "spy_total_return",
                "spy_annualized_vol",
                "spy_max_drawdown",
                "max_vix",
                "PC1_mean",
                "PC2_mean",
                "PC3_mean",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print(f"\nFigures saved to: {figure_dir}")


def _latest_raw_market_data_path() -> Path:
    candidates = sorted(DATA_RAW_DIR.glob("market_data_*.parquet"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            "No raw market data parquet found. Run scripts/01_svd_pca_analysis.py before Step 2."
        )
    return candidates[0]


if __name__ == "__main__":
    main()
