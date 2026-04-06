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

from regime_decomposition.config import (  # noqa: E402
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    MarketDataConfig,
)
from regime_decomposition.data import load_or_download_market_data  # noqa: E402
from regime_decomposition.features import build_feature_matrix, build_returns_matrix  # noqa: E402
from regime_decomposition.svd_pca import run_svd_pca, summarize_pc_interpretation  # noqa: E402
from regime_decomposition.visualization import (  # noqa: E402
    plot_component_loadings,
    plot_cumulative_returns,
    plot_explained_variance,
    plot_pc_scatter,
    plot_pc_scores,
    plot_singular_values,
    set_plot_style,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1: data collection and SVD/PCA analysis.")
    parser.add_argument("--start", default="2000-01-01", help="Start date passed to yfinance.")
    parser.add_argument("--end", default=None, help="End date passed to yfinance. Defaults to latest available.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["SPY", "QQQ", "IWM", "TLT", "GLD", "^VIX"],
        help="Yahoo tickers to download. Include ^VIX for volatility context.",
    )
    parser.add_argument(
        "--return-tickers",
        nargs="+",
        default=["SPY", "QQQ", "IWM", "TLT", "GLD"],
        help="Assets included in the T x N returns matrix.",
    )
    parser.add_argument("--force-download", action="store_true", help="Ignore cached parquet and redownload data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_plot_style()

    config = MarketDataConfig(tickers=tuple(args.tickers), start=args.start, end=args.end)
    suffix_end = args.end or "latest"
    raw_path = DATA_RAW_DIR / f"market_data_{args.start}_{suffix_end}.parquet"

    market_data = load_or_download_market_data(
        config=config,
        output_path=raw_path,
        force_download=args.force_download,
    )
    returns = build_returns_matrix(market_data, return_tickers=tuple(args.return_tickers))
    features = build_feature_matrix(market_data)
    svd_pca = run_svd_pca(returns)
    pc_summary = summarize_pc_interpretation(svd_pca, n_components=min(3, len(args.return_tickers)))

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    returns.to_parquet(DATA_PROCESSED_DIR / "returns_matrix.parquet")
    returns.to_csv(DATA_PROCESSED_DIR / "returns_matrix.csv")
    features.to_parquet(DATA_PROCESSED_DIR / "feature_matrix.parquet")
    features.to_csv(DATA_PROCESSED_DIR / "feature_matrix.csv")
    svd_pca.component_loadings.to_csv(TABLES_DIR / "step1_pca_component_loadings.csv")
    svd_pca.component_scores.to_parquet(DATA_PROCESSED_DIR / "pca_component_scores.parquet")
    pc_summary.to_csv(TABLES_DIR / "step1_pc_interpretation.csv", index=False)

    figure_dir = FIGURES_DIR / "step1"
    plot_cumulative_returns(returns, figure_dir / "cumulative_log_returns.png")
    plot_singular_values(svd_pca, figure_dir / "svd_singular_values.png")
    plot_explained_variance(svd_pca, figure_dir / "pca_explained_variance.png")
    plot_component_loadings(svd_pca, figure_dir / "pca_component_loadings.png")
    plot_pc_scores(svd_pca, figure_dir / "pca_scores_timeseries.png")
    plot_pc_scatter(svd_pca, figure_dir / "pca_pc1_pc2_scatter.png")

    print("Step 1 complete.")
    print(f"Raw data: {raw_path}")
    print(f"Returns matrix shape: {returns.shape}")
    print(f"Feature matrix shape: {features.shape}")
    print("Explained variance:")
    print(svd_pca.explained_variance_ratio.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nPC interpretation:")
    print(pc_summary.to_string(index=False))
    print(f"\nFigures saved to: {figure_dir}")


if __name__ == "__main__":
    main()
