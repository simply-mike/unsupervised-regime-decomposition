from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

from regime_decomposition.clustering import (  # noqa: E402
    DEFAULT_CLUSTER_FEATURES,
    build_clustering_matrix,
    fit_kmeans_regimes,
    plot_kmeans_model_selection,
    plot_regime_scatter,
    plot_return_distribution_by_regime,
    plot_spy_regime_timeline,
    run_hierarchical_sanity_check,
    summarize_regime_distribution_by_window,
)
from regime_decomposition.config import DATA_PROCESSED_DIR, FIGURES_DIR, TABLES_DIR  # noqa: E402
from regime_decomposition.eda import DEFAULT_CRISIS_WINDOWS  # noqa: E402
from regime_decomposition.visualization import set_plot_style  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 3: PCA-space K-Means and hierarchical clustering.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DATA_PROCESSED_DIR,
        help="Directory containing eda_panel.parquet from Step 2.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of K-Means regimes. Use 4 by default for bull/bear/high-vol/chop decomposition.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_CLUSTER_FEATURES),
        help="Feature columns used for clustering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_plot_style()

    panel_path = args.processed_dir / "eda_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError("Missing eda_panel.parquet. Run scripts/02_eda_preprocessing.py first.")

    panel = pd.read_parquet(panel_path)
    model_matrix = build_clustering_matrix(panel=panel, feature_columns=tuple(args.features))
    result = fit_kmeans_regimes(panel=panel, model_matrix=model_matrix, selected_k=args.k)
    hierarchical_labels, adjusted_rand = run_hierarchical_sanity_check(
        model_matrix=model_matrix,
        kmeans_labels=result.labels,
    )
    crisis_regime_distribution = summarize_regime_distribution_by_window(
        labels=result.labels,
        windows=DEFAULT_CRISIS_WINDOWS,
        cluster_summary=result.cluster_summary,
    )

    labeled_panel = panel.reindex(result.labels.index).copy()
    labeled_panel["kmeans_regime"] = result.labels
    labeled_panel["hierarchical_cluster"] = hierarchical_labels

    table_dir = TABLES_DIR
    figure_dir = FIGURES_DIR / "step3"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    model_matrix.to_parquet(args.processed_dir / "clustering_model_matrix.parquet")
    result.labels.to_frame().to_parquet(args.processed_dir / "kmeans_regime_labels.parquet")
    labeled_panel.to_parquet(args.processed_dir / "regime_labeled_panel.parquet")
    result.model_selection.to_csv(table_dir / "step3_kmeans_model_selection.csv", index=False)
    result.cluster_summary.to_csv(table_dir / "step3_kmeans_cluster_interpretation.csv", index=False)
    crisis_regime_distribution.to_csv(table_dir / "step3_crisis_regime_distribution.csv", index=False)
    pd.DataFrame(
        [
            {
                "selected_k": result.selected_k,
                "hierarchical_adjusted_rand_vs_kmeans": adjusted_rand,
                "features": ", ".join(args.features),
            }
        ]
    ).to_csv(table_dir / "step3_hierarchical_sanity_check.csv", index=False)

    plot_kmeans_model_selection(result.model_selection, figure_dir / "kmeans_model_selection.png")
    plot_regime_scatter(panel, result.labels, result.cluster_summary, figure_dir / "pc1_pc2_regime_scatter.png")
    plot_spy_regime_timeline(panel, result.labels, result.cluster_summary, figure_dir / "spy_regime_timeline.png")
    plot_return_distribution_by_regime(
        panel,
        result.labels,
        result.cluster_summary,
        figure_dir / "return_distribution_by_regime.png",
    )

    print("Step 3 complete.")
    print(f"Selected k: {result.selected_k}")
    print(f"Features: {', '.join(args.features)}")
    print(f"Hierarchical adjusted Rand vs K-Means: {adjusted_rand:.4f}")
    print("\nModel selection:")
    print(result.model_selection.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nCluster interpretation:")
    display_columns = [
        "regime",
        "regime_name",
        "count",
        "pct_obs",
        "spy_mean_daily_return",
        "spy_annualized_return",
        "spy_annualized_vol",
        "mean_vix",
        "PC1_mean",
        "PC2_mean",
        "PC3_mean",
        "stress_score",
    ]
    print(result.cluster_summary[display_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nCrisis window regime distribution:")
    print(crisis_regime_distribution.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nFigures saved to: {figure_dir}")


if __name__ == "__main__":
    main()
