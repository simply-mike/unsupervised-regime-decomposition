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
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from regime_decomposition.clustering import (  # noqa: E402
    DEFAULT_CLUSTER_FEATURES,
    build_clustering_matrix,
    plot_regime_scatter,
    plot_return_distribution_by_regime,
    plot_spy_regime_timeline,
    summarize_regime_distribution_by_window,
)
from regime_decomposition.config import DATA_PROCESSED_DIR, FIGURES_DIR, TABLES_DIR  # noqa: E402
from regime_decomposition.eda import DEFAULT_CRISIS_WINDOWS  # noqa: E402
from regime_decomposition.gmm import (  # noqa: E402
    fit_gmm_regimes,
    plot_gmm_confidence,
    plot_gmm_model_selection,
    plot_gmm_probability_timeline,
    summarize_regime_probability_by_window,
)
from regime_decomposition.visualization import set_plot_style  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 4: full-covariance Gaussian Mixture regimes.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DATA_PROCESSED_DIR,
        help="Directory containing eda_panel.parquet from Step 2.",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=None,
        help="Optional fixed GMM component count. Defaults to the lowest BIC over 2..8.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_CLUSTER_FEATURES),
        help="Feature columns used for GMM fitting.",
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
    result = fit_gmm_regimes(panel=panel, model_matrix=model_matrix, selected_components=args.components)

    crisis_regime_distribution = summarize_regime_distribution_by_window(
        labels=result.labels.rename("regime"),
        windows=DEFAULT_CRISIS_WINDOWS,
        cluster_summary=result.regime_summary,
    )
    crisis_regime_probabilities = summarize_regime_probability_by_window(
        probabilities=result.probabilities,
        windows=DEFAULT_CRISIS_WINDOWS,
        regime_summary=result.regime_summary,
    )
    comparison = _compare_with_kmeans(args.processed_dir, result.labels)

    labeled_panel = panel.reindex(result.labels.index).copy()
    labeled_panel["gmm_regime"] = result.labels
    labeled_panel["gmm_max_probability"] = result.probabilities.max(axis=1)

    table_dir = TABLES_DIR
    figure_dir = FIGURES_DIR / "step4"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    result.labels.to_frame().to_parquet(args.processed_dir / "gmm_regime_labels.parquet")
    result.probabilities.to_parquet(args.processed_dir / "gmm_regime_probabilities.parquet")
    labeled_panel.to_parquet(args.processed_dir / "gmm_labeled_panel.parquet")
    result.model_selection.to_csv(table_dir / "step4_gmm_model_selection.csv", index=False)
    result.regime_summary.to_csv(table_dir / "step4_gmm_regime_interpretation.csv", index=False)
    crisis_regime_distribution.to_csv(table_dir / "step4_gmm_crisis_regime_distribution.csv", index=False)
    crisis_regime_probabilities.to_csv(table_dir / "step4_gmm_crisis_regime_probabilities.csv", index=False)
    comparison.to_csv(table_dir / "step4_gmm_vs_kmeans_comparison.csv", index=False)

    plot_gmm_model_selection(result.model_selection, figure_dir / "gmm_model_selection.png")
    plot_regime_scatter(
        panel,
        result.labels.rename("regime"),
        result.regime_summary,
        figure_dir / "pc1_pc2_gmm_regime_scatter.png",
        title="PCA State Space Colored by GMM Regime",
    )
    plot_spy_regime_timeline(
        panel,
        result.labels.rename("regime"),
        result.regime_summary,
        figure_dir / "spy_gmm_regime_timeline.png",
        title="SPY Price With GMM Regime Background",
    )
    plot_return_distribution_by_regime(
        panel,
        result.labels.rename("regime"),
        result.regime_summary,
        figure_dir / "return_distribution_by_gmm_regime.png",
    )
    plot_gmm_probability_timeline(result.probabilities, result.regime_summary, figure_dir / "gmm_probability_timeline.png")
    plot_gmm_confidence(result.probabilities, figure_dir / "gmm_confidence.png")

    print("Step 4 complete.")
    print(f"Selected components: {result.selected_components}")
    print(f"Features: {', '.join(args.features)}")
    print("\nModel selection:")
    print(result.model_selection.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nGMM regime interpretation:")
    display_columns = [
        "regime",
        "regime_name",
        "count",
        "pct_obs",
        "spy_mean_daily_return",
        "spy_annualized_vol",
        "mean_vix",
        "PC1_mean",
        "PC2_mean",
        "PC3_mean",
        "avg_max_probability",
        "avg_probability_entropy",
        "stress_score",
    ]
    print(result.regime_summary[display_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nCrisis window hard-label distribution:")
    print(crisis_regime_distribution.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nGMM vs K-Means:")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nFigures saved to: {figure_dir}")


def _compare_with_kmeans(processed_dir: Path, gmm_labels: pd.Series) -> pd.DataFrame:
    kmeans_path = processed_dir / "kmeans_regime_labels.parquet"
    if not kmeans_path.exists():
        return pd.DataFrame([{"comparison": "kmeans_labels_missing", "adjusted_rand": float("nan")}])

    kmeans_labels = pd.read_parquet(kmeans_path).iloc[:, 0].reindex(gmm_labels.index).dropna()
    aligned_gmm = gmm_labels.reindex(kmeans_labels.index)
    return pd.DataFrame(
        [
            {
                "comparison": "gmm_vs_kmeans",
                "adjusted_rand": float(adjusted_rand_score(kmeans_labels, aligned_gmm)),
                "n_aligned_observations": int(len(kmeans_labels)),
            }
        ]
    )


if __name__ == "__main__":
    main()
