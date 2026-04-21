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
    summarize_regime_distribution_by_window,
)
from regime_decomposition.config import DATA_PROCESSED_DIR, FIGURES_DIR, TABLES_DIR  # noqa: E402
from regime_decomposition.eda import DEFAULT_CRISIS_WINDOWS  # noqa: E402
from regime_decomposition.hmm import (  # noqa: E402
    fit_hmm_regimes,
    plot_hmm_model_selection,
    plot_hmm_spy_timeline,
    plot_hmm_state_probabilities,
    plot_hmm_transition_matrix,
    summarize_hmm_probability_by_window,
)
from regime_decomposition.visualization import set_plot_style  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 5: Gaussian HMM regimes with transition probabilities.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DATA_PROCESSED_DIR,
        help="Directory containing eda_panel.parquet from Step 2.",
    )
    parser.add_argument(
        "--states",
        type=int,
        default=4,
        help="Number of HMM states. Defaults to 4 for interpretable calm/chop/stress/crisis dynamics.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_CLUSTER_FEATURES),
        help="Feature columns used for HMM fitting.",
    )
    parser.add_argument("--restarts", type=int, default=5, help="EM restarts per state count.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_plot_style()

    panel_path = args.processed_dir / "eda_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError("Missing eda_panel.parquet. Run scripts/02_eda_preprocessing.py first.")

    panel = pd.read_parquet(panel_path)
    model_matrix = build_clustering_matrix(panel=panel, feature_columns=tuple(args.features))
    result = fit_hmm_regimes(
        panel=panel,
        model_matrix=model_matrix,
        selected_states=args.states,
        n_restarts=args.restarts,
    )

    crisis_state_distribution = summarize_regime_distribution_by_window(
        labels=result.labels.rename("regime"),
        windows=DEFAULT_CRISIS_WINDOWS,
        cluster_summary=result.regime_summary,
    )
    crisis_state_probabilities = summarize_hmm_probability_by_window(
        probabilities=result.probabilities,
        windows=DEFAULT_CRISIS_WINDOWS,
        regime_summary=result.regime_summary,
    )
    comparison = _compare_with_prior_labels(args.processed_dir, result.labels)

    labeled_panel = panel.reindex(result.labels.index).copy()
    labeled_panel["hmm_state"] = result.labels
    labeled_panel["hmm_max_probability"] = result.probabilities.max(axis=1)

    table_dir = TABLES_DIR
    figure_dir = FIGURES_DIR / "step5"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    result.labels.to_frame().to_parquet(args.processed_dir / "hmm_state_labels.parquet")
    result.probabilities.to_parquet(args.processed_dir / "hmm_state_probabilities.parquet")
    labeled_panel.to_parquet(args.processed_dir / "hmm_labeled_panel.parquet")
    result.model_selection.to_csv(table_dir / "step5_hmm_model_selection.csv", index=False)
    result.regime_summary.to_csv(table_dir / "step5_hmm_state_interpretation.csv", index=False)
    result.transition_matrix.to_csv(table_dir / "step5_hmm_transition_matrix.csv")
    result.expected_durations.to_csv(table_dir / "step5_hmm_expected_durations.csv", index=False)
    crisis_state_distribution.to_csv(table_dir / "step5_hmm_crisis_state_distribution.csv", index=False)
    crisis_state_probabilities.to_csv(table_dir / "step5_hmm_crisis_state_probabilities.csv", index=False)
    comparison.to_csv(table_dir / "step5_hmm_vs_prior_models.csv", index=False)

    plot_hmm_model_selection(result.model_selection, figure_dir / "hmm_model_selection.png")
    plot_hmm_transition_matrix(result.transition_matrix, figure_dir / "hmm_transition_matrix.png")
    plot_hmm_state_probabilities(result.probabilities, result.regime_summary, figure_dir / "hmm_state_probabilities.png")
    plot_hmm_spy_timeline(panel, result.labels.rename("regime"), result.regime_summary, figure_dir / "spy_hmm_state_timeline.png")

    print("Step 5 complete.")
    print(f"Selected states: {result.selected_states}")
    print(f"Features: {', '.join(args.features)}")
    print("\nModel selection:")
    print(result.model_selection.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nHMM state interpretation:")
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
        "avg_max_state_probability",
        "stress_score",
    ]
    print(result.regime_summary[display_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nExpected durations:")
    print(result.expected_durations.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nTransition matrix:")
    print(result.transition_matrix.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nCrisis window hard-label distribution:")
    print(crisis_state_distribution.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nHMM vs prior models:")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nFigures saved to: {figure_dir}")


def _compare_with_prior_labels(processed_dir: Path, hmm_labels: pd.Series) -> pd.DataFrame:
    rows = []
    candidates = {
        "hmm_vs_kmeans": processed_dir / "kmeans_regime_labels.parquet",
        "hmm_vs_gmm": processed_dir / "gmm_regime_labels.parquet",
    }
    for comparison, path in candidates.items():
        if not path.exists():
            rows.append({"comparison": comparison, "adjusted_rand": float("nan"), "n_aligned_observations": 0})
            continue
        labels = pd.read_parquet(path).iloc[:, 0].reindex(hmm_labels.index).dropna()
        aligned_hmm = hmm_labels.reindex(labels.index)
        rows.append(
            {
                "comparison": comparison,
                "adjusted_rand": float(adjusted_rand_score(labels, aligned_hmm)),
                "n_aligned_observations": int(len(labels)),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
