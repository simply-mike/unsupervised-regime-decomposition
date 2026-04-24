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
    summarize_regime_distribution_by_window,
)
from regime_decomposition.config import DATA_PROCESSED_DIR, FIGURES_DIR, TABLES_DIR  # noqa: E402
from regime_decomposition.eda import DEFAULT_CRISIS_WINDOWS  # noqa: E402
from regime_decomposition.hmm import summarize_hmm_probability_by_window  # noqa: E402
from regime_decomposition.visualization import set_plot_style  # noqa: E402
from regime_decomposition.walk_forward import (  # noqa: E402
    plot_walk_forward_probabilities,
    plot_walk_forward_spy_timeline,
    plot_walk_forward_state_distribution,
    run_walk_forward_hmm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 6: expanding-window HMM filtered inference.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DATA_PROCESSED_DIR,
        help="Directory containing eda_panel.parquet from Step 2.",
    )
    parser.add_argument("--states", type=int, default=4, help="Number of HMM states.")
    parser.add_argument("--min-train-size", type=int, default=1260, help="Initial expanding train size in days.")
    parser.add_argument("--refit-frequency", type=int, default=63, help="Calendar index step between refits.")
    parser.add_argument("--test-size", type=int, default=63, help="Number of OOS days inferred after each refit.")
    parser.add_argument("--restarts", type=int, default=3, help="EM restarts per refit.")
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_CLUSTER_FEATURES),
        help="Feature columns used for HMM fitting.",
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
    result = run_walk_forward_hmm(
        panel=panel,
        model_matrix=model_matrix,
        n_states=args.states,
        min_train_size=args.min_train_size,
        refit_frequency=args.refit_frequency,
        test_size=args.test_size,
        n_restarts=args.restarts,
    )

    labeled_panel = panel.reindex(result.labels.index).copy()
    labeled_panel["walk_forward_hmm_state"] = result.labels
    labeled_panel["walk_forward_hmm_max_probability"] = result.probabilities.max(axis=1)
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

    table_dir = TABLES_DIR
    figure_dir = FIGURES_DIR / "step6"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    result.labels.to_frame().to_parquet(args.processed_dir / "walk_forward_hmm_state_labels.parquet")
    result.probabilities.to_parquet(args.processed_dir / "walk_forward_hmm_state_probabilities.parquet")
    labeled_panel.to_parquet(args.processed_dir / "walk_forward_hmm_labeled_panel.parquet")
    result.diagnostics.to_csv(table_dir / "step6_walk_forward_hmm_diagnostics.csv", index=False)
    result.regime_summary.to_csv(table_dir / "step6_walk_forward_hmm_state_interpretation.csv", index=False)
    crisis_state_distribution.to_csv(table_dir / "step6_walk_forward_hmm_crisis_state_distribution.csv", index=False)
    crisis_state_probabilities.to_csv(table_dir / "step6_walk_forward_hmm_crisis_state_probabilities.csv", index=False)

    plot_walk_forward_probabilities(
        result.probabilities,
        result.regime_summary,
        figure_dir / "walk_forward_hmm_probabilities.png",
    )
    plot_walk_forward_spy_timeline(
        panel,
        result.labels,
        result.regime_summary,
        figure_dir / "spy_walk_forward_hmm_timeline.png",
    )
    plot_walk_forward_state_distribution(
        result.labels,
        result.regime_summary,
        figure_dir / "walk_forward_hmm_state_distribution.png",
    )

    print("Step 6 complete.")
    print(f"OOS observations: {len(result.labels)}")
    print(f"Folds: {len(result.diagnostics)}")
    print(f"Features: {', '.join(args.features)}")
    print("\nWalk-forward state interpretation:")
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
        "avg_max_filtered_probability",
        "stress_score",
    ]
    print(result.regime_summary[display_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nDiagnostics head:")
    print(result.diagnostics.head().to_string(index=False))
    print("\nOOS crisis window state distribution:")
    print(crisis_state_distribution.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nFigures saved to: {figure_dir}")


if __name__ == "__main__":
    main()
