from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from regime_decomposition.clustering import add_regime_names, summarize_cluster_behavior


@dataclass(frozen=True)
class GmmRegimeResult:
    model_matrix: pd.DataFrame
    scaled_matrix: pd.DataFrame
    labels: pd.Series
    probabilities: pd.DataFrame
    model_selection: pd.DataFrame
    regime_summary: pd.DataFrame
    scaler: StandardScaler
    model: GaussianMixture
    selected_components: int


def run_gmm_model_selection(
    model_matrix: pd.DataFrame,
    n_components_values: range = range(2, 9),
    covariance_type: str = "full",
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[int, GaussianMixture], pd.DataFrame, StandardScaler]:
    """Fit Gaussian mixture models across component counts and compute AIC/BIC."""

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(model_matrix)
    scaled_matrix = pd.DataFrame(scaled_values, index=model_matrix.index, columns=model_matrix.columns)

    rows = []
    models: dict[int, GaussianMixture] = {}
    for n_components in n_components_values:
        model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            n_init=10,
            max_iter=1000,
            reg_covar=1e-5,
            random_state=random_state,
        )
        model.fit(scaled_matrix)
        labels = model.predict(scaled_matrix)
        counts = pd.Series(labels).value_counts()
        models[n_components] = model
        rows.append(
            {
                "n_components": n_components,
                "aic": float(model.aic(scaled_matrix)),
                "bic": float(model.bic(scaled_matrix)),
                "lower_bound": float(model.lower_bound_),
                "converged": bool(model.converged_),
                "n_iter": int(model.n_iter_),
                "min_component_size": int(counts.min()),
                "max_component_size": int(counts.max()),
            }
        )

    return pd.DataFrame(rows), models, scaled_matrix, scaler


def fit_gmm_regimes(
    panel: pd.DataFrame,
    model_matrix: pd.DataFrame,
    selected_components: int | None = None,
    n_components_values: range = range(2, 9),
    covariance_type: str = "full",
    random_state: int = 42,
) -> GmmRegimeResult:
    """Fit a full-covariance GMM and remap components from calm to stress."""

    model_selection, models, scaled_matrix, scaler = run_gmm_model_selection(
        model_matrix=model_matrix,
        n_components_values=n_components_values,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    if selected_components is None:
        selected_components = int(model_selection.sort_values(["bic", "n_components"]).iloc[0]["n_components"])

    model = models[selected_components]
    raw_labels = pd.Series(model.predict(scaled_matrix), index=model_matrix.index, name="raw_component")
    raw_summary = summarize_cluster_behavior(panel=panel, labels=raw_labels, label_name="raw_component")
    label_mapping = _build_component_mapping(raw_summary)

    labels = raw_labels.map(label_mapping).astype(int).rename("gmm_regime")
    probabilities = _remap_probabilities(
        raw_probabilities=model.predict_proba(scaled_matrix),
        index=model_matrix.index,
        label_mapping=label_mapping,
    )

    regime_summary = summarize_cluster_behavior(panel=panel, labels=labels.rename("regime"), label_name="regime")
    regime_summary = add_regime_names(regime_summary)
    regime_summary = add_probability_diagnostics(regime_summary, probabilities, labels)

    return GmmRegimeResult(
        model_matrix=model_matrix,
        scaled_matrix=scaled_matrix,
        labels=labels,
        probabilities=probabilities,
        model_selection=model_selection,
        regime_summary=regime_summary,
        scaler=scaler,
        model=model,
        selected_components=selected_components,
    )


def add_probability_diagnostics(
    regime_summary: pd.DataFrame,
    probabilities: pd.DataFrame,
    labels: pd.Series,
) -> pd.DataFrame:
    """Add posterior confidence diagnostics by assigned regime."""

    diagnostics = []
    max_probability = probabilities.max(axis=1)
    entropy = _normalized_entropy(probabilities)
    for regime in sorted(labels.unique()):
        mask = labels == regime
        diagnostics.append(
            {
                "regime": int(regime),
                "avg_max_probability": float(max_probability.loc[mask].mean()),
                "avg_probability_entropy": float(entropy.loc[mask].mean()),
            }
        )
    diagnostics_df = pd.DataFrame(diagnostics)
    return regime_summary.merge(diagnostics_df, on="regime", how="left")


def summarize_regime_probability_by_window(
    probabilities: pd.DataFrame,
    windows: tuple,
    regime_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize average posterior probabilities inside known stress windows."""

    name_map = regime_summary.set_index("regime")["regime_name"].to_dict()
    rows = []
    for window in windows:
        window_probabilities = probabilities.loc[window.start : window.end]
        if window_probabilities.empty:
            continue
        means = window_probabilities.mean().sort_index()
        for column, value in means.items():
            regime = int(column.removeprefix("regime_"))
            rows.append(
                {
                    "window": window.name,
                    "label": window.label,
                    "regime": regime,
                    "regime_name": name_map[regime],
                    "mean_probability": float(value),
                }
            )
    return pd.DataFrame(rows)


def plot_gmm_model_selection(model_selection: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(model_selection["n_components"], model_selection["aic"], marker="o", label="AIC", color="#4C78A8")
    ax.plot(model_selection["n_components"], model_selection["bic"], marker="o", label="BIC", color="#E15759")
    ax.set_title("GMM Model Selection")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Information criterion")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_gmm_probability_timeline(
    probabilities: pd.DataFrame,
    regime_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    labels = regime_summary["regime_name"].tolist()
    colors = plt.get_cmap("tab10").colors[: len(labels)]
    ax.stackplot(probabilities.index, probabilities.T.to_numpy(), labels=labels, colors=colors, alpha=0.85)
    ax.set_title("GMM Posterior Regime Probabilities")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.legend(loc="upper left", ncols=2, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_gmm_confidence(
    probabilities: pd.DataFrame,
    output_path: Path,
    rolling_window: int = 20,
) -> None:
    max_probability = probabilities.max(axis=1)
    entropy = _normalized_entropy(probabilities)

    fig, ax = plt.subplots(figsize=(14, 5))
    max_probability.rolling(rolling_window).mean().plot(ax=ax, color="#4C78A8", label="20d avg max posterior")
    entropy.rolling(rolling_window).mean().plot(ax=ax, color="#E15759", label="20d avg normalized entropy")
    ax.set_title("GMM Regime Confidence")
    ax.set_ylabel("Probability / entropy")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.legend(loc="upper left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _build_component_mapping(raw_summary: pd.DataFrame) -> dict[int, int]:
    ordered = raw_summary.sort_values("stress_score", ascending=True)["raw_component"].tolist()
    return {int(raw_component): regime_id for regime_id, raw_component in enumerate(ordered)}


def _remap_probabilities(
    raw_probabilities: np.ndarray,
    index: pd.Index,
    label_mapping: dict[int, int],
) -> pd.DataFrame:
    probabilities = pd.DataFrame(index=index)
    for raw_component, regime in sorted(label_mapping.items(), key=lambda item: item[1]):
        probabilities[f"regime_{regime}"] = raw_probabilities[:, raw_component]
    probabilities.index.name = "date"
    return probabilities


def _normalized_entropy(probabilities: pd.DataFrame) -> pd.Series:
    clipped = probabilities.clip(lower=1e-12)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    max_entropy = np.log(probabilities.shape[1])
    return (entropy / max_entropy).rename("normalized_entropy")
