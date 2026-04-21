from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from regime_decomposition.clustering import add_regime_names, summarize_cluster_behavior


@dataclass(frozen=True)
class HmmRegimeResult:
    model_matrix: pd.DataFrame
    scaled_matrix: pd.DataFrame
    labels: pd.Series
    probabilities: pd.DataFrame
    model_selection: pd.DataFrame
    regime_summary: pd.DataFrame
    transition_matrix: pd.DataFrame
    expected_durations: pd.DataFrame
    scaler: StandardScaler
    model: GaussianHMM
    selected_states: int


def run_hmm_model_selection(
    model_matrix: pd.DataFrame,
    n_states_values: range = range(2, 7),
    covariance_type: str = "full",
    random_state: int = 42,
    n_restarts: int = 5,
) -> tuple[pd.DataFrame, dict[int, GaussianHMM], pd.DataFrame, StandardScaler]:
    """Fit Gaussian HMMs across state counts and keep the best restart by log-likelihood."""

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(model_matrix)
    scaled_matrix = pd.DataFrame(scaled_values, index=model_matrix.index, columns=model_matrix.columns)

    rows = []
    models: dict[int, GaussianHMM] = {}
    for n_states in n_states_values:
        best_model: GaussianHMM | None = None
        best_score = -np.inf
        for restart in range(n_restarts):
            model = GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                n_iter=1000,
                tol=1e-4,
                min_covar=1e-4,
                random_state=random_state + restart,
                implementation="log",
            )
            model.fit(scaled_matrix)
            score = float(model.score(scaled_matrix))
            if score > best_score:
                best_score = score
                best_model = model

        if best_model is None:
            raise RuntimeError(f"Could not fit HMM with {n_states} states.")

        labels = best_model.predict(scaled_matrix)
        counts = pd.Series(labels).value_counts()
        models[n_states] = best_model
        rows.append(
            {
                "n_states": n_states,
                "log_likelihood": best_score,
                "aic": float(best_model.aic(scaled_matrix)),
                "bic": float(best_model.bic(scaled_matrix)),
                "converged": bool(best_model.monitor_.converged),
                "n_iter": int(best_model.monitor_.iter),
                "min_state_size": int(counts.min()),
                "max_state_size": int(counts.max()),
            }
        )

    return pd.DataFrame(rows), models, scaled_matrix, scaler


def fit_hmm_regimes(
    panel: pd.DataFrame,
    model_matrix: pd.DataFrame,
    selected_states: int | None = None,
    n_states_values: range = range(2, 7),
    covariance_type: str = "full",
    random_state: int = 42,
    n_restarts: int = 5,
) -> HmmRegimeResult:
    """Fit a Gaussian HMM and remap hidden states from calm to stress."""

    model_selection, models, scaled_matrix, scaler = run_hmm_model_selection(
        model_matrix=model_matrix,
        n_states_values=n_states_values,
        covariance_type=covariance_type,
        random_state=random_state,
        n_restarts=n_restarts,
    )
    if selected_states is None:
        selected_states = int(model_selection.sort_values(["bic", "n_states"]).iloc[0]["n_states"])

    model = models[selected_states]
    raw_labels = pd.Series(model.predict(scaled_matrix), index=model_matrix.index, name="raw_state")
    raw_summary = summarize_cluster_behavior(panel=panel, labels=raw_labels, label_name="raw_state")
    state_mapping = _build_state_mapping(raw_summary)

    labels = raw_labels.map(state_mapping).astype(int).rename("hmm_state")
    probabilities = _remap_probabilities(
        raw_probabilities=model.predict_proba(scaled_matrix),
        index=model_matrix.index,
        state_mapping=state_mapping,
    )
    transition_matrix = _remap_transition_matrix(model.transmat_, state_mapping)
    regime_summary = summarize_cluster_behavior(panel=panel, labels=labels.rename("regime"), label_name="regime")
    regime_summary = add_regime_names(regime_summary)
    regime_summary = add_hmm_probability_diagnostics(regime_summary, probabilities, labels)
    expected_durations = compute_expected_durations(transition_matrix, regime_summary)

    return HmmRegimeResult(
        model_matrix=model_matrix,
        scaled_matrix=scaled_matrix,
        labels=labels,
        probabilities=probabilities,
        model_selection=model_selection,
        regime_summary=regime_summary,
        transition_matrix=transition_matrix,
        expected_durations=expected_durations,
        scaler=scaler,
        model=model,
        selected_states=selected_states,
    )


def add_hmm_probability_diagnostics(
    regime_summary: pd.DataFrame,
    probabilities: pd.DataFrame,
    labels: pd.Series,
) -> pd.DataFrame:
    max_probability = probabilities.max(axis=1)
    entropy = _normalized_entropy(probabilities)
    rows = []
    for regime in sorted(labels.unique()):
        mask = labels == regime
        rows.append(
            {
                "regime": int(regime),
                "avg_max_state_probability": float(max_probability.loc[mask].mean()),
                "avg_state_probability_entropy": float(entropy.loc[mask].mean()),
            }
        )
    return regime_summary.merge(pd.DataFrame(rows), on="regime", how="left")


def compute_expected_durations(transition_matrix: pd.DataFrame, regime_summary: pd.DataFrame) -> pd.DataFrame:
    name_map = regime_summary.set_index("regime")["regime_name"].to_dict()
    rows = []
    for regime in transition_matrix.index:
        regime_id = int(str(regime).removeprefix("regime_"))
        stay_probability = float(transition_matrix.loc[regime, regime])
        expected_days = np.inf if np.isclose(stay_probability, 1.0) else 1.0 / (1.0 - stay_probability)
        rows.append(
            {
                "regime": regime_id,
                "regime_name": name_map[regime_id],
                "stay_probability": stay_probability,
                "expected_duration_days": float(expected_days),
            }
        )
    return pd.DataFrame(rows)


def summarize_hmm_probability_by_window(
    probabilities: pd.DataFrame,
    windows: tuple,
    regime_summary: pd.DataFrame,
) -> pd.DataFrame:
    name_map = regime_summary.set_index("regime")["regime_name"].to_dict()
    rows = []
    for window in windows:
        window_probabilities = probabilities.loc[window.start : window.end]
        if window_probabilities.empty:
            continue
        for column, value in window_probabilities.mean().sort_index().items():
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


def plot_hmm_model_selection(model_selection: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(model_selection["n_states"], model_selection["log_likelihood"], marker="o", color="#4C78A8")
    axes[0].set_title("HMM Log-Likelihood")
    axes[0].set_xlabel("Number of states")
    axes[0].set_ylabel("Log-likelihood")

    axes[1].plot(model_selection["n_states"], model_selection["aic"], marker="o", label="AIC", color="#59A14F")
    axes[1].plot(model_selection["n_states"], model_selection["bic"], marker="o", label="BIC", color="#E15759")
    axes[1].set_title("HMM Information Criteria")
    axes[1].set_xlabel("Number of states")
    axes[1].set_ylabel("Information criterion")
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_hmm_transition_matrix(transition_matrix: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(transition_matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("HMM Transition Matrix")
    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_hmm_state_probabilities(
    probabilities: pd.DataFrame,
    regime_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    labels = regime_summary["regime_name"].tolist()
    colors = plt.get_cmap("tab10").colors[: len(labels)]
    ax.stackplot(probabilities.index, probabilities.T.to_numpy(), labels=labels, colors=colors, alpha=0.85)
    ax.set_title("HMM Smoothed State Probabilities")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.legend(loc="upper left", ncols=2, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_hmm_spy_timeline(
    panel: pd.DataFrame,
    labels: pd.Series,
    regime_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    data = panel.reindex(labels.index).copy()
    data["regime"] = labels
    palette = _regime_palette(regime_summary)

    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.plot(data.index, data["spy_close"], color="black", linewidth=1.1, zorder=2)
    for regime, start, end in _regime_segments(data["regime"]):
        ax.axvspan(start, end, color=palette[regime], alpha=0.18, linewidth=0, zorder=0)
    ax.set_yscale("log")
    ax.set_title("SPY Price With HMM State Background")
    ax.set_ylabel("SPY close, log scale")
    ax.set_xlabel("")
    handles = [
        mpatches.Patch(color=palette[row.regime], alpha=0.35, label=row.regime_name)
        for row in regime_summary.itertuples(index=False)
    ]
    ax.legend(handles=handles, loc="upper left", ncols=2, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _build_state_mapping(raw_summary: pd.DataFrame) -> dict[int, int]:
    ordered = raw_summary.sort_values("stress_score", ascending=True)["raw_state"].tolist()
    return {int(raw_state): regime_id for regime_id, raw_state in enumerate(ordered)}


def _remap_probabilities(
    raw_probabilities: np.ndarray,
    index: pd.Index,
    state_mapping: dict[int, int],
) -> pd.DataFrame:
    probabilities = pd.DataFrame(index=index)
    for raw_state, regime in sorted(state_mapping.items(), key=lambda item: item[1]):
        probabilities[f"regime_{regime}"] = raw_probabilities[:, raw_state]
    probabilities.index.name = "date"
    return probabilities


def _remap_transition_matrix(raw_transition_matrix: np.ndarray, state_mapping: dict[int, int]) -> pd.DataFrame:
    n_states = len(state_mapping)
    matrix = np.zeros((n_states, n_states))
    for raw_from, regime_from in state_mapping.items():
        for raw_to, regime_to in state_mapping.items():
            matrix[regime_from, regime_to] = raw_transition_matrix[raw_from, raw_to]

    labels = [f"regime_{i}" for i in range(n_states)]
    return pd.DataFrame(matrix, index=labels, columns=labels)


def _normalized_entropy(probabilities: pd.DataFrame) -> pd.Series:
    clipped = probabilities.clip(lower=1e-12)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    max_entropy = np.log(probabilities.shape[1])
    return (entropy / max_entropy).rename("normalized_entropy")


def _regime_palette(regime_summary: pd.DataFrame) -> dict[int, tuple[float, float, float]]:
    colors = sns.color_palette("colorblind", n_colors=len(regime_summary))
    return {int(regime): colors[i] for i, regime in enumerate(regime_summary["regime"])}


def _regime_segments(regime_series: pd.Series) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    series = regime_series.sort_index()
    segments: list[tuple[int, pd.Timestamp, pd.Timestamp]] = []
    start = series.index[0]
    previous_date = series.index[0]
    previous_regime = int(series.iloc[0])

    for date, regime in series.iloc[1:].items():
        regime = int(regime)
        if regime != previous_regime:
            segments.append((previous_regime, start, previous_date))
            start = date
        previous_date = date
        previous_regime = regime
    segments.append((previous_regime, start, previous_date))
    return segments
