from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler

from regime_decomposition.clustering import add_regime_names, summarize_cluster_behavior


@dataclass(frozen=True)
class WalkForwardHmmResult:
    labels: pd.Series
    probabilities: pd.DataFrame
    diagnostics: pd.DataFrame
    regime_summary: pd.DataFrame


def run_walk_forward_hmm(
    panel: pd.DataFrame,
    model_matrix: pd.DataFrame,
    n_states: int = 4,
    min_train_size: int = 1260,
    refit_frequency: int = 63,
    test_size: int = 21,
    n_restarts: int = 3,
    random_state: int = 42,
) -> WalkForwardHmmResult:
    """Run expanding-window HMM inference with one-step recursive filtering.

    Each refit sees only observations before the test block. Within the test
    block, probabilities are updated forward one day at a time and do not use
    future test observations.
    """

    if min_train_size >= len(model_matrix):
        raise ValueError("min_train_size must be smaller than the model matrix length.")
    if refit_frequency != test_size:
        raise ValueError("refit_frequency and test_size must match for non-overlapping walk-forward inference.")

    probability_frames = []
    label_frames = []
    diagnostic_rows = []
    start = min_train_size
    fold = 0

    while start < len(model_matrix):
        train_end = start
        test_end = min(start + test_size, len(model_matrix))
        train_matrix = model_matrix.iloc[:train_end]
        test_matrix = model_matrix.iloc[train_end:test_end]
        if test_matrix.empty:
            break

        scaler = StandardScaler()
        train_scaled_values = scaler.fit_transform(train_matrix)
        train_scaled = pd.DataFrame(train_scaled_values, index=train_matrix.index, columns=train_matrix.columns)
        test_scaled = pd.DataFrame(scaler.transform(test_matrix), index=test_matrix.index, columns=test_matrix.columns)

        model = _fit_hmm_best_restart(
            train_scaled=train_scaled,
            n_states=n_states,
            random_state=random_state + fold * 100,
            n_restarts=n_restarts,
        )
        raw_train_labels = pd.Series(model.predict(train_scaled), index=train_scaled.index, name="raw_state")
        raw_summary = summarize_cluster_behavior(
            panel=panel.reindex(train_scaled.index),
            labels=raw_train_labels,
            label_name="raw_state",
        )
        state_mapping = _build_state_mapping(raw_summary=raw_summary, n_states=n_states)

        raw_train_probabilities = model.predict_proba(train_scaled)
        raw_prior = raw_train_probabilities[-1] @ model.transmat_
        filtered_probabilities = _filter_test_block(
            model=model,
            test_scaled=test_scaled,
            initial_raw_prior=raw_prior,
            state_mapping=state_mapping,
        )
        labels = filtered_probabilities.idxmax(axis=1).str.removeprefix("regime_").astype(int).rename("walk_forward_hmm_state")

        probability_frames.append(filtered_probabilities)
        label_frames.append(labels)
        diagnostic_rows.append(
            {
                "fold": fold,
                "train_start": train_matrix.index[0].date(),
                "train_end": train_matrix.index[-1].date(),
                "test_start": test_matrix.index[0].date(),
                "test_end": test_matrix.index[-1].date(),
                "train_size": len(train_matrix),
                "test_size": len(test_matrix),
                "log_likelihood_train": float(model.score(train_scaled)),
                "converged": bool(model.monitor_.converged),
                "n_iter": int(model.monitor_.iter),
            }
        )
        fold += 1
        start += refit_frequency

    probabilities = pd.concat(probability_frames).sort_index()
    labels = pd.concat(label_frames).sort_index()
    diagnostics = pd.DataFrame(diagnostic_rows)
    regime_summary = summarize_cluster_behavior(
        panel=panel.reindex(labels.index),
        labels=labels.rename("regime"),
        label_name="regime",
    )
    regime_summary = add_regime_names(regime_summary)
    regime_summary = _add_probability_diagnostics(regime_summary, probabilities, labels)

    return WalkForwardHmmResult(
        labels=labels,
        probabilities=probabilities,
        diagnostics=diagnostics,
        regime_summary=regime_summary,
    )


def plot_walk_forward_probabilities(
    probabilities: pd.DataFrame,
    regime_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    labels = regime_summary["regime_name"].tolist()
    colors = plt.get_cmap("tab10").colors[: len(labels)]
    ax.stackplot(probabilities.index, probabilities.T.to_numpy(), labels=labels, colors=colors, alpha=0.85)
    ax.set_title("Walk-Forward HMM Filtered State Probabilities")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.legend(loc="upper left", ncols=2, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_walk_forward_spy_timeline(
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
    ax.set_title("SPY Price With Walk-Forward HMM State Background")
    ax.set_ylabel("SPY close, log scale")
    ax.set_xlabel("")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_walk_forward_state_distribution(labels: pd.Series, regime_summary: pd.DataFrame, output_path: Path) -> None:
    counts = labels.value_counts(normalize=True).sort_index()
    name_map = regime_summary.set_index("regime")["regime_name"].to_dict()
    plot_df = counts.rename("pct_obs").reset_index()
    plot_df.columns = ["regime", "pct_obs"]
    plot_df["regime_name"] = plot_df["regime"].map(name_map)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_df, x="regime_name", y="pct_obs", ax=ax, color="#4C78A8")
    ax.set_title("Walk-Forward HMM State Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Share of OOS observations")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _fit_hmm_best_restart(
    train_scaled: pd.DataFrame,
    n_states: int,
    random_state: int,
    n_restarts: int,
) -> GaussianHMM:
    best_model: GaussianHMM | None = None
    best_score = -np.inf
    for restart in range(n_restarts):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=1000,
            tol=1e-4,
            min_covar=1e-4,
            random_state=random_state + restart,
            implementation="log",
        )
        model.fit(train_scaled)
        score = float(model.score(train_scaled))
        if score > best_score:
            best_model = model
            best_score = score
    if best_model is None:
        raise RuntimeError("HMM fitting failed for all restarts.")
    return best_model


def _filter_test_block(
    model: GaussianHMM,
    test_scaled: pd.DataFrame,
    initial_raw_prior: np.ndarray,
    state_mapping: dict[int, int],
) -> pd.DataFrame:
    raw_prior = np.asarray(initial_raw_prior, dtype=float)
    raw_prior = raw_prior / raw_prior.sum()

    rows = []
    for _, row in test_scaled.iterrows():
        log_emission = _log_emission_probabilities(model, row.to_numpy())
        log_posterior = np.log(raw_prior.clip(1e-12)) + log_emission
        normalizer = logsumexp(log_posterior)
        if np.isfinite(normalizer):
            posterior_raw = np.exp(log_posterior - normalizer)
        else:
            posterior_raw = raw_prior.copy()
        rows.append(_remap_probability_row(posterior_raw, state_mapping))
        raw_prior = posterior_raw @ model.transmat_
        raw_prior = raw_prior / raw_prior.sum()

    columns = [f"regime_{i}" for i in range(len(state_mapping))]
    probabilities = pd.DataFrame(rows, index=test_scaled.index, columns=columns)
    probabilities.index.name = "date"
    return probabilities


def _log_emission_probabilities(model: GaussianHMM, x: np.ndarray) -> np.ndarray:
    log_probabilities = []
    for state in range(model.n_components):
        covariance = np.asarray(model.covars_[state], dtype=float)
        covariance = covariance + np.eye(covariance.shape[0]) * 1e-6
        log_probabilities.append(
            multivariate_normal.logpdf(
                x,
                mean=model.means_[state],
                cov=covariance,
                allow_singular=True,
            )
        )
    return np.asarray(log_probabilities)


def _build_state_mapping(raw_summary: pd.DataFrame, n_states: int) -> dict[int, int]:
    ordered = [int(state) for state in raw_summary.sort_values("stress_score")["raw_state"].tolist()]
    missing = [state for state in range(n_states) if state not in ordered]
    ordered.extend(missing)
    return {raw_state: regime for regime, raw_state in enumerate(ordered)}


def _remap_probability_row(raw_probability: np.ndarray, state_mapping: dict[int, int]) -> np.ndarray:
    remapped = np.zeros(len(state_mapping))
    for raw_state, regime in state_mapping.items():
        remapped[regime] = raw_probability[raw_state]
    return remapped


def _add_probability_diagnostics(
    regime_summary: pd.DataFrame,
    probabilities: pd.DataFrame,
    labels: pd.Series,
) -> pd.DataFrame:
    rows = []
    max_probability = probabilities.max(axis=1)
    for regime in sorted(labels.unique()):
        mask = labels == regime
        rows.append(
            {
                "regime": int(regime),
                "avg_max_filtered_probability": float(max_probability.loc[mask].mean()),
            }
        )
    return regime_summary.merge(pd.DataFrame(rows), on="regime", how="left")


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
