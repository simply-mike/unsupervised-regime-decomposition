from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


DEFAULT_CLUSTER_FEATURES: tuple[str, ...] = ("PC1", "PC2", "PC3", "SPY_rv_20d", "vix_close")


@dataclass(frozen=True)
class KMeansClusteringResult:
    model_matrix: pd.DataFrame
    scaled_matrix: pd.DataFrame
    labels: pd.Series
    model_selection: pd.DataFrame
    cluster_summary: pd.DataFrame
    scaler: StandardScaler
    model: KMeans
    selected_k: int


def build_clustering_matrix(
    panel: pd.DataFrame,
    feature_columns: tuple[str, ...] = DEFAULT_CLUSTER_FEATURES,
) -> pd.DataFrame:
    """Select and clean the model matrix for PCA-space clustering."""

    missing = sorted(set(feature_columns).difference(panel.columns))
    if missing:
        raise KeyError(f"Missing clustering features: {missing}")

    model_matrix = panel.loc[:, list(feature_columns)].replace([np.inf, -np.inf], np.nan).dropna(how="any")
    model_matrix.index.name = "date"
    return model_matrix


def run_kmeans_model_selection(
    model_matrix: pd.DataFrame,
    k_values: range = range(2, 9),
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[int, KMeans], pd.DataFrame, StandardScaler]:
    """Fit K-Means models across k and return elbow/silhouette diagnostics."""

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(model_matrix)
    scaled_matrix = pd.DataFrame(scaled_values, index=model_matrix.index, columns=model_matrix.columns)

    rows = []
    models: dict[int, KMeans] = {}
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=50, random_state=random_state)
        labels = model.fit_predict(scaled_matrix)
        models[k] = model
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(scaled_matrix, labels)),
                "min_cluster_size": int(pd.Series(labels).value_counts().min()),
                "max_cluster_size": int(pd.Series(labels).value_counts().max()),
            }
        )

    model_selection = pd.DataFrame(rows)
    return model_selection, models, scaled_matrix, scaler


def fit_kmeans_regimes(
    panel: pd.DataFrame,
    model_matrix: pd.DataFrame,
    selected_k: int | None = None,
    k_values: range = range(2, 9),
    random_state: int = 42,
) -> KMeansClusteringResult:
    """Fit the selected K-Means model and remap clusters from calm to stress."""

    model_selection, models, scaled_matrix, scaler = run_kmeans_model_selection(
        model_matrix=model_matrix,
        k_values=k_values,
        random_state=random_state,
    )
    if selected_k is None:
        selected_k = int(model_selection.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])

    model = models[selected_k]
    raw_labels = pd.Series(model.labels_, index=model_matrix.index, name="raw_cluster")
    raw_summary = summarize_cluster_behavior(panel=panel, labels=raw_labels, label_name="raw_cluster")
    label_mapping = _build_calm_to_stress_mapping(raw_summary)
    labels = raw_labels.map(label_mapping).astype(int).rename("regime")
    cluster_summary = summarize_cluster_behavior(panel=panel, labels=labels, label_name="regime")
    cluster_summary = add_regime_names(cluster_summary)

    return KMeansClusteringResult(
        model_matrix=model_matrix,
        scaled_matrix=scaled_matrix,
        labels=labels,
        model_selection=model_selection,
        cluster_summary=cluster_summary,
        scaler=scaler,
        model=model,
        selected_k=selected_k,
    )


def run_hierarchical_sanity_check(
    model_matrix: pd.DataFrame,
    kmeans_labels: pd.Series,
) -> tuple[pd.Series, float]:
    """Fit Ward hierarchical clustering and compare it to K-Means labels."""

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(model_matrix)
    model = AgglomerativeClustering(n_clusters=int(kmeans_labels.nunique()), linkage="ward")
    raw_labels = pd.Series(model.fit_predict(scaled_values), index=model_matrix.index, name="hierarchical_raw_cluster")

    comparison_labels = kmeans_labels.reindex(raw_labels.index)
    score = float(adjusted_rand_score(comparison_labels, raw_labels))
    return raw_labels, score


def summarize_cluster_behavior(panel: pd.DataFrame, labels: pd.Series, label_name: str) -> pd.DataFrame:
    """Summarize economic behavior by cluster/regime."""

    data = panel.reindex(labels.index).copy()
    data[label_name] = labels

    rows = []
    for label, group in data.groupby(label_name, sort=True):
        spy_ret = group["spy_ret"].dropna()
        ann_vol = spy_ret.std() * np.sqrt(252)
        conditional_ann_return = np.expm1(spy_ret.mean() * 252)
        rows.append(
            {
                label_name: int(label),
                "count": int(len(group)),
                "pct_obs": float(len(group) / len(data)),
                "spy_mean_daily_return": float(spy_ret.mean()),
                "spy_conditional_annualized_return": float(conditional_ann_return),
                "spy_annualized_vol": float(ann_vol),
                "spy_hit_rate": float((spy_ret > 0).mean()),
                "avg_spy_drawdown": float(group["spy_drawdown"].mean()),
                "mean_vix": float(group["vix_close"].mean()),
                "mean_spy_rv_20d": float(group["SPY_rv_20d"].mean()),
                "PC1_mean": float(group["PC1"].mean()),
                "PC2_mean": float(group["PC2"].mean()),
                "PC3_mean": float(group["PC3"].mean()),
            }
        )

    summary = pd.DataFrame(rows)
    summary["stress_score"] = _stress_score(summary)
    return summary.sort_values(label_name).reset_index(drop=True)


def summarize_regime_distribution_by_window(
    labels: pd.Series,
    windows: tuple,
    cluster_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Measure how much each known stress window overlaps each regime."""

    name_map = cluster_summary.set_index("regime")["regime_name"].to_dict()
    rows = []
    for window in windows:
        window_labels = labels.loc[window.start : window.end]
        if window_labels.empty:
            continue
        counts = window_labels.value_counts().sort_index()
        for regime, count in counts.items():
            rows.append(
                {
                    "window": window.name,
                    "label": window.label,
                    "regime": int(regime),
                    "regime_name": name_map[int(regime)],
                    "count": int(count),
                    "pct_window": float(count / len(window_labels)),
                }
            )
    return pd.DataFrame(rows)


def add_regime_names(summary: pd.DataFrame) -> pd.DataFrame:
    """Assign compact, human-readable names from calm to stress."""

    named = summary.copy()
    max_stress_regime = int(named.sort_values("stress_score").iloc[-1]["regime"])
    min_stress_regime = int(named.sort_values("stress_score").iloc[0]["regime"])
    median_vol = named["spy_annualized_vol"].median()
    median_vix = named["mean_vix"].median()

    names: dict[int, str] = {}
    for row in named.itertuples(index=False):
        regime = int(row.regime)
        if regime == min_stress_regime and row.spy_mean_daily_return >= 0:
            base = "calm_bull"
        elif regime == max_stress_regime and row.spy_mean_daily_return < 0:
            base = "high_vol_bear"
        elif (
            (row.mean_vix >= 25 or row.spy_annualized_vol >= 0.20)
            and row.spy_mean_daily_return < 0.0005
            and (row.PC2_mean < 0 or row.PC3_mean < 0)
        ):
            base = "macro_stress"
        elif row.PC2_mean < 0 and row.PC3_mean < 0 and row.spy_mean_daily_return < 0:
            base = "bear_chop"
        elif row.spy_mean_daily_return < 0 and row.spy_annualized_vol > named["spy_annualized_vol"].median():
            base = "volatile_bear"
        elif row.spy_mean_daily_return < 0:
            base = "bear_chop"
        elif row.spy_annualized_vol > named["spy_annualized_vol"].median() and row.spy_mean_daily_return > 0.001:
            base = "high_vol_rebound"
        elif row.PC2_mean < 0 and row.PC3_mean < 0:
            base = "cross_asset_risk_on"
        elif row.spy_annualized_vol >= median_vol or row.mean_vix >= median_vix:
            base = "risk_on_chop"
        else:
            base = "steady_risk_on"
        names[regime] = f"R{regime}_{base}"

    named.insert(1, "regime_name", named["regime"].map(names))
    return named


def plot_kmeans_model_selection(model_selection: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(model_selection["k"], model_selection["inertia"], marker="o", color="#4C78A8")
    axes[0].set_title("K-Means Elbow")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(model_selection["k"], model_selection["silhouette"], marker="o", color="#59A14F")
    axes[1].set_title("K-Means Silhouette")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette score")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_regime_scatter(
    panel: pd.DataFrame,
    labels: pd.Series,
    cluster_summary: pd.DataFrame,
    output_path: Path,
    title: str = "PCA State Space Colored by Regime",
) -> None:
    data = panel.reindex(labels.index).copy()
    data["regime"] = labels
    palette = _regime_palette(cluster_summary)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for regime, group in data.groupby("regime", sort=True):
        name = _regime_name(cluster_summary, regime)
        ax.scatter(group["PC1"], group["PC2"], s=10, alpha=0.65, linewidths=0, color=palette[regime], label=name)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8, markerscale=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_spy_regime_timeline(
    panel: pd.DataFrame,
    labels: pd.Series,
    cluster_summary: pd.DataFrame,
    output_path: Path,
    title: str = "SPY Price With Regime Background",
) -> None:
    data = panel.reindex(labels.index).copy()
    data["regime"] = labels
    palette = _regime_palette(cluster_summary)

    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.plot(data.index, data["spy_close"], color="black", linewidth=1.1, zorder=2)
    for regime, start, end in _regime_segments(data["regime"]):
        ax.axvspan(start, end, color=palette[regime], alpha=0.18, linewidth=0, zorder=0)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("SPY close, log scale")
    ax.set_xlabel("")
    handles = [
        mpatches.Patch(color=palette[row.regime], alpha=0.35, label=row.regime_name)
        for row in cluster_summary.itertuples(index=False)
    ]
    ax.legend(handles=handles, loc="upper left", ncols=2, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_return_distribution_by_regime(
    panel: pd.DataFrame,
    labels: pd.Series,
    cluster_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    data = panel.reindex(labels.index).copy()
    name_map = cluster_summary.set_index("regime")["regime_name"].to_dict()
    data["regime_name"] = labels.map(name_map)
    order = cluster_summary["regime_name"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=data, x="regime_name", y="spy_ret", order=order, ax=axes[0], color="#A0CBE8")
    axes[0].set_title("SPY Daily Returns by Regime")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Daily log return")
    axes[0].tick_params(axis="x", rotation=25)

    for regime_name, group in data.groupby("regime_name", sort=False):
        sns.kdeplot(group["spy_ret"], ax=axes[1], label=regime_name, linewidth=1.4)
    axes[1].set_title("Return Density by Regime")
    axes[1].set_xlabel("Daily log return")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _build_calm_to_stress_mapping(raw_summary: pd.DataFrame) -> dict[int, int]:
    ordered = raw_summary.sort_values("stress_score", ascending=True)["raw_cluster"].tolist()
    return {int(raw_label): regime_id for regime_id, raw_label in enumerate(ordered)}


def _stress_score(summary: pd.DataFrame) -> pd.Series:
    components = pd.DataFrame(
        {
            "vol": summary["spy_annualized_vol"],
            "vix": summary["mean_vix"],
            "rv": summary["mean_spy_rv_20d"],
            "neg_return": -summary["spy_mean_daily_return"],
            "neg_drawdown": -summary["avg_spy_drawdown"],
        }
    )
    std = components.std(ddof=0).replace(0.0, 1.0)
    normalized = (components - components.mean()) / std
    return normalized.mean(axis=1)


def _regime_palette(cluster_summary: pd.DataFrame) -> dict[int, tuple[float, float, float]]:
    colors = sns.color_palette("colorblind", n_colors=len(cluster_summary))
    return {int(regime): colors[i] for i, regime in enumerate(cluster_summary["regime"])}


def _regime_name(cluster_summary: pd.DataFrame, regime: int) -> str:
    row = cluster_summary.loc[cluster_summary["regime"] == regime].iloc[0]
    return str(row["regime_name"])


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
