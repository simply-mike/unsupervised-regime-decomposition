from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from regime_decomposition.svd_pca import SvdPcaResult


def set_plot_style() -> None:
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        palette="colorblind",
        rc={
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )


def plot_cumulative_returns(returns: pd.DataFrame, output_path: Path) -> None:
    cumulative = returns.cumsum()

    fig, ax = plt.subplots(figsize=(12, 6))
    cumulative.plot(ax=ax, linewidth=1.4)
    ax.set_title("Cumulative Log Returns")
    ax.set_ylabel("Cumulative log return")
    ax.set_xlabel("")
    ax.legend(loc="upper left", ncols=3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_singular_values(result: SvdPcaResult, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    result.singular_values.plot(kind="bar", ax=ax, color="#4C78A8")
    ax.set_title("SVD Singular Values")
    ax.set_ylabel("Singular value")
    ax.set_xlabel("Principal component")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_explained_variance(result: SvdPcaResult, output_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    result.explained_variance_ratio.plot(kind="bar", ax=ax1, color="#59A14F", alpha=0.85)
    ax1.set_ylabel("Explained variance ratio")
    ax1.set_xlabel("Principal component")
    ax1.set_ylim(0, max(0.05, result.explained_variance_ratio.max() * 1.15))

    ax2 = ax1.twinx()
    result.cumulative_explained_variance.plot(ax=ax2, color="#E15759", marker="o", linewidth=2)
    ax2.set_ylabel("Cumulative explained variance")
    ax2.set_ylim(0, 1.05)
    ax1.set_title("PCA Explained Variance")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_component_loadings(result: SvdPcaResult, output_path: Path, n_components: int = 3) -> None:
    loadings = result.component_loadings.iloc[:, :n_components]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="vlag", center=0, linewidths=0.5, ax=ax)
    ax.set_title("Principal Component Loadings")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_pc_scores(result: SvdPcaResult, output_path: Path, n_components: int = 3) -> None:
    scores = result.component_scores.iloc[:, :n_components]
    fig, axes = plt.subplots(n_components, 1, figsize=(12, 2.8 * n_components), sharex=True)
    if n_components == 1:
        axes = [axes]
    for ax, component in zip(axes, scores.columns, strict=True):
        scores[component].plot(ax=ax, linewidth=0.8, color="#4C78A8")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_title(f"{component} Score Through Time")
        ax.set_ylabel("Score")
    axes[-1].set_xlabel("")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_pc_scatter(result: SvdPcaResult, output_path: Path) -> None:
    scores = result.component_scores
    if scores.shape[1] < 2:
        return

    plot_df = scores[["PC1", "PC2"]].copy()
    plot_df["year"] = plot_df.index.year

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        plot_df["PC1"],
        plot_df["PC2"],
        c=plot_df["year"],
        s=10,
        cmap="viridis",
        alpha=0.7,
        linewidths=0,
    )
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("Daily Market States in PC Space")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="Year")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
