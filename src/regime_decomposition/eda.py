from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regime_decomposition.features import get_field


@dataclass(frozen=True)
class CrisisWindow:
    name: str
    start: str
    end: str
    label: str


DEFAULT_CRISIS_WINDOWS: tuple[CrisisWindow, ...] = (
    CrisisWindow("gfc_2008", "2007-10-09", "2009-03-09", "Global Financial Crisis"),
    CrisisWindow("covid_2020", "2020-02-19", "2020-03-23", "COVID Crash"),
    CrisisWindow("rates_2022", "2022-01-03", "2022-10-12", "Rates / Inflation Bear Market"),
)


def build_eda_panel(
    market_data: pd.DataFrame,
    returns: pd.DataFrame,
    features: pd.DataFrame,
    pca_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Join raw price, returns, volatility, VIX, and PCA scores for diagnostics."""

    close = get_field(market_data, "Close")
    panel_parts = [
        close["SPY"].rename("spy_close"),
        returns["SPY"].rename("spy_ret"),
        features.reindex(returns.index).filter(items=["SPY_rv_20d", "SPY_rv_60d", "vix_close", "vix_log_change"]),
        pca_scores.reindex(returns.index).filter(items=["PC1", "PC2", "PC3"]),
    ]
    panel = pd.concat(panel_parts, axis=1).sort_index()
    panel["spy_cum_log_ret"] = panel["spy_ret"].fillna(0.0).cumsum()
    panel["spy_drawdown"] = panel["spy_close"].div(panel["spy_close"].cummax()).sub(1.0)
    panel.index.name = "date"
    return panel


def summarize_data_quality(
    market_data: pd.DataFrame,
    returns: pd.DataFrame,
    features: pd.DataFrame,
    pca_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact audit of date coverage and missing values."""

    rows = []
    datasets = {
        "raw_market_data": market_data,
        "returns_matrix": returns,
        "feature_matrix": features,
        "pca_scores": pca_scores,
    }
    for name, data in datasets.items():
        rows.append(
            {
                "dataset": name,
                "start": data.index.min().date(),
                "end": data.index.max().date(),
                "rows": len(data),
                "columns": data.shape[1],
                "missing_cells": int(data.isna().sum().sum()),
                "missing_pct": float(data.isna().sum().sum() / data.size) if data.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_crisis_windows(panel: pd.DataFrame, windows: tuple[CrisisWindow, ...]) -> pd.DataFrame:
    """Summarize return, risk, and latent-factor behavior during stress windows."""

    rows = []
    for window in windows:
        window_data = panel.loc[window.start : window.end].dropna(subset=["spy_ret"])
        if window_data.empty:
            continue

        trading_days = len(window_data)
        total_log_return = window_data["spy_ret"].sum()
        ann_vol = window_data["spy_ret"].std() * np.sqrt(252)
        hit_rate = (window_data["spy_ret"] > 0).mean()
        row = {
            "window": window.name,
            "label": window.label,
            "start": window_data.index.min().date(),
            "end": window_data.index.max().date(),
            "trading_days": trading_days,
            "spy_total_return": float(np.expm1(total_log_return)),
            "spy_annualized_vol": float(ann_vol),
            "spy_hit_rate": float(hit_rate),
            "spy_max_drawdown": float(window_data["spy_drawdown"].min()),
            "spy_worst_day": float(window_data["spy_ret"].min()),
            "spy_best_day": float(window_data["spy_ret"].max()),
            "mean_vix": float(window_data["vix_close"].mean()),
            "max_vix": float(window_data["vix_close"].max()),
            "mean_spy_rv_20d": float(window_data["SPY_rv_20d"].mean()),
            "max_spy_rv_20d": float(window_data["SPY_rv_20d"].max()),
        }
        for pc in ("PC1", "PC2", "PC3"):
            if pc in window_data:
                row[f"{pc}_mean"] = float(window_data[pc].mean())
                row[f"{pc}_min"] = float(window_data[pc].min())
                row[f"{pc}_max"] = float(window_data[pc].max())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_feature_missingness(data: pd.DataFrame) -> pd.DataFrame:
    """Rank columns by missing observations before model fitting."""

    missing = data.isna().sum().rename("missing_count").to_frame()
    missing["missing_pct"] = missing["missing_count"] / len(data)
    return missing.sort_values(["missing_pct", "missing_count"], ascending=False)


def plot_stress_overview(panel: pd.DataFrame, windows: tuple[CrisisWindow, ...], output_path: Path) -> None:
    """Plot full-sample SPY, drawdown, VIX, and PCA scores with crisis shading."""

    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True, gridspec_kw={"height_ratios": [1.4, 1, 1, 1, 1]})

    panel["spy_close"].plot(ax=axes[0], color="#4C78A8", linewidth=1.2)
    axes[0].set_yscale("log")
    axes[0].set_title("SPY Price and Stress Windows")
    axes[0].set_ylabel("SPY close, log scale")

    panel["spy_drawdown"].plot(ax=axes[1], color="#E15759", linewidth=1.0)
    axes[1].set_title("SPY Drawdown")
    axes[1].set_ylabel("Drawdown")

    panel["vix_close"].plot(ax=axes[2], color="#F28E2B", linewidth=1.0)
    axes[2].set_title("VIX Close")
    axes[2].set_ylabel("VIX")

    panel[["SPY_rv_20d", "SPY_rv_60d"]].plot(ax=axes[3], linewidth=1.0)
    axes[3].set_title("SPY Realized Volatility")
    axes[3].set_ylabel("Annualized vol")
    axes[3].legend(loc="upper left")

    panel[["PC1", "PC2", "PC3"]].plot(ax=axes[4], linewidth=0.8)
    axes[4].axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axes[4].set_title("PCA Scores")
    axes[4].set_ylabel("Score")
    axes[4].legend(loc="upper left", ncols=3)

    _shade_windows(axes, windows)
    axes[-1].set_xlabel("")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_crisis_window_dashboard(panel: pd.DataFrame, window: CrisisWindow, output_path: Path) -> None:
    """Plot detailed diagnostics for a single crisis window."""

    context_start = pd.Timestamp(window.start) - pd.DateOffset(months=3)
    context_end = pd.Timestamp(window.end) + pd.DateOffset(months=3)
    data = panel.loc[context_start:context_end].copy()

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    data["spy_close"].plot(ax=axes[0], color="#4C78A8", linewidth=1.3)
    axes[0].set_title(f"{window.label}: SPY Price")
    axes[0].set_ylabel("SPY close")

    data["spy_ret"].plot(ax=axes[1], kind="bar", color="#9C755F", width=1.0)
    axes[1].set_title("SPY Daily Log Returns")
    axes[1].set_ylabel("Daily return")
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(8))

    data[["SPY_rv_20d", "SPY_rv_60d", "vix_close"]].plot(ax=axes[2], linewidth=1.0)
    axes[2].set_title("Volatility Diagnostics")
    axes[2].set_ylabel("Vol / VIX")
    axes[2].legend(loc="upper left", ncols=3)

    data[["PC1", "PC2", "PC3"]].plot(ax=axes[3], linewidth=1.0)
    axes[3].axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axes[3].set_title("PCA Scores")
    axes[3].set_ylabel("Score")
    axes[3].legend(loc="upper left", ncols=3)

    _shade_windows(axes, (window,), alpha=0.18)
    axes[-1].set_xlabel("")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _shade_windows(axes: np.ndarray | list[plt.Axes], windows: tuple[CrisisWindow, ...], alpha: float = 0.12) -> None:
    for ax in axes:
        for window in windows:
            ax.axvspan(pd.Timestamp(window.start), pd.Timestamp(window.end), color="#E15759", alpha=alpha, linewidth=0)
