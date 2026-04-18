from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SvdPcaResult:
    returns: pd.DataFrame
    standardized_returns: pd.DataFrame
    singular_values: pd.Series
    explained_variance_ratio: pd.Series
    cumulative_explained_variance: pd.Series
    component_loadings: pd.DataFrame
    component_scores: pd.DataFrame
    reconstruction_rank1: pd.DataFrame
    scaler: StandardScaler
    pca: PCA


def run_svd_pca(returns: pd.DataFrame, n_components: int | None = None) -> SvdPcaResult:
    """Run standardized SVD/PCA on a T x N return matrix."""

    clean_returns = returns.dropna(how="any")
    scaler = StandardScaler()
    standardized_values = scaler.fit_transform(clean_returns)
    standardized_returns = pd.DataFrame(
        standardized_values,
        index=clean_returns.index,
        columns=clean_returns.columns,
    )

    max_components = min(standardized_returns.shape)
    if n_components is None:
        n_components = max_components
    n_components = min(n_components, max_components)

    pca = PCA(n_components=n_components, svd_solver="full", random_state=42)
    scores = pca.fit_transform(standardized_returns)

    component_names = [f"PC{i}" for i in range(1, n_components + 1)]
    singular_values = pd.Series(pca.singular_values_, index=component_names, name="singular_value")
    explained = pd.Series(
        pca.explained_variance_ratio_,
        index=component_names,
        name="explained_variance_ratio",
    )
    cumulative = explained.cumsum().rename("cumulative_explained_variance")

    loadings = pd.DataFrame(
        pca.components_.T,
        index=clean_returns.columns,
        columns=component_names,
    )
    scores_df = pd.DataFrame(scores, index=clean_returns.index, columns=component_names)

    rank1_values = np.outer(scores[:, 0], pca.components_[0])
    reconstruction_rank1 = pd.DataFrame(
        rank1_values,
        index=clean_returns.index,
        columns=clean_returns.columns,
    )

    return SvdPcaResult(
        returns=clean_returns,
        standardized_returns=standardized_returns,
        singular_values=singular_values,
        explained_variance_ratio=explained,
        cumulative_explained_variance=cumulative,
        component_loadings=loadings,
        component_scores=scores_df,
        reconstruction_rank1=reconstruction_rank1,
        scaler=scaler,
        pca=pca,
    )


def summarize_pc_interpretation(result: SvdPcaResult, n_components: int = 3) -> pd.DataFrame:
    """Create a compact table for interpreting the first components."""

    rows = []
    for pc in result.component_loadings.columns[:n_components]:
        loadings = result.component_loadings[pc].sort_values(key=np.abs, ascending=False)
        positive_loadings = result.component_loadings[pc].loc[lambda values: values > 0].sort_values(ascending=False)
        negative_loadings = result.component_loadings[pc].loc[lambda values: values < 0].sort_values(ascending=True)
        rows.append(
            {
                "component": pc,
                "explained_variance": result.explained_variance_ratio.loc[pc],
                "dominant_asset_abs_loading": loadings.index[0],
                "dominant_abs_loading": loadings.iloc[0],
                "top_positive_loadings": _format_loadings(positive_loadings.head(3)),
                "top_negative_loadings": _format_loadings(negative_loadings.head(3)),
            }
        )
    return pd.DataFrame(rows)


def _format_loadings(loadings: pd.Series) -> str:
    if loadings.empty:
        return "none"
    return ", ".join(f"{idx}:{val:.2f}" for idx, val in loadings.items())
