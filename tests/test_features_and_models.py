from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_decomposition.backtest import build_default_exposure_map, run_regime_backtests, run_strategy_backtest
from regime_decomposition.clustering import build_clustering_matrix, fit_kmeans_regimes
from regime_decomposition.features import build_feature_matrix, build_returns_matrix
from regime_decomposition.gmm import fit_gmm_regimes
from regime_decomposition.hmm import fit_hmm_regimes
from regime_decomposition.eda import CrisisWindow
from regime_decomposition.robustness import build_exposure_scenarios, summarize_stress_period_performance
from regime_decomposition.svd_pca import run_svd_pca, summarize_pc_interpretation
from regime_decomposition.walk_forward import run_walk_forward_hmm


def test_feature_builders_create_clean_return_and_feature_matrices() -> None:
    market_data = _synthetic_market_data(periods=140)

    returns = build_returns_matrix(market_data)
    features = build_feature_matrix(market_data)

    assert returns.shape[1] == 5
    assert not returns.isna().any().any()
    assert "vix_close" in features.columns
    assert "SPY_rv_20d" in features.columns
    assert "SPY_volume_z_60d" in features.columns
    assert np.isfinite(features.to_numpy()).all()


def test_svd_pca_outputs_have_consistent_shapes_and_interpretation_columns() -> None:
    market_data = _synthetic_market_data(periods=120)
    returns = build_returns_matrix(market_data)

    result = run_svd_pca(returns)
    interpretation = summarize_pc_interpretation(result, n_components=3)

    assert result.component_scores.shape == returns.shape
    assert result.component_loadings.shape == (returns.shape[1], returns.shape[1])
    assert np.isclose(result.explained_variance_ratio.sum(), 1.0)
    assert {"top_positive_loadings", "top_negative_loadings"}.issubset(interpretation.columns)


def test_kmeans_and_gmm_regime_outputs_are_aligned_and_probabilistic() -> None:
    panel = _synthetic_eda_panel()
    model_matrix = build_clustering_matrix(panel)

    kmeans = fit_kmeans_regimes(panel=panel, model_matrix=model_matrix, selected_k=3, k_values=range(2, 5))
    gmm = fit_gmm_regimes(
        panel=panel,
        model_matrix=model_matrix,
        selected_components=3,
        n_components_values=range(2, 5),
    )

    assert kmeans.labels.index.equals(model_matrix.index)
    assert gmm.labels.index.equals(model_matrix.index)
    assert gmm.probabilities.index.equals(model_matrix.index)
    assert np.allclose(gmm.probabilities.sum(axis=1), 1.0)
    assert "spy_conditional_annualized_return" in kmeans.cluster_summary.columns
    assert "avg_max_probability" in gmm.regime_summary.columns


def test_hmm_outputs_include_persistent_state_diagnostics() -> None:
    panel = _synthetic_eda_panel()
    model_matrix = build_clustering_matrix(panel)

    hmm = fit_hmm_regimes(
        panel=panel,
        model_matrix=model_matrix,
        selected_states=3,
        n_states_values=range(2, 4),
        n_restarts=2,
    )

    assert hmm.labels.index.equals(model_matrix.index)
    assert hmm.probabilities.index.equals(model_matrix.index)
    assert np.allclose(hmm.probabilities.sum(axis=1), 1.0)
    assert np.allclose(hmm.transition_matrix.sum(axis=1), 1.0)
    assert (hmm.expected_durations["expected_duration_days"] > 0).all()
    assert "avg_max_state_probability" in hmm.regime_summary.columns


def test_walk_forward_hmm_outputs_oos_filtered_probabilities() -> None:
    panel = _synthetic_eda_panel()
    model_matrix = build_clustering_matrix(panel)

    result = run_walk_forward_hmm(
        panel=panel,
        model_matrix=model_matrix,
        n_states=3,
        min_train_size=90,
        refit_frequency=30,
        test_size=30,
        n_restarts=1,
    )

    assert len(result.labels) == len(model_matrix) - 90
    assert result.labels.index.equals(result.probabilities.index)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert not result.probabilities.index.duplicated().any()
    assert result.diagnostics["train_size"].is_monotonic_increasing


def test_walk_forward_hmm_rejects_overlapping_oos_blocks() -> None:
    panel = _synthetic_eda_panel()
    model_matrix = build_clustering_matrix(panel)

    with pytest.raises(ValueError, match="non-overlapping"):
        run_walk_forward_hmm(
            panel=panel,
            model_matrix=model_matrix,
            n_states=3,
            min_train_size=90,
            refit_frequency=10,
            test_size=30,
            n_restarts=1,
        )


def test_strategy_backtest_lags_signals_and_charges_turnover_costs() -> None:
    dates = pd.bdate_range("2022-01-03", periods=4)
    returns = pd.Series(np.log1p([0.01, 0.02, -0.01, 0.03]), index=dates)
    target = pd.Series([1.0, 0.0, 1.0, 1.0], index=dates)

    result = run_strategy_backtest(
        spy_log_returns=returns,
        target_exposure=target,
        strategy_name="test",
        transaction_cost_bps=10.0,
        signal_lag=1,
    )

    assert result["test_exposure"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert np.isclose(result["test_turnover"].sum(), 3.0)
    assert np.isclose(result["test_transaction_cost"].sum(), 0.003)
    assert result["test_return"].iloc[0] == 0.0


def test_regime_backtests_return_aligned_summary_and_probability_scaled_exposure() -> None:
    panel = _synthetic_eda_panel().iloc[:90].copy()
    panel["walk_forward_hmm_state"] = np.tile([0, 1, 2], 30)
    probabilities = pd.DataFrame(
        {
            "regime_0": (panel["walk_forward_hmm_state"] == 0).astype(float),
            "regime_1": (panel["walk_forward_hmm_state"] == 1).astype(float),
            "regime_2": (panel["walk_forward_hmm_state"] == 2).astype(float),
        },
        index=panel.index,
    )
    exposure_map = build_default_exposure_map(n_regimes=3)

    result = run_regime_backtests(
        panel=panel,
        probabilities=probabilities,
        exposure_map=exposure_map,
        transaction_cost_bps=1.0,
        signal_lag=1,
    )

    assert result.daily_results.index.equals(panel.index)
    assert set(result.summary["strategy"]) == {"buy_and_hold", "hard_regime_filter", "probability_scaled"}
    assert np.isclose(result.exposure_map["target_exposure"].sum(), 1.5)
    assert result.daily_results["probability_scaled_exposure"].between(0.0, 1.0).all()


def test_robustness_helpers_build_scenarios_and_recompute_window_returns() -> None:
    panel = _synthetic_eda_panel().iloc[:40].copy()
    panel["walk_forward_hmm_state"] = np.repeat([0, 1, 2, 3], 10)
    probabilities = pd.DataFrame(
        {
            f"regime_{regime}": (panel["walk_forward_hmm_state"] == regime).astype(float)
            for regime in range(4)
        },
        index=panel.index,
    )
    scenarios = build_exposure_scenarios(n_regimes=4)
    result = run_regime_backtests(
        panel=panel,
        probabilities=probabilities,
        exposure_map=scenarios["balanced"],
        transaction_cost_bps=0.0,
        signal_lag=1,
    )
    windows = (
        CrisisWindow("observed", str(panel.index[5].date()), str(panel.index[20].date()), "Observed"),
        CrisisWindow("empty", "1999-01-01", "1999-01-31", "Empty"),
    )

    stress = summarize_stress_period_performance(result.daily_results, windows)

    assert set(scenarios) == {"balanced", "defensive", "aggressive", "crisis_cut"}
    assert all(exposure.between(0.0, 1.0).all() for exposure in scenarios.values())
    assert stress.query("window == 'observed'")["observations"].gt(0).all()
    assert stress.query("window == 'empty'")["observations"].eq(0).all()


def test_exposure_scenario_fallback_keeps_defensive_below_balanced() -> None:
    scenarios = build_exposure_scenarios(n_regimes=5)

    assert set(scenarios) == {"balanced", "defensive", "aggressive", "crisis_cut"}
    assert (scenarios["defensive"] <= scenarios["balanced"]).all()
    assert (scenarios["aggressive"] >= scenarios["balanced"]).all()


def _synthetic_market_data(periods: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", periods=periods)
    tickers = ("SPY", "QQQ", "IWM", "TLT", "GLD", "^VIX")
    frames: dict[tuple[str, str], np.ndarray] = {}

    for ticker in tickers:
        drift = 0.0002 if ticker != "^VIX" else 0.0
        vol = 0.01 if ticker != "^VIX" else 0.03
        close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, size=periods)))
        if ticker == "^VIX":
            close = np.clip(20 + np.cumsum(rng.normal(0.0, 0.4, size=periods)), 10, 80)
        volume = rng.integers(1_000_000, 10_000_000, size=periods).astype(float)
        frames[(ticker, "Close")] = close
        frames[(ticker, "Volume")] = volume

    columns = pd.MultiIndex.from_tuples(frames.keys())
    return pd.DataFrame(frames, index=dates, columns=columns)


def _synthetic_eda_panel() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2021-01-01", periods=180)
    states = np.repeat([0, 1, 2], repeats=60)
    spy_ret = np.select(
        [states == 0, states == 1, states == 2],
        [
            rng.normal(0.0006, 0.004, len(dates)),
            rng.normal(0.0001, 0.010, len(dates)),
            rng.normal(-0.0009, 0.020, len(dates)),
        ],
    )
    spy_close = 100 * np.exp(np.cumsum(spy_ret))
    panel = pd.DataFrame(
        {
            "spy_close": spy_close,
            "spy_ret": spy_ret,
            "SPY_rv_20d": np.select([states == 0, states == 1, states == 2], [0.08, 0.16, 0.35]),
            "SPY_rv_60d": np.select([states == 0, states == 1, states == 2], [0.09, 0.17, 0.32]),
            "vix_close": np.select([states == 0, states == 1, states == 2], [12.0, 20.0, 38.0]),
            "vix_log_change": rng.normal(0.0, 0.02, len(dates)),
            "PC1": np.select([states == 0, states == 1, states == 2], [0.4, 0.0, -0.7])
            + rng.normal(0, 0.05, len(dates)),
            "PC2": np.select([states == 0, states == 1, states == 2], [0.2, -0.2, -0.4])
            + rng.normal(0, 0.05, len(dates)),
            "PC3": np.select([states == 0, states == 1, states == 2], [0.1, -0.1, 0.0])
            + rng.normal(0, 0.05, len(dates)),
        },
        index=dates,
    )
    panel["spy_cum_log_ret"] = panel["spy_ret"].cumsum()
    panel["spy_drawdown"] = panel["spy_close"].div(panel["spy_close"].cummax()).sub(1.0)
    panel.index.name = "date"
    return panel
