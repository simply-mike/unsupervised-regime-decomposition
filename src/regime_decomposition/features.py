from __future__ import annotations

import numpy as np
import pandas as pd


def get_field(market_data: pd.DataFrame, field: str) -> pd.DataFrame:
    """Extract a single OHLCV field from a canonical (ticker, field) frame."""

    return market_data.xs(field, axis=1, level=1)


def compute_log_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns for all assets."""

    returns = np.log(close).diff()
    returns.index.name = "date"
    return returns


def compute_realized_volatility(returns: pd.DataFrame, windows: tuple[int, ...] = (20, 60)) -> pd.DataFrame:
    """Annualized rolling realized volatility from daily returns."""

    vol_frames: list[pd.DataFrame] = []
    for window in windows:
        vol = returns.rolling(window=window, min_periods=window).std() * np.sqrt(252)
        vol.columns = [f"{col}_rv_{window}d" for col in vol.columns]
        vol_frames.append(vol)
    return pd.concat(vol_frames, axis=1)


def compute_volume_features(volume: pd.DataFrame, windows: tuple[int, ...] = (20, 60)) -> pd.DataFrame:
    """Log-volume and rolling volume z-score features."""

    safe_volume = volume.replace(0.0, np.nan)
    log_volume = np.log(safe_volume)
    log_volume.columns = [f"{col}_log_volume" for col in log_volume.columns]

    features = [log_volume]
    for window in windows:
        rolling_mean = log_volume.rolling(window=window, min_periods=window).mean()
        rolling_std = log_volume.rolling(window=window, min_periods=window).std()
        zscore = (log_volume - rolling_mean) / rolling_std
        zscore.columns = [f"{col.replace('_log_volume', '')}_volume_z_{window}d" for col in log_volume.columns]
        features.append(zscore)

    return pd.concat(features, axis=1)


def build_returns_matrix(
    market_data: pd.DataFrame,
    return_tickers: tuple[str, ...] = ("SPY", "QQQ", "IWM", "TLT", "GLD"),
) -> pd.DataFrame:
    """Build the T x N return matrix used by SVD/PCA."""

    close = get_field(market_data, "Close")
    returns = compute_log_returns(close)
    missing = sorted(set(return_tickers).difference(returns.columns))
    if missing:
        raise KeyError(f"Missing return tickers: {missing}")
    return returns.loc[:, list(return_tickers)].dropna(how="any")


def build_feature_matrix(market_data: pd.DataFrame) -> pd.DataFrame:
    """Build an interpretable first-pass feature matrix for later regime models."""

    close = get_field(market_data, "Close")
    volume = get_field(market_data, "Volume")
    returns = compute_log_returns(close)

    asset_returns = returns.drop(columns=["^VIX"], errors="ignore").add_suffix("_ret")
    realized_vol = compute_realized_volatility(asset_returns.rename(columns=lambda c: c.removesuffix("_ret")))
    volume_features = compute_volume_features(volume.drop(columns=["^VIX"], errors="ignore"))

    features = [asset_returns, realized_vol, volume_features]
    if "^VIX" in close.columns:
        vix = close["^VIX"].rename("vix_close")
        vix_change = np.log(vix).diff().rename("vix_log_change")
        features.extend([vix, vix_change])

    feature_matrix = pd.concat(features, axis=1).replace([np.inf, -np.inf], np.nan)
    feature_matrix.index.name = "date"
    return feature_matrix.dropna(how="any")
