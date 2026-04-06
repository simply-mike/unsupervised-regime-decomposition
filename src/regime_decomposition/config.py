from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


@dataclass(frozen=True)
class MarketDataConfig:
    """Configuration for the first market data universe."""

    tickers: tuple[str, ...] = ("SPY", "QQQ", "IWM", "TLT", "GLD", "^VIX")
    start: str = "2000-01-01"
    end: str | None = None
    auto_adjust: bool = True


DEFAULT_MARKET_CONFIG = MarketDataConfig()
