"""A-share research adapters for AlphaGPT (no live trading)."""

from .daily import AShareDailyLoader, make_daily_features
from .minute import AShareMinuteLoader, build_minute_intervals

__all__ = ["AShareDailyLoader", "make_daily_features", "AShareMinuteLoader", "build_minute_intervals"]
