"""Exceptions used by the live trading add-on."""


class LiveTradingError(RuntimeError):
    """Base class for live trading pipeline failures."""


class SafetyError(LiveTradingError):
    """Raised when a safety gate blocks order generation or execution."""


class BrokerUnavailable(LiveTradingError):
    """Raised when a requested broker integration is unavailable."""


class DataUnavailable(LiveTradingError):
    """Raised when required market data is missing or stale."""
