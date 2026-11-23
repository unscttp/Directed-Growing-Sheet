"""Ordered, JSON-friendly risk levels shared by tools and allowances."""

RISK_LEVELS = ("low", "medium", "high", "critical")


def risk_rank(level: str) -> int:
    if not isinstance(level, str) or level not in RISK_LEVELS:
        raise ValueError(f"Risk level must be one of {RISK_LEVELS}, got {level!r}.")
    return RISK_LEVELS.index(level)
