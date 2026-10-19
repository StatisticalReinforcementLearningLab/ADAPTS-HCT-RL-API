"""
The result type of a monitoring check (Monitoring-Spec §2.2, §9.1).

A check is a pure function of the database and a logical window that returns
:class:`Finding` objects; the caller (an inline hook or the runner) turns each
into an alarm row. ``dedup_key`` is the identity of the finding: the same
problem found again (by a rerun over the same period, by the runner after the
inline hook, or by a second process) has the same key and raises nothing new.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import math
from typing import Any


@dataclasses.dataclass
class Finding:
    """One problem found by a check; becomes one ``monitor_alarms`` row."""

    item_id: str
    message: str
    dedup_key: str
    at: datetime.datetime | None = None          # logical event time
    detail: dict = dataclasses.field(default_factory=dict)
    group_id: str | None = None
    decision_type: str | None = None
    decision_idx: int | None = None
    rid: str | None = None
    value: float | None = None
    threshold: float | None = None
    fallback_used: str | None = None
    severity: str | None = None                  # overrides the catalog default (escalation)
    owner: str | None = None


MAX_DEDUP_LEN = 255


def dedup_key(*parts: Any) -> str:
    """``part1:part2:...``; hashed when longer than the 255-character column."""
    key = ":".join("" if p is None else (p.isoformat() if hasattr(p, "isoformat") else str(p)) for p in parts)
    if len(key) <= MAX_DEDUP_LEN:
        return key
    head = key.split(":", 1)[0]
    return f"{head}:sha256:{hashlib.sha256(key.encode('utf-8')).hexdigest()}"


def jsonable(value: Any, _depth: int = 0) -> Any:
    """
    A JSON-safe copy of ``value`` for ``detail`` columns: dates as ISO
    strings, numpy scalars and arrays as Python values, non-finite floats as
    their ``repr`` (PostgreSQL JSON rejects NaN), unknown objects as ``str``.
    """
    if _depth > 20:
        return str(value)
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): jsonable(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
        return [jsonable(v, _depth + 1) for v in items]
    try:  # numpy scalars / arrays without importing numpy here
        if hasattr(value, "tolist"):
            return jsonable(value.tolist(), _depth + 1)
        if hasattr(value, "item"):
            return jsonable(value.item(), _depth + 1)
    except Exception:  # noqa: BLE001
        pass
    return str(value)


def finite_or_none(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def capped(items: list, limit: int = 50) -> list:
    """At most ``limit`` items (detail payloads stay small)."""
    return list(items[:limit])


__all__ = ["Finding", "dedup_key", "jsonable", "finite_or_none", "capped", "MAX_DEDUP_LEN"]
