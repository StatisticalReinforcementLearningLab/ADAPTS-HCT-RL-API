"""
Study calendar for ADAPTS-HCT (contract D2, D3, D5; main_new.tex §3.1 eq. (cohort-index),
Algorithm 3, §5.2; API-Spec §2.1, §4.1; Monitoring-Spec §3).

Calendar weeks run Monday 00:00 to Sunday 24:00 in the study timezone. A dyad's
``consent_start_date`` may be any day; its week 1 starts on the first Monday on
or after it (``enr_i``), and the days before that Monday are its partial week
(study week 0).

Index sets of main_new.tex eq. (cohort-index), for the week starting Monday ``W``:

- ``E_W`` (enrolled): dyads with ``week1_start <= W``;
- ``I_W`` (active):   dyads whose study week of ``W`` is in ``1..T``.

Cohort warm-up (contract D2; Algorithm 3 line 3): a decision in week ``W`` is
cohort warm-up iff ``|E_{W-1}| < N_warm``, i.e. fewer than ``n_warm`` dyads have
*completed* week 1 (``week1_start <= W - 7 days``). Per-dyad warm-up (D3): the
decision falls in the dyad's partial week or in study weeks ``1..t_warm``.

Every function that needs the dyad list takes an optional ``groups`` argument;
``None`` reads the ``groups`` table (application context required). Monitoring
passes consent dates as of the time being checked (from ``group_history``).
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any

from app.clock import to_study_local

WEEKDAY_CODES = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")

WARMUP_REASON_COHORT = "cohort"
WARMUP_REASON_WEEK1 = "week1"


# --------------------------------------------------------------------- dates


def as_date(value: Any) -> datetime.date:
    """
    Coerce ``value`` to a study-calendar ``date``: a ``date``; a ``datetime``
    (aware ones are first converted to Detroit local time); or an ISO string
    (``"YYYY-MM-DD"`` or a full timestamp). Raises ``ValueError`` otherwise.
    """
    if isinstance(value, datetime.datetime):
        return to_study_local(value).date()
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if len(text) == 10:
            return datetime.date.fromisoformat(text)
        return to_study_local(text).date()
    raise ValueError(f"expected a date, datetime or ISO string, got {type(value).__name__}.")


def week_monday(value: Any) -> datetime.date:
    """The Monday that starts the calendar week containing ``value``."""
    d = as_date(value)
    return d - datetime.timedelta(days=d.weekday())


def week_start_datetime(value: Any) -> datetime.datetime:
    """Monday 00:00 (naive Detroit local) of the calendar week containing ``value``."""
    return datetime.datetime.combine(week_monday(value), datetime.time.min)


def week1_start(consent_start: Any) -> datetime.date:
    """First Monday on or after ``consent_start`` (``enr_i`` in main_new.tex §5.2)."""
    d = as_date(consent_start)
    return d + datetime.timedelta(days=-d.weekday() % 7)


def study_week(consent_start: Any, current: Any) -> int:
    """``week_in_study`` of ``current`` (API-Spec §4.1): 0 in the partial week (or before), else 1-indexed."""
    start = week1_start(consent_start)
    d = as_date(current)
    if d < start:
        return 0
    return (d - start).days // 7 + 1


def day_in_study(consent_start: Any, current: Any) -> int:
    """``day_in_study`` of ``current`` (API-Spec §4.1): day 1 is ``consent_start``."""
    return (as_date(current) - as_date(consent_start)).days + 1


def parse_weekly_time(spec: str) -> tuple[int, datetime.time]:
    """
    Parse a weekly time setting such as ``"MON 05:45"`` (Monitoring-Spec §10)
    into ``(weekday, time)`` with Monday = 0.
    """
    try:
        day_code, hhmm = spec.strip().split()
        weekday = WEEKDAY_CODES.index(day_code.upper()[:3])
        hour, minute = (int(part) for part in hhmm.split(":"))
        return weekday, datetime.time(hour, minute)
    except Exception as exc:  # noqa: BLE001 - normalize every parse failure
        raise ValueError(f"weekly time {spec!r} is not of the form 'MON 05:45'.") from exc


def weekly_deadline(week_monday_date: Any, spec: str = "MON 05:45") -> datetime.datetime:
    """
    Naive Detroit-local datetime of ``spec`` in the calendar week starting at
    ``week_monday_date`` (e.g. the Monday 05:45 policy-publication deadline of
    contract D13; API-Spec §8).
    """
    weekday, at = parse_weekly_time(spec)
    day = week_monday(week_monday_date) + datetime.timedelta(days=weekday)
    return datetime.datetime.combine(day, at)


# -------------------------------------------------------------------- dyads


def consent_start_of(item: Any) -> datetime.date:
    """
    ``consent_start_date`` of a dyad record: a ``Group`` row (``group_info``),
    a ``GroupHistory`` row, a mapping with ``consent_start_date`` (or a
    ``group_info`` mapping), or the date itself.
    """
    if isinstance(item, (datetime.date, str)):
        return as_date(item)
    if isinstance(item, Mapping):
        if "consent_start_date" in item:
            return as_date(item["consent_start_date"])
        if isinstance(item.get("group_info"), Mapping):
            return as_date(item["group_info"]["consent_start_date"])
        raise ValueError("mapping has no consent_start_date.")
    direct = getattr(item, "consent_start_date", None)
    if direct is not None:
        return as_date(direct)
    info = getattr(item, "group_info", None)
    if isinstance(info, Mapping) and "consent_start_date" in info:
        return as_date(info["consent_start_date"])
    raise ValueError(f"cannot read consent_start_date from {type(item).__name__}.")


def _group_id_of(item: Any) -> str | None:
    if isinstance(item, Mapping):
        gid = item.get("group_id")
    else:
        gid = getattr(item, "group_id", None)
    return None if gid is None else str(gid)


def dyad_consents(groups: Any = None) -> list[tuple[str | None, datetime.date]]:
    """
    Normalize ``groups`` to ``[(group_id, consent_start_date), ...]``.

    ``groups`` may be ``None`` (read the ``groups`` table), a mapping
    ``{group_id: consent_start}``, or an iterable of ``Group`` /
    ``GroupHistory`` rows, mappings, or bare consent dates (``group_id`` is
    then ``None``). An iterable of rows keeps the last row per ``group_id``.
    """
    if groups is None:
        from app.models import Group

        groups = Group.query.order_by(Group.id.asc()).all()
    if isinstance(groups, Mapping):
        return [(str(gid), as_date(start)) for gid, start in groups.items()]

    out: dict[Any, tuple[str | None, datetime.date]] = {}
    anon = 0
    for item in groups:
        gid = _group_id_of(item) if not isinstance(item, (datetime.date, str)) else None
        if gid is None:
            out[("__anon__", anon)] = (None, consent_start_of(item))
            anon += 1
        else:
            out[gid] = (gid, consent_start_of(item))
    return list(out.values())


def enrolled_dyads(week_monday_date: Any, groups: Any = None) -> list[str]:
    """``E_W``: ids of dyads with ``week1_start <= W`` (eq. (cohort-index)), sorted."""
    monday = week_monday(week_monday_date)
    return sorted(
        gid for gid, start in dyad_consents(groups) if gid is not None and week1_start(start) <= monday
    )


def n_enrolled(week_monday_date: Any, groups: Any = None) -> int:
    """``|E_W|``, counting anonymous consent dates too."""
    monday = week_monday(week_monday_date)
    return sum(1 for _, start in dyad_consents(groups) if week1_start(start) <= monday)


def active_dyads(week_monday_date: Any, T: int | None = None, groups: Any = None) -> list[str]:
    """``I_W``: ids of dyads whose study week of ``W`` is in ``1..T`` (default ``T`` = ``trial_weeks``), sorted."""
    if T is None:
        T = _hp().trial_weeks
    monday = week_monday(week_monday_date)
    return sorted(
        gid
        for gid, start in dyad_consents(groups)
        if gid is not None and 1 <= study_week(start, monday) <= int(T)
    )


def cohort_warmup_active(week_monday_date: Any, groups: Any = None, *, n_warm: int | None = None) -> bool:
    """
    Cohort warm-up for decisions in the week of ``week_monday_date``
    (contract D2): ``|E_{W-1}| < n_warm``, i.e. fewer than ``n_warm`` dyads
    have ``week1_start <= W - 7 days`` (have completed week 1). Used by
    ``/action``, ``/update`` and monitoring A10.
    """
    if n_warm is None:
        n_warm = _hp().n_warm
    previous_monday = week_monday(week_monday_date) - datetime.timedelta(days=7)
    return n_enrolled(previous_monday, groups) < int(n_warm)


def per_dyad_warmup(consent_start: Any, decision_date: Any, t_warm: int | None = None) -> bool:
    """Per-dyad warm-up (contract D3): partial week (study week 0) or study weeks ``1..t_warm``."""
    if t_warm is None:
        t_warm = _hp().t_warm
    return study_week(consent_start, decision_date) <= int(t_warm)


def warmup_reason(
    consent_start: Any,
    decision_date: Any,
    groups: Any = None,
    *,
    n_warm: int | None = None,
    t_warm: int | None = None,
) -> str | None:
    """
    Scheduled warm-up reason of a decision (contract D3): ``"cohort"`` (checked
    first), then ``"week1"``, else ``None`` (the learned policy applies).
    """
    if cohort_warmup_active(week_monday(decision_date), groups, n_warm=n_warm):
        return WARMUP_REASON_COHORT
    if per_dyad_warmup(consent_start, decision_date, t_warm):
        return WARMUP_REASON_WEEK1
    return None


def _hp():
    from app.hyperparams import get_hyperparams

    return get_hyperparams()


__all__ = [
    "WEEKDAY_CODES",
    "WARMUP_REASON_COHORT",
    "WARMUP_REASON_WEEK1",
    "as_date",
    "week_monday",
    "week_start_datetime",
    "week1_start",
    "study_week",
    "day_in_study",
    "parse_weekly_time",
    "weekly_deadline",
    "consent_start_of",
    "dyad_consents",
    "enrolled_dyads",
    "n_enrolled",
    "active_dyads",
    "cohort_warmup_active",
    "per_dyad_warmup",
    "warmup_reason",
]
