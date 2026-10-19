"""
The scheduled monitoring checks (Monitoring-Spec §5) and when each runs.

Every entry of :data:`REGISTRY` is a pure check ``fn(ctx) -> list[Finding]``
with its schedule:

- ``every_run``: once per monitoring run over the run's window (the
  per-decision checks over decisions stamped in the window; the update checks
  over updates of the last ``UPDATE_LOOKBACK_DAYS`` days; the error-code
  alarms over answers in the window; E1, E6);
- ``event``: B11, at the arrival of each week's first ``/update`` in the window;
- ``daily HH:MM`` / ``weekly DOW HH:MM``: each scheduled time in the window,
  evaluated at that logical time (so a run after an outage catches up, E5).

The inline hooks run the same per-decision and per-update functions with the
same dedup keys; the runner is the safety net and what a rerun over a past
period uses.
"""

from __future__ import annotations

import dataclasses
import datetime
from collections.abc import Callable

from app.monitoring import catalog
from app.monitoring.checks import decisions, diagnostics, records, rerun, summary, system, update, uploads
from app.monitoring.timeutil import parse_hhmm

EVERY_RUN = "every_run"
EVENT = "event"
DAILY = "daily"
WEEKLY = "weekly"


@dataclasses.dataclass(frozen=True)
class Schedule:
    kind: str
    at: str | None = None            # "HH:MM", or "DOW HH:MM" for weekly
    config_key: str | None = None    # weekly time taken from this setting when present

    def spec(self) -> str | None:
        if self.config_key:
            from flask import current_app

            return str(current_app.config.get(self.config_key) or self.at)
        return self.at

    def period(self) -> datetime.timedelta:
        return {DAILY: datetime.timedelta(days=1), WEEKLY: datetime.timedelta(days=7)}.get(
            self.kind, datetime.timedelta(days=7) if self.kind == EVENT else datetime.timedelta(0)
        )

    def occurrences(self, start: datetime.datetime, end: datetime.datetime) -> list[datetime.datetime]:
        """Scheduled times in ``(start, end]`` (daily and weekly schedules)."""
        if self.kind not in (DAILY, WEEKLY):
            return []
        spec = self.spec()
        if self.kind == DAILY:
            weekday, at = None, parse_hhmm(spec)
        else:
            from app.study_calendar import parse_weekly_time

            weekday, at = parse_weekly_time(spec)
        out = []
        day = start.date()
        while day <= end.date():
            if weekday is None or day.weekday() == weekday:
                t = datetime.datetime.combine(day, at)
                if start < t <= end:
                    out.append(t)
            day += datetime.timedelta(days=1)
        return out


@dataclasses.dataclass(frozen=True)
class Item:
    item_id: str
    fn: Callable
    schedule: Schedule
    order: int = 50


def _items() -> dict[str, Item]:
    daily = Schedule(DAILY, catalog.DAILY_AT)
    nightly = Schedule(DAILY, catalog.NIGHTLY_AT)
    weekly_mon = Schedule(WEEKLY, f"MON {catalog.WEEKLY_MONDAY_AT}")
    run = Schedule(EVERY_RUN)
    items = [
        Item("codes", records.check_codes, run, 10),
        Item("A13", decisions.check_a13, run, 11),
        Item("A8", decisions.check_a8, run, 12),
        Item("A15", decisions.check_a15, run, 13),
        Item("A17", decisions.check_a17, run, 14),
        Item("C9", update.check_c9, run, 15),
        Item("B11", uploads.check_b11, Schedule(EVENT), 16),
        Item("C1", update.check_c1, run, 20),
        Item("C3", update.check_c3, run, 21),
        Item("B5", uploads.check_b5, run, 22),
        Item("C10", update.check_c10, run, 23),
        Item("L1", diagnostics.check_l1, run, 24),
        Item("L2", diagnostics.check_l2, run, 25),
        Item("L3", diagnostics.check_l3, run, 26),
        Item("E1", system.check_e1, run, 30),
        Item("E6", system.check_e6, run, 31),
        Item("A4", decisions.check_a4, nightly, 40),
        Item("A10", decisions.check_a10, nightly, 41),
        Item("C11", update.check_c11, Schedule(WEEKLY, "MON 03:30", "MONITOR_UPDATE_EXPECTED_BY"), 42),
        Item("A6", decisions.check_a6, weekly_mon, 43),
        Item("A7", decisions.check_a7, weekly_mon, 44),
        Item("L4", diagnostics.check_l4, weekly_mon, 45),
        Item("C7", update.check_c7, Schedule(WEEKLY, "MON 05:45", "MONITOR_WEEK_READY_BY"), 46),
        Item("C5", rerun.check_c5, Schedule(WEEKLY, catalog.C5_AT), 47),
        Item("A2", decisions.check_a2, daily, 50),
        Item("A5", decisions.check_a5, daily, 51),
        Item("A16", decisions.check_a16, daily, 52),
        Item("B8", uploads.check_b8, daily, 53),
        Item("B3", uploads.check_b3, daily, 54),
        Item("D1", records.check_d1, daily, 55),
        Item("D3", records.check_d3, Schedule(WEEKLY, catalog.D3_AT), 56),
        Item("G0", summary.check_g0, daily, 99),  # last: it counts the alarms of its own time
    ]
    return {item.item_id: item for item in items}


REGISTRY: dict[str, Item] = _items()

#: Checks that run only inline or at start-up, never on a schedule.
INLINE_ONLY = ("E5", "E7")


__all__ = ["Schedule", "Item", "REGISTRY", "EVERY_RUN", "EVENT", "DAILY", "WEEKLY", "INLINE_ONLY"]
