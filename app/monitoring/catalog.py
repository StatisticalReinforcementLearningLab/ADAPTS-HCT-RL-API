"""
The monitoring check list (Monitoring-Spec §5) and the error-code list
(Monitoring-Spec §6), version ``CHECK_LIST_VERSION`` (§11.4).

Every alarm row carries an ``item_id`` that is either a check ID of §5 (e.g.
``"C7"``) or an error code of §6 (e.g. ``"AC-201-NOSTATE"``). This module is
the single place that says, per item: its default severity (red / yellow /
green, §4.1), its owner (``rl`` / ``host``, §4.2), where it runs (§2.1), its
cadence, the automatic response (fallback, §7) and its escalation rule.
:func:`lookup` also covers the open family ``XX-<status>-UNMAPPED`` that the
after-response step uses for an answer that is not in §6 (contract D17; D3
reports them).

Thresholds inside the checks belong to the check list too (§10 last
paragraph): they are the module constants below, not settings. Changing one
means a new ``CHECK_LIST_VERSION`` and a change-log row in Monitoring-Spec.
"""

from __future__ import annotations

import dataclasses
import re
from types import MappingProxyType

#: Version of the Monitoring-Spec check list in force (Monitoring-Spec §11.4, change log 0.2).
CHECK_LIST_VERSION = "0.2"

RED = "red"
YELLOW = "yellow"
GREEN = "green"
SEVERITIES = (RED, YELLOW, GREEN)

OWNER_RL = "rl"
OWNER_HOST = "host"
OWNERS = (OWNER_RL, OWNER_HOST)

# ``monitor_alarms.source`` values (Monitoring-Spec §9.1), plus ``startup`` for
# the C9 start-up refusal (Hyperparameters.md §2.3), which happens before any
# request is served.
SOURCE_INSIDE_ACTION = "inside_action"
SOURCE_AFTER_RESPONSE = "after_response"
SOURCE_BEFORE_PUBLISH = "before_publish"
SOURCE_SCHEDULED = "scheduled"
SOURCE_ERROR_CODE = "error_code"
SOURCE_STARTUP = "startup"
SOURCES = (
    SOURCE_INSIDE_ACTION,
    SOURCE_AFTER_RESPONSE,
    SOURCE_BEFORE_PUBLISH,
    SOURCE_SCHEDULED,
    SOURCE_ERROR_CODE,
    SOURCE_STARTUP,
)

# ----------------------------------------------------- check-list thresholds

#: A13 tolerance on the allocation bounds (contract D12).
A13_PI_TOL = 1e-12
#: A4 tolerance on a recomputed pi (Monitoring-Spec A4 (1)).
A4_PI_TOL = 1e-12
#: A5: share of active dyads at a limit, minimum such dyads, closeness, run length.
A5_SHARE = 0.80
A5_MIN_DYADS = 5
A5_LIMIT_DISTANCE = 0.01
A5_DAYS = 7
#: B3: change in the share of "miss" per field against the previous 7 days.
B3_MISS_DELTA = 0.15
B3_BASELINE_DAYS = 7
#: B8: an upload must precede each decision within this many minutes.
B8_UPLOAD_MINUTES = 30
#: C3 numerics (Monitoring-Spec C3 (2)-(4)) are applied inside the learner's gate.
#: D1: day-over-day row-count tolerance and manifest freshness.
D1_ROW_TOLERANCE = 0.20
D1_MAX_AGE_HOURS = 26
#: E1 limits in percent (memory, disk, database connections).
E1_LIMITS = MappingProxyType({"memory_pct": 85.0, "disk_pct": 90.0, "db_conn_pct": 90.0})
#: E5: a gap longer than this since the last run means monitoring was down.
E5_GAP_MINUTES = 15
#: E6: no successful host pull for this long.
E6_GAP_HOURS = 24
#: L1: a between-dyad variance within this fraction of its floor.
L1_FLOOR_REL = 0.01
#: L3: spectral-norm threshold of G X' X_+.
L3_NORM_THRESHOLD = 1.0
#: L4: reported percentiles of pi and the closeness to a limit.
L4_PERCENTILES = (5, 25, 50, 75, 95)
L4_LIMIT_DISTANCE = 0.01
#: How far back the every-run update checks (B5, C1, C3, C9, C10, L1-L3) look
#: for finished updates; dedup keys make the overlap harmless.
UPDATE_LOOKBACK_DAYS = 8

# ------------------------------------------------------------ scheduled times
# Times of the scheduled checks (naive America/Detroit; Monitoring-Spec §5).
# C11 and C7 follow the settings MONITOR_UPDATE_EXPECTED_BY / MONITOR_WEEK_READY_BY.
NIGHTLY_AT = "01:00"        # A4, A10: "every night, for all of yesterday's decisions"
DAILY_AT = "07:00"          # A2 ("daily 07:00"), A5, A16, B8, B3, D1, G0 ("daily 07:00")
WEEKLY_MONDAY_AT = "04:00"  # A6, A7, L4: after the week-closing uploads and the 03:00 update
C5_AT = "TUE 02:00"         # C5: "weekly, Tuesday 02:00"
D3_AT = "MON 07:00"         # D3: weekly, over the week that just ended


@dataclasses.dataclass(frozen=True)
class CatalogEntry:
    """One item of the check list (§5) or the error-code list (§6)."""

    item_id: str
    severity: str
    owner: str
    title: str
    runs: str
    where: str
    fallback: str | None = None
    escalation: str | None = None
    kind: str = "check"            # "check" | "code"
    alarm: bool = True             # False only for AC-200-REPLAY (logged, not an alarm)
    http_status: int | None = None


def _check(item_id, severity, owner, title, runs, where, fallback=None, escalation=None):
    return CatalogEntry(item_id, severity, owner, title, runs, where, fallback, escalation)


def _code(item_id, severity, owner, title, fallback=None, alarm=True):
    status = int(item_id.split("-")[1]) if item_id.split("-")[1].isdigit() else None
    return CatalogEntry(
        item_id,
        severity,
        owner,
        title,
        "when the endpoint returns it",
        SOURCE_ERROR_CODE,
        fallback,
        None,
        "code",
        alarm,
        status,
    )


_CHECKS = (
    # 5.A Decisions
    _check("A13", RED, OWNER_RL, "pi out of range", "inside /action", SOURCE_INSIDE_ACTION, "F-A2"),
    _check("A4", RED, OWNER_RL, "Each decision can be reproduced", "nightly, for all of yesterday's decisions",
           SOURCE_SCHEDULED),
    _check("A10", RED, OWNER_RL, "Warm-up follows the algorithm's rules", "nightly, for all of yesterday's decisions",
           SOURCE_SCHEDULED),
    _check("A6", YELLOW, OWNER_RL, "Too many messages", "weekly, Monday", SOURCE_SCHEDULED),
    _check("A7", YELLOW, OWNER_RL, "Too few messages", "weekly, Monday", SOURCE_SCHEDULED),
    _check("A16", YELLOW, OWNER_HOST, "Delivery does not match the action", "daily", SOURCE_SCHEDULED),
    _check("A17", YELLOW, OWNER_RL, "Game assignment has not varied for three weeks",
           "after each dyad_game action", SOURCE_AFTER_RESPONSE),
    _check("A2", YELLOW, OWNER_HOST, "Number of calls per day", "daily 07:00, for the previous day",
           SOURCE_SCHEDULED, "F-A1"),
    _check("A8", YELLOW, OWNER_HOST, "Outside the consent dates", "after each /action", SOURCE_AFTER_RESPONSE),
    _check("A15", YELLOW, OWNER_HOST, "Game status does not match", "after each /action", SOURCE_AFTER_RESPONSE),
    _check("A5", YELLOW, OWNER_RL, "The whole cohort is at the limit", "daily", SOURCE_SCHEDULED),
    # 5.B Registration, uploads, rewards
    _check("B11", YELLOW, OWNER_HOST, "Week-closing upload missing", "when the update arrives",
           SOURCE_AFTER_RESPONSE, None, "red if the same dyad misses two weeks in a row"),
    _check("B5", YELLOW, OWNER_RL, "Dyad has no policy this week", "each weekly update", SOURCE_AFTER_RESPONSE,
           "F-A3"),
    _check("B8", YELLOW, OWNER_HOST, "Upload missing before a decision", "daily", SOURCE_SCHEDULED),
    _check("B3", GREEN, OWNER_HOST, "Missing-data rate changed", "daily", SOURCE_SCHEDULED),
    # 5.C The weekly update
    _check("C7", YELLOW, OWNER_RL, "Week not ready at 05:45", "Monday 05:45", SOURCE_SCHEDULED, "F-U2/F-A3"),
    _check("C11", RED, OWNER_HOST, "Update never arrived", "Monday 03:30", SOURCE_SCHEDULED),
    _check("C9", RED, OWNER_RL, "Unapproved settings used",
           "after each decision, at each update, and at start-up", SOURCE_AFTER_RESPONSE),
    _check("C3", YELLOW, OWNER_RL, "Validity of model parameters", "inside the update", SOURCE_BEFORE_PUBLISH,
           "F-U2", "red if the check fails two weeks in a row for the same agent"),
    _check("C1", YELLOW, OWNER_RL, "Update failed or fell back", "every 5 min; after each update",
           SOURCE_AFTER_RESPONSE, "F-U2, F-U5", "red after two weeks in a row for the same agent"),
    _check("C5", YELLOW, OWNER_RL, "Update can be rerun exactly", "weekly, Tuesday 02:00", SOURCE_SCHEDULED,
           None, "red after two weeks in a row"),
    _check("C10", GREEN, OWNER_RL, "Noise scale set", "after the first learning update", SOURCE_AFTER_RESPONSE,
           None, "yellow when the upper limit was used"),
    # 5.L Model diagnostics
    _check("L1", GREEN, OWNER_RL, "Variance at its floor", "after each update", SOURCE_AFTER_RESPONSE),
    _check("L2", GREEN, OWNER_RL, "How much each dyad's own data counts", "after each update",
           SOURCE_AFTER_RESPONSE),
    _check("L3", YELLOW, OWNER_RL, "Learning step may not settle", "after each update", SOURCE_AFTER_RESPONSE,
           None, "the week-to-week change of theta for dyads whose data no longer change is reported green"),
    _check("L4", GREEN, OWNER_RL, "Spread of pi", "weekly", SOURCE_SCHEDULED),
    # 5.D Saved records
    _check("D1", GREEN, OWNER_RL, "Daily export", "daily", SOURCE_SCHEDULED),
    _check("D3", GREEN, OWNER_RL, "Error list out of date", "weekly", SOURCE_SCHEDULED),
    # 5.E Server, alarm endpoint, monitoring itself
    _check("E1", RED, OWNER_RL, "Server running out of resources", "every 5 min", SOURCE_SCHEDULED),
    _check("E5", YELLOW, OWNER_RL, "Monitoring was down", "at start", SOURCE_SCHEDULED),
    _check("E6", YELLOW, OWNER_HOST, "Host stopped collecting alarms", "every 5 min", SOURCE_SCHEDULED),
    _check("E7", YELLOW, OWNER_RL, "A monitoring check crashed", "when it happens", SOURCE_SCHEDULED),
    # 5.G Summaries
    _check("G0", GREEN, OWNER_RL, "Daily status", "daily 07:00", SOURCE_SCHEDULED),
)

_CODES = (
    # 6.2 /register_group
    _code("RG-400-PARAM", YELLOW, OWNER_HOST, "Registration rejected"),
    _code("RG-409-MEMBER", RED, OWNER_HOST, "Registration with different members"),
    _code("RG-500-INTERNAL", YELLOW, OWNER_RL, "Registration failed: server error"),
    _code("RG-503-DB", YELLOW, OWNER_RL, "Registration not saved"),
    # 6.3 /action
    _code("AC-201-NOSTATE", YELLOW, OWNER_HOST, "50/50 no-state fallback", "F-A2"),
    _code("AC-201-FALLBACK", YELLOW, OWNER_RL, "50/50 API fallback", "F-A2/F-A3"),
    _code("AC-200-REPLAY", GREEN, OWNER_HOST, "Repeated decision request (logged, not an alarm)", alarm=False),
    _code("AC-400-PARAM", RED, OWNER_HOST, "Decision request rejected", "F-A1"),
    _code("AC-404-GROUP", RED, OWNER_HOST, "Decision for an unregistered dyad", "F-A1"),
    _code("AC-500-PERSIST", RED, OWNER_RL, "Decision could not be saved", "F-A1"),
    _code("AC-503-STORE", RED, OWNER_RL, "Decision store unavailable", "F-A1"),
    # 6.4 /upload_data
    _code("UD-400-PARAM", YELLOW, OWNER_HOST, "Upload rejected: request fields", "F-U1"),
    _code("UD-400-UNKNOWN", YELLOW, OWNER_HOST, "Upload rejected: unknown field", "F-U1"),
    _code("UD-400-MISSINGFIELD", YELLOW, OWNER_HOST, "Upload rejected: field missing", "F-U1"),
    _code("UD-400-TYPE", YELLOW, OWNER_HOST, "Upload rejected: wrong type", "F-U1"),
    _code("UD-400-RANGE", YELLOW, OWNER_HOST, "Upload rejected: value out of range", "F-U1"),
    _code("UD-400-INCONSISTENT", YELLOW, OWNER_HOST, "Upload rejected: values do not fit together", "F-U1"),
    _code("UD-404-GROUP", YELLOW, OWNER_HOST, "Upload for an unregistered dyad"),
    _code("UD-500-INTERNAL", YELLOW, OWNER_RL, "Upload failed: server error"),
    _code("UD-503-DB", YELLOW, OWNER_RL, "Upload not saved"),
    # 6.5 /update. UP-500-INTERNAL is in §6.5 but has no §5.C row; it takes the
    # severity and owner of the other endpoints' 500 rows (RG/UD-500-INTERNAL).
    _code("UP-400-PARAM", YELLOW, OWNER_HOST, "Update request rejected"),
    _code("UP-500-INTERNAL", YELLOW, OWNER_RL, "Update request failed: server error"),
    _code("UP-503-ENQUEUE", RED, OWNER_RL, "Update could not be started"),
    # 6.6 /monitor/alarms
    _code("MA-500-INTERNAL", YELLOW, OWNER_RL, "Alarm request failed: server error"),
    _code("MA-503-STORE", YELLOW, OWNER_RL, "Alarms could not be read"),
    # 6.7 framework-level answers
    _code("XX-404-ROUTE", YELLOW, OWNER_HOST, "Unknown address"),
    _code("XX-405-METHOD", YELLOW, OWNER_HOST, "Wrong HTTP method"),
    _code("XX-500-UNCAUGHT", RED, OWNER_RL, "Unhandled crash"),
)

CHECKS: "MappingProxyType[str, CatalogEntry]" = MappingProxyType({e.item_id: e for e in _CHECKS})
CODES: "MappingProxyType[str, CatalogEntry]" = MappingProxyType({e.item_id: e for e in _CODES})

#: ``XX-<status>-UNMAPPED``: an answer missing from §6 (contract D17). Not in
#: the spec's tables; yellow and owned by the RL team, who must add the answer
#: to §6 (D3 reports these weekly).
UNMAPPED_RE = re.compile(r"^XX-(\d{3})-UNMAPPED$")


def unmapped_code(status: int) -> str:
    """The code of an answer that is not in Monitoring-Spec §6 (contract D17)."""
    return f"XX-{int(status):03d}-UNMAPPED"


def is_unmapped(code: str | None) -> bool:
    return bool(code) and UNMAPPED_RE.match(code) is not None


def lookup(item_id: str) -> CatalogEntry:
    """
    The catalog entry of ``item_id`` (a §5 check ID, a §6 code, or
    ``XX-<status>-UNMAPPED``). Raises ``KeyError`` for anything else, so an
    alarm can never be written for an item that is not in the check list.
    """
    if item_id in CHECKS:
        return CHECKS[item_id]
    if item_id in CODES:
        return CODES[item_id]
    match = UNMAPPED_RE.match(item_id or "")
    if match:
        return CatalogEntry(
            item_id,
            YELLOW,
            OWNER_RL,
            f"Answer not in the error-code list (HTTP {match.group(1)})",
            "when the endpoint returns it",
            SOURCE_ERROR_CODE,
            None,
            None,
            "code",
            True,
            int(match.group(1)),
        )
    raise KeyError(f"unknown monitoring item {item_id!r}")


def all_entries() -> list[CatalogEntry]:
    """Every check then every code, in Monitoring-Spec order."""
    return list(_CHECKS) + list(_CODES)


__all__ = [
    "CHECK_LIST_VERSION",
    "RED",
    "YELLOW",
    "GREEN",
    "SEVERITIES",
    "OWNER_RL",
    "OWNER_HOST",
    "OWNERS",
    "SOURCES",
    "SOURCE_INSIDE_ACTION",
    "SOURCE_AFTER_RESPONSE",
    "SOURCE_BEFORE_PUBLISH",
    "SOURCE_SCHEDULED",
    "SOURCE_ERROR_CODE",
    "SOURCE_STARTUP",
    "CatalogEntry",
    "CHECKS",
    "CODES",
    "lookup",
    "all_entries",
    "unmapped_code",
    "is_unmapped",
]
