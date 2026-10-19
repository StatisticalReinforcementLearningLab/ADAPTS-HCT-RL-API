"""
ADAPTS-HCT host simulator: drives the RL API exactly the way the study host does.

The host contract (API-Spec §2, §3, §4.1) is followed literally:

- ``/register_group`` once per dyad at recruitment (12:00 the day before its
  ``consent_start_date``). Consent starts fall on any weekday; the consent
  window covers the dyad's partial week plus ``T = 14`` Monday-Sunday study
  weeks (``consent_end_date`` = the Sunday that ends study week 14, a 98-104
  day window), so no decision falls outside it.
- Decision calendar (API-Spec §3): AYA-AM 07:30, CP 07:35 and AYA-PM 19:30
  every day from ``consent_start_date``; the weekly ``dyad_game`` decision at
  Monday 06:00 from study week 1 (none in the partial week).
- Exactly one ``/upload_data`` 2-5 minutes before every ``/action``
  (API-Spec §2.3 upload schedule) and the week-closing upload at Monday
  00:30 (pre-``dyad_game`` convention, ``current_game_on = "miss"``, final
  delivered counts) for every dyad that had a decision in the week just
  ended.
- ``/update`` every Monday 03:00 once the study has started (a dyad is
  registered); ``GET /monitor/alarms`` at 05:50 and 17:50, deduplicated by
  ``event_id`` (API-Spec §2.5).
- ``decision_idx``: per-(dyad, agent) counters starting at 1 with the first
  partial-week decision; ``dyad_game`` uses the study week ``w``
  (API-Spec §3 "``decision_idx = w``").
- Every snapshot field is computed per API-Spec §4.1 from a simulated
  history (MTW adherence reports, diaries 1-5 or ``"miss"``, app opens,
  Sunday-evening relationship surveys, the host's delivery log) and checked
  by :class:`SnapshotAuditor` before it is sent.

Outcomes come from a small latent model (:class:`OutcomeModel`) with modest
positive message effects and a burden penalty, so learning is visible. Every
random draw is keyed by ``(seed, dyad, purpose, date)``: a run is
deterministic given the server's answers, and one dyad's outcomes do not
depend on other dyads or on the order of events.

Transports (all return ``(status, json)``): :class:`FlaskClientTransport`
(in-process Flask test client), :class:`HttpTransport` (``requests`` against
a base URL) and :class:`OfflineTransport` (no server; answers every
``/action`` with a Bernoulli(1/2) warm-up draw).

Host-side faults of the Monitoring-Spec §11.3 fault table are
:class:`HostFaults`. The simulator records every request/response and fails
loudly (``strict``) on any status that is not expected for the call or on
the fault allow-list.

This module deliberately imports nothing from ``app``: it is the host side
of the contract and must run against a remote server.
"""

from __future__ import annotations

import bisect
import datetime as dt
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol
from zoneinfo import ZoneInfo

# ----------------------------------------------------------------- constants

API_PREFIX = "/api/v1"
STUDY_TZ = ZoneInfo("America/Detroit")  # API-Spec §3, §8
MISS = "miss"
AGENTS = ("aya_message", "cp_message", "dyad_game")
TRIAL_WEEKS = 14  # main_new.tex Table 1 (trial_weeks)
GAMMA = {"aya": 13 / 14, "cp": 6 / 7}  # API-Spec §4.1 burden discounts (= learner discounts)
BURDEN_BOUND = {"aya": 1.0 / (1.0 - 13 / 14), "cp": 1.0 / (1.0 - 6 / 7)}
ROLE_OF = {"aya_message": "aya", "cp_message": "cp", "dyad_game": "game"}

# Host clock (API-Spec §3 worked example; Detroit local wall time).
T_REGISTER = dt.time(12, 0)
T_CLOSING = dt.time(0, 30)
CLOSING_SPACING = dt.timedelta(seconds=20)
T_CLOSING_LATEST = dt.time(2, 45)
T_UPDATE = dt.time(3, 0)
T_GAME = dt.time(6, 0)
T_AYA_AM = dt.time(7, 30)
T_CP = dt.time(7, 35)
T_AYA_PM = dt.time(19, 30)
T_DELIVERY = {
    ("aya_message", "am"): dt.time(7, 45),
    ("cp_message", "am"): dt.time(7, 50),
    ("aya_message", "pm"): dt.time(19, 45),
}
T_MTW_END = {"am": dt.time(10, 0), "pm": dt.time(22, 0)}  # adherence (and evening diaries) final
T_DIARY = T_MTW_END["pm"]
T_SURVEY = dt.time(21, 0)  # Sunday evening relationship survey
T_END_OF_DAY = dt.time(23, 59, 59)
PULL_TIMES = (dt.time(5, 50), dt.time(17, 50))
UPLOAD_LEAD_SECONDS = (120, 299)  # "2-5 minutes" before the /action
BAD_UPLOAD_LEAD = dt.timedelta(seconds=60)  # rejected upload goes out this much before the corrected one
NOON_HOUR = 12  # slot boundary (UPLOAD_SLOT_NOON_HOUR)

# Upload kinds (API-Spec §2.3 schedule).
CLOSING = "closing"
PRE_GAME = "pre_game"
PRE_AYA_AM = "pre_aya_am"
PRE_CP = "pre_cp"
PRE_AYA_PM = "pre_aya_pm"
UPLOAD_KINDS = (CLOSING, PRE_GAME, PRE_AYA_AM, PRE_CP, PRE_AYA_PM)
AM_KINDS = (CLOSING, PRE_GAME, PRE_AYA_AM, PRE_CP)

# Endpoints and the status each ordinary call must get.
ENDPOINT_PATHS = {
    "register_group": API_PREFIX + "/register_group",
    "upload_data": API_PREFIX + "/upload_data",
    "action": API_PREFIX + "/action",
    "update": API_PREFIX + "/update",
    "monitor_alarms": API_PREFIX + "/monitor/alarms",
}
EXPECTED_STATUS = {
    "register_group": 201,
    "upload_data": 201,
    "action": 201,
    "update": 202,
    "monitor_alarms": 200,
}
ENVELOPE_KEYS = ("status", "title", "message", "server_timestamp", "rid")

# API-Spec §4.1 field dictionary, in its order, with each field's encoding.
SNAPSHOT_FIELDS: dict[str, str] = {
    # bookkeeping
    "day_in_study": "positive_int",
    "week_in_study": "nonneg_int",
    "slot": "slot",
    # AYA-side
    "aya_diary_mood": "item_1_5_or_miss",
    "aya_diary_physical": "item_1_5_or_miss",
    "aya_app_engagement": "engagement_or_miss",
    "aya_app_burden": "nonneg_float",
    "aya_missing_rate_7d": "unit_interval",
    "previous_med_adherence": "binary_or_miss",
    "prompted_by_message": "bool",
    "aya_messages_delivered_week": "nonneg_int",
    # CP-side
    "cp_diary_mood": "item_1_5_or_miss",
    "cp_app_engagement": "engagement_or_miss",
    "cp_app_burden": "nonneg_float",
    "cp_missing_rate_7d": "unit_interval",
    "daily_diary_completed": "bool",
    "cp_prompted_by_message": "bool",
    "cp_messages_delivered_week": "nonneg_int",
    # dyad-level
    "relationship_quality_aya": "score_1_5_or_miss",
    "relationship_quality_cp": "score_1_5_or_miss",
    "current_game_on": "binary_or_miss",
    "prior_game_action": "binary_or_miss",
    "aya_diary_summary": "unit_interval_or_miss",
    "cp_diary_summary": "unit_interval_or_miss",
    "weekly_survey_completed": "bool",
    "weekly_relationship_score": "nonneg_float",
}

FLOAT_TOL = 1e-9


# --------------------------------------------------------------- exceptions


class SimulationError(AssertionError):
    """The simulated host met something it must not ignore (strict mode)."""


class UnexpectedStatusError(SimulationError):
    """A call got a status that is neither expected for it nor on the fault allow-list."""


class ContractViolationError(SimulationError):
    """A response breaks the API-Spec envelope / endpoint shape."""


class SnapshotValidationError(SimulationError):
    """A snapshot the host was about to send breaks an API-Spec §4.1 rule."""


class TransportError(RuntimeError):
    """The transport could not reach the API (connection refused, timeout)."""


# ------------------------------------------------------------ study calendar


def week_monday(day: dt.date) -> dt.date:
    """Monday of the Monday-Sunday calendar week containing ``day``."""
    return day - dt.timedelta(days=day.weekday())


def week1_start(consent_start: dt.date) -> dt.date:
    """First Monday on or after ``consent_start`` (API-Spec §2.1; ``enr_i``)."""
    return consent_start + dt.timedelta(days=(-consent_start.weekday()) % 7)


def study_week(consent_start: dt.date, day: dt.date) -> int:
    """``week_in_study`` (API-Spec §4.1): 0 in the partial week, then 1-indexed Monday-Sunday weeks."""
    start = week1_start(consent_start)
    return 0 if day < start else (day - start).days // 7 + 1


def day_in_study(consent_start: dt.date, day: dt.date) -> int:
    """``day_in_study`` (API-Spec §4.1): day 1 is ``consent_start``."""
    return (day - consent_start).days + 1


def at(day: dt.date, clock: dt.time) -> dt.datetime:
    """Naive Detroit-local datetime of ``clock`` on ``day``."""
    return dt.datetime.combine(day, clock)


def consent_end_for(consent_start: dt.date, trial_weeks: int = TRIAL_WEEKS) -> dt.date:
    """The Sunday that ends study week ``trial_weeks`` (the last decision day)."""
    return week1_start(consent_start) + dt.timedelta(days=7 * trial_weeks - 1)


def format_timestamp(value: dt.datetime, tz_aware: bool = False) -> str:
    """ISO-8601 host timestamp: naive Detroit local time, or with the Detroit UTC offset."""
    if tz_aware:
        return value.replace(tzinfo=STUDY_TZ).isoformat(timespec="seconds")
    return value.isoformat(timespec="seconds")


def _rng(seed: int, *keys: Any) -> random.Random:
    """A private RNG for one draw, keyed by ``(seed, *keys)`` (string seeds hash deterministically)."""
    return random.Random("|".join(str(k) for k in (seed, *keys)))


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


# ----------------------------------------------------------------- transports


class Transport(Protocol):
    """How the simulated host reaches the API. Returns ``(status, json-or-None)``."""

    def request(self, method: str, path: str, json_body: dict | None = None) -> tuple[int, dict | None]: ...


class FlaskClientTransport:
    """In-process transport over a Flask test client (``app.test_client()``)."""

    def __init__(self, client):
        self.client = client

    def request(self, method: str, path: str, json_body: dict | None = None) -> tuple[int, dict | None]:
        if json_body is None:
            response = self.client.open(path, method=method)
        else:
            response = self.client.open(path, method=method, json=json_body)
        return response.status_code, response.get_json(silent=True)


class HttpTransport:
    """HTTP transport (``requests``) against ``base_url`` (e.g. ``http://127.0.0.1:5001``)."""

    def __init__(self, base_url: str, timeout: float = 120.0, session=None):
        import requests

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._requests = requests
        self.session = session or requests.Session()

    def request(self, method: str, path: str, json_body: dict | None = None) -> tuple[int, dict | None]:
        url = self.base_url + path
        try:
            response = self.session.request(method, url, json=json_body, timeout=self.timeout)
        except self._requests.RequestException as exc:
            raise TransportError(f"{method} {url}: {exc}") from exc
        try:
            body = response.json()
        except ValueError:
            body = None
        return response.status_code, body if isinstance(body, dict) else None


class OfflineTransport:
    """
    No server: answers like the API would during warm-up (API-Spec §2), for
    host-side tests and resource estimates. ``/action`` draws Bernoulli(1/2)
    from its own seeded RNG (or ``action_fn(payload) -> 0/1``) with
    ``warmup = true``; ``/upload_data`` rejects (400) a snapshot whose fields
    break their §4.1 encodings, like the API's field-level validation.
    """

    def __init__(self, seed: int = 0, action_fn: Callable[[dict], int] | None = None):
        self.rng = random.Random(seed)
        self.action_fn = action_fn
        self.calls: list[tuple[str, str, dict | None]] = []
        self._n = 0

    def _envelope(self, status: int, title: str, message: str, **extra) -> tuple[int, dict]:
        self._n += 1
        body = {
            "status": status,
            "title": title,
            "message": message,
            "server_timestamp": "1970-01-01T00:00:00",
            "rid": f"offline-{self._n:06d}",
        }
        body.update(extra)
        return status, body

    def request(self, method: str, path: str, json_body: dict | None = None) -> tuple[int, dict | None]:
        self.calls.append((method, path, json_body))
        if method == "POST" and path == ENDPOINT_PATHS["register_group"]:
            return self._envelope(201, "Success", "Group registered successfully.", **json_body)
        if method == "POST" and path == ENDPOINT_PATHS["upload_data"]:
            issues = field_encoding_issues(json_body.get("data", {}))
            if issues:
                return self._envelope(400, "Invalid Parameter", issues[0])
            return self._envelope(201, "Success", "Snapshot accepted.")
        if method == "POST" and path == ENDPOINT_PATHS["action"]:
            action = int(self.action_fn(json_body)) if self.action_fn else int(self.rng.random() < 0.5)
            return self._envelope(
                201,
                "Success",
                "Action requested successfully.",
                group_id=json_body["group_id"],
                decision_type=json_body["decision_type"],
                decision_idx=json_body["decision_idx"],
                action=action,
                action_prob=0.5,
                warmup=True,
                state=None,
                model_param=None,
            )
        if method == "POST" and path == ENDPOINT_PATHS["update"]:
            return self._envelope(202, "Update Accepted", "Model update started.")
        if method == "GET" and path == ENDPOINT_PATHS["monitor_alarms"]:
            return self._envelope(200, "Success", "Returning 0 alarms.", count=0, alarms=[])
        return self._envelope(404, "Not Found", "Unknown route.")


# --------------------------------------------------------------- the world


@dataclass(frozen=True)
class OutcomeModel:
    """
    Latent outcome model of the simulated participants. Effects are on the
    probability scale (clipped to [0.05, 0.95]) or on the raw 1-5 item scale.
    Delivered messages help modestly; the dose-trace burden hurts a little,
    so a policy that sends somewhat more than half the time is best.
    """

    msg_adherence: float = 0.12  # AYA message delivered for this MTW
    msg_response: float = 0.04  # ... also raises the chance the adherence question is answered
    burden_adherence: float = 0.006  # per unit of aya_app_burden (0..14)
    game_adherence: float = 0.03
    msg_aya_mood: float = 0.3  # any AYA message delivered that day, on the 1-5 mood item
    msg_aya_diary: float = 0.05  # AYA-PM message delivered, on diary submission
    burden_aya_diary: float = 0.004
    msg_cp_diary: float = 0.10  # CP message delivered that morning, on CP diary submission
    burden_cp_diary: float = 0.02  # per unit of cp_app_burden (0..7)
    msg_cp_mood: float = 0.4
    game_cp_mood: float = 0.2
    game_aya_mood: float = 0.15
    game_relq: float = 0.4  # game on this week, on each relationship-survey item
    msg_open: float = 0.10  # any message delivered that day, on opening the app
    mood_sd: float = 0.8
    relq_sd: float = 0.7
    mood_only_physical: float = 0.05  # submitted diary with mood skipped
    physical_skipped: float = 0.03  # submitted diary with physical skipped
    survey_items: int = 4


@dataclass(frozen=True)
class DyadLatent:
    """Per-dyad baseline propensities (drawn once from the seed)."""

    adh_base: float
    adh_response: float
    aya_diary_rate: float
    aya_mood: float
    aya_physical: float
    aya_open: float
    cp_diary_rate: float
    cp_mood: float
    cp_open: float
    rel_level: float
    aya_survey_rate: float
    cp_survey_rate: float

    @classmethod
    def draw(cls, seed: int, group_id: str) -> "DyadLatent":
        r = _rng(seed, group_id, "latent")
        return cls(
            adh_base=r.uniform(0.45, 0.75),
            adh_response=r.uniform(0.80, 0.95),
            aya_diary_rate=r.uniform(0.60, 0.90),
            aya_mood=r.uniform(2.5, 4.0),
            aya_physical=r.uniform(2.5, 4.0),
            aya_open=r.uniform(0.50, 0.85),
            cp_diary_rate=r.uniform(0.55, 0.90),
            cp_mood=r.uniform(2.5, 4.0),
            cp_open=r.uniform(0.50, 0.85),
            rel_level=r.uniform(2.5, 4.0),
            aya_survey_rate=r.uniform(0.75, 0.95),
            cp_survey_rate=r.uniform(0.75, 0.95),
        )


@dataclass
class DyadSpec:
    """A dyad as registered (API-Spec §2.1)."""

    index: int
    group_id: str
    member_list: list[str]
    consent_start: dt.date
    consent_end: dt.date

    @property
    def week1_start(self) -> dt.date:
        return week1_start(self.consent_start)

    @property
    def registration_time(self) -> dt.datetime:
        return at(self.consent_start - dt.timedelta(days=1), T_REGISTER)

    def registration_payload(self) -> dict:
        return {
            "group_id": self.group_id,
            "member_list": list(self.member_list),
            "consent_start_date": self.consent_start.isoformat(),
            "consent_end_date": self.consent_end.isoformat(),
        }


@dataclass
class Decision:
    """One scheduled decision and what became of it."""

    group_id: str
    decision_type: str
    decision_idx: int
    slot: str
    day: dt.date
    time: dt.datetime
    upload_time: dt.datetime
    kind: str
    study_week: int
    calendar_week: int
    action: int | None = None
    action_prob: float | None = None
    warmup: bool | None = None
    source: str | None = None  # "api", "host_skip" (fault: no call), "host_fallback" (F-A1 after an error)
    status: int | None = None
    rid: str | None = None
    delivered: bool | None = None
    burden_before: float | None = None
    state_len: int | None = None
    model_param_len: int | None = None

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.group_id, self.decision_type, self.decision_idx)

    @property
    def role(self) -> str:
        return ROLE_OF[self.decision_type]


class DyadHost:
    """
    Everything the host knows about one dyad: the delivery log, MTW adherence
    reports, diaries, app-open analytics, weekly surveys and game actions.
    :meth:`snapshot` turns it into the API-Spec §4.1 upload at a time ``t``.
    """

    def __init__(self, spec: DyadSpec, latent: DyadLatent, seed: int, trial_weeks: int = TRIAL_WEEKS):
        self.spec = spec
        self.latent = latent
        self.seed = seed
        self.trial_weeks = trial_weeks
        self.decisions: dict[tuple[str, int], Decision] = {}
        self.aya_by_mtw: dict[tuple[dt.date, str], Decision] = {}
        self.cp_by_day: dict[dt.date, Decision] = {}
        self.game_by_week: dict[int, int] = {}
        self.stuck_week: int | None = None
        self.adherence: dict[tuple[dt.date, str], Any] = {}
        self.aya_diary: dict[dt.date, tuple[Any, Any]] = {}
        self.cp_diary: dict[dt.date, Any] = {}
        self.opens: dict[str, dict[dt.date, bool]] = {"aya": {}, "cp": {}}
        self.surveys: list[tuple[dt.datetime, Any, Any]] = []
        self._delivery_times: dict[str, list[dt.datetime]] = {"aya": [], "cp": []}
        self._delivery_flags: dict[str, list[bool]] = {"aya": [], "cp": []}
        self._burden_after: dict[str, list[float]] = {"aya": [], "cp": []}

    @property
    def group_id(self) -> str:
        return self.spec.group_id

    @property
    def consent_start(self) -> dt.date:
        return self.spec.consent_start

    # ---------------------------------------------------------------- ledger

    def add_decision(self, decision: Decision) -> None:
        self.decisions[(decision.decision_type, decision.decision_idx)] = decision
        if decision.decision_type == "aya_message":
            self.aya_by_mtw[(decision.day, decision.slot)] = decision
        elif decision.decision_type == "cp_message":
            self.cp_by_day[decision.day] = decision

    def record_delivery(self, role: str, when: dt.datetime, delivered: bool) -> None:
        """One delivery step of the dose trace: ``b <- gamma b + d`` (API-Spec §4.1)."""
        previous = self._burden_after[role][-1] if self._burden_after[role] else 0.0
        self._delivery_times[role].append(when)
        self._delivery_flags[role].append(bool(delivered))
        self._burden_after[role].append(GAMMA[role] * previous + (1.0 if delivered else 0.0))

    def burden_at(self, role: str, t: dt.datetime) -> float:
        """Burden for the upcoming decision: every delivery step at or before ``t`` folded in."""
        i = bisect.bisect_right(self._delivery_times[role], t)
        return self._burden_after[role][i - 1] if i else 0.0

    def delivered_between(self, role: str, start: dt.datetime, end: dt.datetime) -> int:
        """Successful deliveries with ``start <= time <= end``."""
        lo = bisect.bisect_left(self._delivery_times[role], start)
        hi = bisect.bisect_right(self._delivery_times[role], end)
        return sum(self._delivery_flags[role][lo:hi])

    def last_delivery_flag(self, role: str, t: dt.datetime) -> bool:
        i = bisect.bisect_right(self._delivery_times[role], t)
        return self._delivery_flags[role][i - 1] if i else False

    def delivered_on(self, role: str, day: dt.date) -> bool:
        return self.delivered_between(role, at(day, dt.time.min), at(day, dt.time.max)) > 0

    def game_on(self, day: dt.date) -> int:
        """The game actually running on ``day`` (0 in the partial week)."""
        week = study_week(self.consent_start, day)
        return int(self.game_by_week.get(week, 0)) if week >= 1 else 0

    # -------------------------------------------------------------- snapshot

    def last_mtw(self, t: dt.datetime) -> tuple[dt.date, str] | None:
        """The most recently elapsed MTW at ``t`` (its end time <= t), within the study window."""
        day = t.date()
        if t >= at(day, T_MTW_END["pm"]):
            mtw = (day, "pm")
        elif t >= at(day, T_MTW_END["am"]):
            mtw = (day, "am")
        else:
            mtw = (day - dt.timedelta(days=1), "pm")
        return mtw if mtw[0] >= self.consent_start else None

    def last_diary_day(self, t: dt.datetime) -> dt.date | None:
        """The day of the most recent evening diary that is final at ``t``."""
        day = t.date() if t >= at(t.date(), T_DIARY) else t.date() - dt.timedelta(days=1)
        return day if day >= self.consent_start else None

    def _diary_submitted(self, role: str, day: dt.date) -> bool:
        if role == "aya":
            mood, physical = self.aya_diary.get(day, (MISS, MISS))
            return mood != MISS or physical != MISS
        return self.cp_diary.get(day, MISS) != MISS

    def engagement(self, role: str, t: dt.datetime, gap_rate: float) -> Any:
        """App-engagement bucket for days d-1..d-3 (API-Spec §4.1); ``"miss"`` before analytics exist or on a gap."""
        day = t.date()
        d1 = day - dt.timedelta(days=1)
        if d1 < self.consent_start:
            return MISS  # first uploads of the study: analytics not computed yet
        if gap_rate > 0 and _rng(self.seed, self.group_id, "analytics_gap", role, t.isoformat()).random() < gap_rate:
            return MISS  # operational analytics gap
        opened = [self.opens[role].get(day - dt.timedelta(days=k), False) for k in (1, 2, 3)]
        if not any(opened):
            return 1
        if not opened[0]:
            return 2
        return 4 if self._diary_submitted(role, d1) else 3

    def aya_missing_rate(self, t: dt.datetime) -> float:
        """|M|/|E| over check-ins (2 MTW adherence + 1 evening diary per day) ending in ``(t - 7d, t]``."""
        lo = t - dt.timedelta(days=7)
        expected = missed = 0
        day = max(self.consent_start, lo.date())
        while day <= t.date():
            for slot in ("am", "pm"):
                if lo < at(day, T_MTW_END[slot]) <= t:
                    expected += 1
                    missed += self.adherence.get((day, slot), MISS) == MISS
            if lo < at(day, T_DIARY) <= t:
                expected += 1
                missed += not self._diary_submitted("aya", day)
            day += dt.timedelta(days=1)
        return missed / expected if expected else 0.0

    def cp_missing_rate(self, t: dt.datetime) -> float:
        """|M|/|D| over the past 7 calendar days (excluding today) inside the study (API-Spec §4.1)."""
        days = [t.date() - dt.timedelta(days=k) for k in range(1, 8)]
        days = [d for d in days if d >= self.consent_start]
        if not days:
            return 0.0
        return sum(self.cp_diary.get(d, MISS) == MISS for d in days) / len(days)

    def diary_summary(self, role: str, t: dt.datetime) -> Any:
        """Mean normalized mood ``(m-1)/4`` over diaries final in ``(t - 7d, t]``; ``"miss"`` if none."""
        lo = t - dt.timedelta(days=7)
        values = []
        day = max(self.consent_start, lo.date())
        while day <= t.date():
            if lo < at(day, T_DIARY) <= t:
                mood = self.aya_diary.get(day, (MISS, MISS))[0] if role == "aya" else self.cp_diary.get(day, MISS)
                if mood != MISS:
                    values.append((mood - 1) / 4)
            day += dt.timedelta(days=1)
        return sum(values) / len(values) if values else MISS

    def relationship(self, t: dt.datetime) -> tuple[Any, Any, bool]:
        """(cached AYA score, cached CP score, latest administration completed by both)."""
        aya = cp = MISS
        completed = False
        for when, aya_score, cp_score in self.surveys:
            if when > t:
                break
            if aya_score != MISS:
                aya = aya_score
            if cp_score != MISS:
                cp = cp_score
            completed = aya_score != MISS and cp_score != MISS
        return aya, cp, completed

    def snapshot(self, t: dt.datetime, kind: str, gap_rate: float = 0.0) -> dict:
        """The full API-Spec §4.1 snapshot the host uploads at ``t`` (``kind`` in :data:`UPLOAD_KINDS`)."""
        day = t.date()
        week = study_week(self.consent_start, day)
        mtw = self.last_mtw(t)
        adherence = self.adherence.get(mtw, MISS) if mtw else MISS
        prompted = False
        if mtw is not None and adherence != MISS:
            decision = self.aya_by_mtw.get(mtw)
            prompted = bool(decision is not None and decision.delivered)
        diary_day = self.last_diary_day(t)
        aya_mood, aya_physical = self.aya_diary.get(diary_day, (MISS, MISS)) if diary_day else (MISS, MISS)
        cp_mood = self.cp_diary.get(diary_day, MISS) if diary_day else MISS

        monday = at(week_monday(day), dt.time.min)
        if kind == CLOSING:  # final counts of the week just ended
            count_start, count_end = monday - dt.timedelta(days=7), monday - dt.timedelta(microseconds=1)
        else:
            count_start, count_end = monday, t

        if kind in (CLOSING, PRE_GAME) or week == 0:
            current_game = MISS
        else:
            game_week = week - 1 if self.stuck_week == week else week
            current_game = self.game_by_week.get(game_week, MISS)
        prior_game = self.game_by_week.get(week - 1, MISS) if week >= 2 else MISS

        rq_aya, rq_cp, survey_completed = self.relationship(t)
        return {
            "day_in_study": day_in_study(self.consent_start, day),
            "week_in_study": week,
            "slot": "am" if t.hour < NOON_HOUR else "pm",
            "aya_diary_mood": aya_mood,
            "aya_diary_physical": aya_physical,
            "aya_app_engagement": self.engagement("aya", t, gap_rate),
            "aya_app_burden": self.burden_at("aya", t),
            "aya_missing_rate_7d": self.aya_missing_rate(t),
            "previous_med_adherence": adherence,
            "prompted_by_message": prompted,
            "aya_messages_delivered_week": self.delivered_between("aya", count_start, count_end),
            "cp_diary_mood": cp_mood,
            "cp_app_engagement": self.engagement("cp", t, gap_rate),
            "cp_app_burden": self.burden_at("cp", t),
            "cp_missing_rate_7d": self.cp_missing_rate(t),
            "daily_diary_completed": cp_mood != MISS,
            "cp_prompted_by_message": self.last_delivery_flag("cp", t),
            "cp_messages_delivered_week": self.delivered_between("cp", count_start, count_end),
            "relationship_quality_aya": rq_aya,
            "relationship_quality_cp": rq_cp,
            "current_game_on": current_game,
            "prior_game_action": prior_game,
            "aya_diary_summary": self.diary_summary("aya", t),
            "cp_diary_summary": self.diary_summary("cp", t),
            "weekly_survey_completed": survey_completed,
            "weekly_relationship_score": 0.5 * (rq_aya + rq_cp) if survey_completed else 0.0,
        }

    # ------------------------------------------------------------ outcomes

    def resolve_mtw(self, day: dt.date, slot: str, model: OutcomeModel) -> None:
        """Adherence report of one MTW; at the evening MTW also the AYA and CP diaries (API-Spec §3)."""
        lat = self.latent
        aya = self.aya_by_mtw.get((day, slot))
        delivered = 1.0 if aya is not None and aya.delivered else 0.0
        burden = aya.burden_before if aya is not None and aya.burden_before is not None else self.burden_at("aya", at(day, T_MTW_END[slot]))
        game = self.game_on(day)
        r = _rng(self.seed, self.group_id, "mtw", day.isoformat(), slot)
        p_respond = _clip(lat.adh_response + model.msg_response * delivered - 0.004 * burden, 0.3, 0.98)
        p_adhere = _clip(
            lat.adh_base + model.msg_adherence * delivered - model.burden_adherence * burden + model.game_adherence * game,
            0.05,
            0.95,
        )
        responded = r.random() < p_respond
        adhered = r.random() < p_adhere
        self.adherence[(day, slot)] = int(adhered) if responded else MISS
        if slot != "pm":
            return

        # AYA evening diary (mood + physical, raw 1-5 items).
        any_aya = 1.0 if self.delivered_on("aya", day) else 0.0
        r = _rng(self.seed, self.group_id, "aya_diary", day.isoformat())
        p_submit = _clip(lat.aya_diary_rate + model.msg_aya_diary * delivered - model.burden_aya_diary * burden, 0.1, 0.97)
        if r.random() < p_submit:
            mood = int(_clip(round(lat.aya_mood + model.msg_aya_mood * any_aya + model.game_aya_mood * game + r.gauss(0, model.mood_sd)), 1, 5))
            physical = int(_clip(round(lat.aya_physical + r.gauss(0, model.mood_sd)), 1, 5))
            u = r.random()
            if u < model.mood_only_physical:
                mood = MISS
            elif u < model.mood_only_physical + model.physical_skipped:
                physical = MISS
            self.aya_diary[day] = (mood, physical)
        else:
            self.aya_diary[day] = (MISS, MISS)

        # CP daily diary (mood only).
        cp = self.cp_by_day.get(day)
        cp_delivered = 1.0 if cp is not None and cp.delivered else 0.0
        cp_burden = cp.burden_before if cp is not None and cp.burden_before is not None else self.burden_at("cp", at(day, T_DIARY))
        r = _rng(self.seed, self.group_id, "cp_diary", day.isoformat())
        p_submit = _clip(lat.cp_diary_rate + model.msg_cp_diary * cp_delivered - model.burden_cp_diary * cp_burden + 0.03 * game, 0.1, 0.97)
        if r.random() < p_submit:
            self.cp_diary[day] = int(
                _clip(round(lat.cp_mood + model.msg_cp_mood * cp_delivered + model.game_cp_mood * game + r.gauss(0, model.mood_sd)), 1, 5)
            )
        else:
            self.cp_diary[day] = MISS

    def resolve_app_opens(self, day: dt.date, model: OutcomeModel) -> None:
        """End-of-day app analytics: a submitted diary implies an app open."""
        r = _rng(self.seed, self.group_id, "opens", day.isoformat())
        for role, base in (("aya", self.latent.aya_open), ("cp", self.latent.cp_open)):
            p_open = _clip(base + model.msg_open * self.delivered_on(role, day), 0.05, 0.98)
            self.opens[role][day] = self._diary_submitted(role, day) or r.random() < p_open

    def resolve_survey(self, when: dt.datetime, model: OutcomeModel) -> None:
        """Sunday-evening relationship survey: each respondent's score is the mean of raw 1-5 items."""
        game = self.game_on(when.date())
        r = _rng(self.seed, self.group_id, "survey", when.date().isoformat())

        def score(rate: float) -> Any:
            if r.random() >= rate:
                return MISS
            items = [
                int(_clip(round(self.latent.rel_level + model.game_relq * game + r.gauss(0, model.relq_sd)), 1, 5))
                for _ in range(model.survey_items)
            ]
            return sum(items) / len(items)

        aya_score = score(self.latent.aya_survey_rate)
        cp_score = score(self.latent.cp_survey_rate)
        self.surveys.append((when, aya_score, cp_score))


# ------------------------------------------------------------- validation


def field_encoding_issues(data: Any) -> list[str]:
    """Field-level API-Spec §4.1 problems of a snapshot: key set and each field's encoding."""
    if not isinstance(data, dict):
        return ["data must be a dictionary"]
    issues = [f"unknown field {name!r}" for name in data if name not in SNAPSHOT_FIELDS]
    issues += [f"missing field {name!r}" for name in SNAPSHOT_FIELDS if name not in data]
    for name, encoding in SNAPSHOT_FIELDS.items():
        if name in data and not _encoding_ok(encoding, data[name]):
            issues.append(f"{name}={data[name]!r} is not {encoding}")
    return issues


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _encoding_ok(encoding: str, value: Any) -> bool:
    if encoding == "slot":
        return value in ("am", "pm")
    if encoding == "bool":
        return isinstance(value, bool)
    if encoding.endswith("_or_miss") and value == MISS:
        return True
    base = encoding.removesuffix("_or_miss")
    if base == "positive_int":
        return _is_int(value) and value >= 1
    if base == "nonneg_int":
        return _is_int(value) and value >= 0
    if base == "nonneg_float":
        return _is_number(value) and value >= 0
    if base == "unit_interval":
        return _is_number(value) and 0 <= value <= 1
    if base == "binary":
        return _is_int(value) and value in (0, 1)
    if base == "item_1_5":
        return _is_int(value) and 1 <= value <= 5
    if base == "score_1_5":
        return _is_number(value) and 1 <= value <= 5
    if base == "engagement":
        return _is_int(value) and 1 <= value <= 4
    raise ValueError(f"unknown encoding {encoding!r}")


def _is_multiple(value: float, denominator: int) -> bool:
    scaled = value * denominator
    return abs(scaled - round(scaled)) <= 1e-7


def expected_checkins(day_index: int, kind: str) -> int:
    """
    |E| of ``aya_missing_rate_7d`` for an upload on study day ``day_index``
    (API-Spec §4.1): an AM-side upload (before 10:00) sees 3 check-ins per
    elapsed study day (at most 7 days); a PM upload also sees that
    morning's MTW.
    """
    elapsed = min(7, day_index - 1)
    if kind == PRE_AYA_PM:
        return min(7, day_index) + 2 * elapsed
    return 3 * elapsed


def check_snapshot_rules(
    data: dict, *, consent_start: dt.date, t: dt.datetime, kind: str, trial_weeks: int = TRIAL_WEEKS
) -> list[str]:
    """
    Static API-Spec §4.1 rules for one snapshot (no history needed). Returns
    the violated rules as messages; ``[]`` means valid.
    """
    issues = field_encoding_issues(data)
    if issues:
        return issues
    day = t.date()
    week = study_week(consent_start, day)
    n_day = day_in_study(consent_start, day)

    # Bookkeeping (§4.1 "Bookkeeping").
    if data["day_in_study"] != n_day:
        issues.append(f"day_in_study {data['day_in_study']} != {n_day}")
    if data["week_in_study"] != week:
        issues.append(f"week_in_study {data['week_in_study']} != {week} (Monday-week formula)")
    slot = "am" if t.hour < NOON_HOUR else "pm"
    if data["slot"] != slot:
        issues.append(f"slot {data['slot']!r} does not match upload time {t.time()}")

    # Upload schedule (§2.3, §3).
    if kind in (CLOSING, PRE_GAME) and day.weekday() != 0:
        issues.append(f"{kind} upload on a {day.strftime('%A')}")
    if kind == CLOSING and not (dt.time(0, 0) <= t.time() <= T_CLOSING_LATEST):
        issues.append(f"week-closing upload at {t.time()} outside 00:00-02:45")
    if kind == PRE_GAME and not 1 <= week <= trial_weeks:
        issues.append(f"pre-dyad_game upload in study week {week}")
    if (kind in AM_KINDS) != (data["slot"] == "am"):
        issues.append(f"{kind} upload with slot {data['slot']!r}")

    # Game fields (§4.1 current_game_on / prior_game_action).
    if kind in (CLOSING, PRE_GAME) or week == 0:
        if data["current_game_on"] != MISS:
            issues.append(f"current_game_on must be 'miss' at a {kind} upload / partial week")
    elif data["current_game_on"] == MISS:
        issues.append(f"current_game_on is 'miss' after the week-{week} dyad_game decision")
    if week <= 1 and data["prior_game_action"] != MISS:
        issues.append(f"prior_game_action must be 'miss' in study week {week}")
    if week >= 2 and data["prior_game_action"] == MISS:
        issues.append(f"prior_game_action is 'miss' in study week {week}")

    # Delivered counts reset after the week-closing upload (§4.1).
    if kind == PRE_GAME and (data["aya_messages_delivered_week"] or data["cp_messages_delivered_week"]):
        issues.append("delivered-week counts must be 0 at the Monday pre-dyad_game upload")
    if data["aya_messages_delivered_week"] > 14 or data["cp_messages_delivered_week"] > 7:
        issues.append("delivered-week count above the scheduled opportunities")

    # Burden bounds of the dose trace.
    for role in ("aya", "cp"):
        if data[f"{role}_app_burden"] > BURDEN_BOUND[role] + FLOAT_TOL:
            issues.append(f"{role}_app_burden {data[f'{role}_app_burden']} above 1/(1-gamma)")

    # Weekly survey (§4.1 weekly_survey_completed / weekly_relationship_score).
    rq_aya, rq_cp = data["relationship_quality_aya"], data["relationship_quality_cp"]
    if data["weekly_survey_completed"]:
        if MISS in (rq_aya, rq_cp):
            issues.append("weekly_survey_completed is true with a 'miss' relationship score")
        elif abs(data["weekly_relationship_score"] - 0.5 * (rq_aya + rq_cp)) > FLOAT_TOL:
            issues.append("weekly_relationship_score != (aya + cp) / 2")
    elif data["weekly_relationship_score"] != 0.0:
        issues.append("weekly_relationship_score must be 0.0 when the survey is not completed")

    # CP diary / adherence linkage.
    if data["daily_diary_completed"] != (data["cp_diary_mood"] != MISS):
        issues.append("daily_diary_completed disagrees with cp_diary_mood")
    if data["prompted_by_message"] and data["previous_med_adherence"] == MISS:
        issues.append("prompted_by_message is true while previous_med_adherence is 'miss'")

    # Missing-rate denominators and diary summaries (§4.1 formulas).
    n_checkins = expected_checkins(n_day, kind)
    if n_checkins == 0:
        if data["aya_missing_rate_7d"] != 0.0:
            issues.append("aya_missing_rate_7d must be 0.0 before any check-in")
    elif not _is_multiple(data["aya_missing_rate_7d"], n_checkins):
        issues.append(f"aya_missing_rate_7d {data['aya_missing_rate_7d']} is not k/{n_checkins}")
    n_days = min(7, n_day - 1)
    if n_days == 0:
        if data["cp_missing_rate_7d"] != 0.0:
            issues.append("cp_missing_rate_7d must be 0.0 on day 1")
    elif not _is_multiple(data["cp_missing_rate_7d"], n_days):
        issues.append(f"cp_missing_rate_7d {data['cp_missing_rate_7d']} is not k/{n_days}")
    for name in ("aya_diary_summary", "cp_diary_summary"):
        value = data[name]
        if value == MISS:
            continue
        if n_days == 0 or not any(_is_multiple(4 * value, k) for k in range(1, n_days + 1)):
            issues.append(f"{name} {value} is not a mean of (m-1)/4 over at most {n_days} diaries")

    # Cold start (§3 "Cold-start"): AM uploads on day 1, before any decision outcome exists.
    if n_day == 1 and kind in AM_KINDS:
        cold = {
            "previous_med_adherence": MISS,
            "aya_diary_mood": MISS,
            "aya_diary_physical": MISS,
            "cp_diary_mood": MISS,
            "prior_game_action": MISS,
            "aya_diary_summary": MISS,
            "cp_diary_summary": MISS,
            "relationship_quality_aya": MISS,
            "relationship_quality_cp": MISS,
            "aya_app_burden": 0.0,
            "cp_app_burden": 0.0,
            "aya_missing_rate_7d": 0.0,
            "cp_missing_rate_7d": 0.0,
            "weekly_relationship_score": 0.0,
            "prompted_by_message": False,
            "cp_prompted_by_message": False,
        }
        for name, value in cold.items():
            if data[name] != value:
                issues.append(f"cold start: {name} must be {value!r}, got {data[name]!r}")
    return issues


class SnapshotAuditor:
    """
    Sequential API-Spec §4.1 checks, from the host's own ledger (deliveries,
    game actions, surveys, accepted uploads) kept separately from the
    snapshot builder:

    - burden recurrence ``b <- gamma b + d`` over every delivery step;
    - delivered-week counts (final counts at the week-closing upload, reset after it);
    - ``current_game_on`` / ``prior_game_action`` against the game ledger
      (a declared stuck week expects last week's value);
    - ``prompted_by_message`` / ``cp_prompted_by_message`` against the delivery log;
    - relationship scores change only at a survey;
    - every ``/action`` has exactly one accepted upload 2-5 minutes before it.
    """

    def __init__(self, trial_weeks: int = TRIAL_WEEKS):
        self.trial_weeks = trial_weeks
        self.consent: dict[str, dt.date] = {}
        self.burden: dict[tuple[str, str], float] = defaultdict(float)
        self.week_counts: dict[tuple[str, str, dt.date], int] = defaultdict(int)
        self.last_cp_flag: dict[str, bool] = defaultdict(bool)
        self.aya_flags: dict[tuple[str, dt.date, str], bool] = {}
        self.games: dict[tuple[str, int], int] = {}
        self.stuck: dict[str, int] = {}
        self.survey_times: dict[str, list[dt.datetime]] = defaultdict(list)
        self.last_snapshot: dict[str, tuple[dt.datetime, dict]] = {}
        self.uploads: dict[str, list[dt.datetime]] = defaultdict(list)
        self.last_action: dict[str, dt.datetime] = {}

    def register(self, group_id: str, consent_start: dt.date) -> None:
        self.consent[group_id] = consent_start

    def note_delivery(self, group_id: str, role: str, when: dt.datetime, delivered: bool, mtw: tuple | None) -> None:
        self.burden[(group_id, role)] = GAMMA[role] * self.burden[(group_id, role)] + (1.0 if delivered else 0.0)
        if delivered:
            self.week_counts[(group_id, role, week_monday(when.date()))] += 1
        if role == "cp":
            self.last_cp_flag[group_id] = bool(delivered)
        if role == "aya" and mtw is not None:
            self.aya_flags[(group_id, mtw[0], mtw[1])] = bool(delivered)

    def note_game(self, group_id: str, week: int, action: int) -> None:
        self.games[(group_id, week)] = int(action)

    def note_stuck(self, group_id: str, week: int) -> None:
        self.stuck[group_id] = week

    def note_survey(self, group_id: str, when: dt.datetime) -> None:
        self.survey_times[group_id].append(when)

    def note_upload_accepted(self, group_id: str, when: dt.datetime, data: dict) -> None:
        self.uploads[group_id].append(when)
        self.last_snapshot[group_id] = (when, data)

    def check(self, group_id: str, t: dt.datetime, kind: str, data: dict) -> list[str]:
        consent = self.consent[group_id]
        issues = check_snapshot_rules(data, consent_start=consent, t=t, kind=kind, trial_weeks=self.trial_weeks)
        if issues:
            return issues
        week = study_week(consent, t.date())
        for role in ("aya", "cp"):
            expected = self.burden[(group_id, role)]
            if abs(data[f"{role}_app_burden"] - expected) > FLOAT_TOL:
                issues.append(f"{role}_app_burden {data[f'{role}_app_burden']} != recurrence value {expected}")
            monday = week_monday(t.date())
            if kind == CLOSING:
                monday -= dt.timedelta(days=7)
            count = self.week_counts[(group_id, role, monday)]
            if data[f"{role}_messages_delivered_week"] != count:
                issues.append(f"{role}_messages_delivered_week {data[f'{role}_messages_delivered_week']} != ledger {count}")
        if data["cp_prompted_by_message"] != self.last_cp_flag[group_id]:
            issues.append("cp_prompted_by_message disagrees with the last CP delivery")
        if data["previous_med_adherence"] != MISS:
            day = t.date()
            if t >= at(day, T_MTW_END["pm"]):
                mtw = (day, "pm")
            elif t >= at(day, T_MTW_END["am"]):
                mtw = (day, "am")
            else:
                mtw = (day - dt.timedelta(days=1), "pm")
            if data["prompted_by_message"] != self.aya_flags.get((group_id, *mtw), False):
                issues.append(f"prompted_by_message disagrees with the delivery log for MTW {mtw}")
        if kind not in (CLOSING, PRE_GAME) and week >= 1:
            game_week = week - 1 if self.stuck.get(group_id) == week else week
            expected_game = self.games.get((group_id, game_week))
            if data["current_game_on"] != expected_game:
                issues.append(f"current_game_on {data['current_game_on']!r} != week-{game_week} game {expected_game!r}")
        if week >= 2 and data["prior_game_action"] != self.games.get((group_id, week - 1)):
            issues.append("prior_game_action != last week's dyad_game action")
        previous = self.last_snapshot.get(group_id)
        if previous is not None:
            prev_t, prev = previous
            surveyed = any(prev_t < s <= t for s in self.survey_times[group_id])
            for name in ("relationship_quality_aya", "relationship_quality_cp", "weekly_survey_completed"):
                if not surveyed and data[name] != prev[name]:
                    issues.append(f"{name} changed without a survey in between")
        return issues

    def check_action(self, group_id: str, t: dt.datetime) -> list[str]:
        """Exactly one accepted upload in the 2-5 minutes before ``t`` and none after it (API-Spec §2.3)."""
        since = self.last_action.get(group_id, dt.datetime.min)
        recent = [u for u in self.uploads[group_id] if max(since, t - dt.timedelta(minutes=5)) < u <= t]
        self.last_action[group_id] = t
        if len(recent) != 1:
            return [f"/action at {t} has {len(recent)} accepted uploads in the 5 minutes before it"]
        lead = t - recent[0]
        if not dt.timedelta(minutes=2) <= lead < dt.timedelta(minutes=5):
            return [f"/action at {t}: preceding upload is {lead} earlier (want 2-5 minutes)"]
        return []


# ------------------------------------------------------------ plan / faults


@dataclass
class TrialPlan:
    """
    What to simulate. ``enrollment`` lists consent-start dates (one per
    dyad); when ``None``, dyad ``i`` consents ``7 i + (i mod 7)`` days after
    the first Monday on or after ``start_date`` (one dyad a week, cycling the
    weekday so partial weeks of every length occur). The horizon is
    ``num_weeks`` Monday-Sunday calendar weeks from the Monday of the first
    consent, plus the following Monday's week-closing uploads, ``/update``
    and 05:50 alarm pull (no decisions that Monday).
    """

    num_dyads: int = 6
    num_weeks: int = 16
    start_date: dt.date = dt.date(2026, 11, 2)
    enrollment: tuple[dt.date, ...] | None = None
    seed: int = 42
    trial_weeks: int = TRIAL_WEEKS
    group_prefix: str = ""
    delivery_success: float = 1.0
    analytics_gap_rate: float = 0.02
    pull_alarms: bool = True
    tz_aware_timestamps: bool = False
    strict: bool = True
    record_payloads: bool = False
    outcome_model: OutcomeModel = field(default_factory=OutcomeModel)

    def consent_starts(self) -> list[dt.date]:
        if self.enrollment is not None:
            return list(self.enrollment)
        first = week1_start(self.start_date)
        return [first + dt.timedelta(days=7 * i + i % 7) for i in range(self.num_dyads)]

    @property
    def start_monday(self) -> dt.date:
        return week_monday(min(self.consent_starts()))

    @property
    def horizon_last_day(self) -> dt.date:
        return self.start_monday + dt.timedelta(days=7 * self.num_weeks - 1)

    @property
    def final_monday(self) -> dt.date:
        return self.start_monday + dt.timedelta(days=7 * self.num_weeks)

    @property
    def end_time(self) -> dt.datetime:
        return at(self.final_monday, T_GAME)

    def calendar_week(self, day: dt.date) -> int:
        return (week_monday(day) - self.start_monday).days // 7

    def dyad_specs(self) -> list[DyadSpec]:
        specs = []
        for i, start in enumerate(self.consent_starts()):
            if start > self.horizon_last_day:
                continue
            n = i + 1
            specs.append(
                DyadSpec(
                    index=i,
                    group_id=f"{self.group_prefix}dyad_{n:03d}",
                    member_list=[f"{self.group_prefix}cp_{n:03d}", f"{self.group_prefix}aya_{n:03d}"],
                    consent_start=start,
                    consent_end=consent_end_for(start, self.trial_weeks),
                )
            )
        return specs

    def to_json(self) -> dict:
        return {
            "num_dyads": len(self.consent_starts()),
            "num_weeks": self.num_weeks,
            "start_monday": self.start_monday.isoformat(),
            "final_monday": self.final_monday.isoformat(),
            "consent_starts": [d.isoformat() for d in self.consent_starts()],
            "seed": self.seed,
            "trial_weeks": self.trial_weeks,
            "delivery_success": self.delivery_success,
            "analytics_gap_rate": self.analytics_gap_rate,
            "tz_aware_timestamps": self.tz_aware_timestamps,
        }


@dataclass
class HostFaults:
    """
    Host-side faults (Monitoring-Spec §11.3). Decision keys are
    ``(group_id, decision_type, decision_idx)``; Mondays are calendar dates.

    - ``skip_actions``: the host's scheduler skips the upload + ``/action``
      pair and flips its own coin (F-A1); the decision still happens host-side.
    - ``withhold_closing``: ``(group_id, monday)`` week-closing upload not sent.
    - ``bad_diary_uploads``: the upload before this decision is first sent
      with ``aya_diary_mood = 7`` (expect 400 ``UD-400-RANGE``), then the
      corrected snapshot.
    - ``skip_updates``: no ``/update`` that Monday; ``delay_updates``: send it
      at the given time instead of 03:00; ``resend_updates``: an extra
      ``/update`` at the given time (e.g. after a killed update).
    - ``stuck_game``: ``group_id -> week``: in the first study week ``>= week``
      whose game action differs from last week's, the host keeps reporting
      last week's ``current_game_on``.
    - ``allowed_statuses``: ``(endpoint, status)`` accepted anywhere.
    """

    skip_actions: set[tuple[str, str, int]] = field(default_factory=set)
    withhold_closing: set[tuple[str, dt.date]] = field(default_factory=set)
    bad_diary_uploads: set[tuple[str, str, int]] = field(default_factory=set)
    skip_updates: set[dt.date] = field(default_factory=set)
    delay_updates: dict[dt.date, dt.time] = field(default_factory=dict)
    resend_updates: dict[dt.date, dt.time] = field(default_factory=dict)
    stuck_game: dict[str, int] = field(default_factory=dict)
    allowed_statuses: set[tuple[str, int]] = field(default_factory=set)

    def to_json(self) -> dict:
        return {
            "skip_actions": sorted(list(k) for k in self.skip_actions),
            "withhold_closing": sorted([g, m.isoformat()] for g, m in self.withhold_closing),
            "bad_diary_uploads": sorted(list(k) for k in self.bad_diary_uploads),
            "skip_updates": sorted(m.isoformat() for m in self.skip_updates),
            "delay_updates": {m.isoformat(): t.isoformat() for m, t in self.delay_updates.items()},
            "resend_updates": {m.isoformat(): t.isoformat() for m, t in self.resend_updates.items()},
            "stuck_game": dict(self.stuck_game),
            "allowed_statuses": sorted(list(k) for k in self.allowed_statuses),
        }


@dataclass
class SimulatorHooks:
    """
    Harness callbacks. ``on_tick(now)`` runs on a fixed logical grid
    (``tick_every``, aligned to midnight) for every grid point the logical
    clock passes; ``wait_for_update(rid, when)`` blocks until an accepted
    ``/update`` has finished (a threaded server); ``after_request(record,
    simulator)`` sees every call; ``timed`` runs ``callable(now)`` at fixed
    logical times.
    """

    tick_every: dt.timedelta | None = None
    on_tick: Callable[[dt.datetime], None] | None = None
    wait_for_update: Callable[[str, dt.datetime], None] | None = None
    after_request: Callable[["RequestRecord", "HostSimulator"], None] | None = None
    timed: list[tuple[dt.datetime, Callable[[dt.datetime], None]]] = field(default_factory=list)


@dataclass
class RequestRecord:
    """One call the host made and what came back."""

    seq: int
    logical_time: dt.datetime
    method: str
    endpoint: str
    status: int | None
    title: str | None
    message: str | None
    rid: str | None
    expected: bool
    group_id: str | None = None
    decision_type: str | None = None
    decision_idx: int | None = None
    kind: str | None = None
    note: str | None = None
    request: dict | None = None
    response: dict | None = None

    def to_json(self) -> dict:
        out = {
            "seq": self.seq,
            "logical_time": self.logical_time.isoformat(),
            "method": self.method,
            "endpoint": self.endpoint,
            "status": self.status,
            "title": self.title,
            "message": self.message,
            "rid": self.rid,
            "expected": self.expected,
        }
        for name in ("group_id", "decision_type", "decision_idx", "kind", "note", "request", "response"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        return out


@dataclass(order=True)
class _Event:
    time: dt.datetime
    priority: int
    seq: int
    kind: str = field(compare=False)
    group_id: str | None = field(default=None, compare=False)
    data: Any = field(default=None, compare=False)


# Same-time ordering: world first, then host calls in API-Spec §3 order.
_PRIORITY = {
    "mtw": 0,
    "end_of_day": 0,
    "survey": 0,
    "hook": 1,
    "register": 2,
    "closing": 3,
    "update": 4,
    "pull": 5,
    "bad_upload": 6,
    "upload": 7,
    "action": 8,
    "host_decision": 8,
    "delivery": 9,
}


# --------------------------------------------------------------- simulator


class HostSimulator:
    """
    The simulated host. ``run()`` plays the whole calendar against the
    transport and returns a :class:`SimulationResult`.
    """

    def __init__(
        self,
        transport: Transport,
        plan: TrialPlan | None = None,
        faults: HostFaults | None = None,
        hooks: SimulatorHooks | None = None,
        log: Callable[[str], None] | None = None,
    ):
        self.transport = transport
        self.plan = plan or TrialPlan()
        self.faults = faults or HostFaults()
        self.hooks = hooks or SimulatorHooks()
        self.log = log
        self.auditor = SnapshotAuditor(self.plan.trial_weeks)
        self.dyads: dict[str, DyadHost] = {}
        for spec in self.plan.dyad_specs():
            self.dyads[spec.group_id] = DyadHost(
                spec, DyadLatent.draw(self.plan.seed, spec.group_id), self.plan.seed, self.plan.trial_weeks
            )
            self.auditor.register(spec.group_id, spec.consent_start)
        self.records: list[RequestRecord] = []
        self.unexpected: list[RequestRecord] = []
        self.violations: list[dict] = []
        self.alarms: dict[str, dict] = {}
        self.update_rids: list[tuple[dt.datetime, str | None, int | None]] = []
        self.now: dt.datetime | None = None
        self._seq = 0
        self._next_tick: dt.datetime | None = None
        self._events = self._build_schedule()

    # ------------------------------------------------------------ schedule

    def _build_schedule(self) -> list[_Event]:
        plan, faults = self.plan, self.faults
        events: list[_Event] = []
        seq = iter(range(10**9))

        def add(time: dt.datetime, kind: str, group_id: str | None = None, data: Any = None) -> None:
            events.append(_Event(time, _PRIORITY[kind], next(seq), kind, group_id, data))

        first_registration = min((d.spec.registration_time for d in self.dyads.values()), default=None)
        for dyad in self.dyads.values():
            spec = dyad.spec
            add(spec.registration_time, "register", spec.group_id)
            last_day = min(spec.consent_end, plan.horizon_last_day)
            aya_idx = cp_idx = 0
            day = spec.consent_start
            while day <= last_day:
                week = study_week(spec.consent_start, day)
                pairs = []
                if day.weekday() == 0 and 1 <= week <= plan.trial_weeks:
                    pairs.append(("dyad_game", week, "am", T_GAME, PRE_GAME))
                aya_idx += 1
                pairs.append(("aya_message", aya_idx, "am", T_AYA_AM, PRE_AYA_AM))
                cp_idx += 1
                pairs.append(("cp_message", cp_idx, "am", T_CP, PRE_CP))
                aya_idx += 1
                pairs.append(("aya_message", aya_idx, "pm", T_AYA_PM, PRE_AYA_PM))
                for decision_type, idx, slot, clock, kind in pairs:
                    when = at(day, clock)
                    lead = _rng(plan.seed, spec.group_id, "lead", decision_type, idx).randint(*UPLOAD_LEAD_SECONDS)
                    decision = Decision(
                        group_id=spec.group_id,
                        decision_type=decision_type,
                        decision_idx=idx,
                        slot=slot,
                        day=day,
                        time=when,
                        upload_time=when - dt.timedelta(seconds=lead),
                        kind=kind,
                        study_week=week,
                        calendar_week=plan.calendar_week(day),
                    )
                    dyad.add_decision(decision)
                    if decision.key in faults.skip_actions:
                        add(when, "host_decision", spec.group_id, decision)
                    else:
                        if decision.key in faults.bad_diary_uploads:
                            add(decision.upload_time - BAD_UPLOAD_LEAD, "bad_upload", spec.group_id, decision)
                        add(decision.upload_time, "upload", spec.group_id, decision)
                        add(when, "action", spec.group_id, decision)
                    if decision_type != "dyad_game":
                        add(at(day, T_DELIVERY[(decision_type, slot)]), "delivery", spec.group_id, decision)
                add(at(day, T_MTW_END["am"]), "mtw", spec.group_id, (day, "am"))
                add(at(day, T_MTW_END["pm"]), "mtw", spec.group_id, (day, "pm"))
                if day.weekday() == 6:
                    add(at(day, T_SURVEY), "survey", spec.group_id, day)
                add(at(day, T_END_OF_DAY), "end_of_day", spec.group_id, day)
                day += dt.timedelta(days=1)

        # Mondays: week-closing uploads, /update, then the day's pulls.
        monday = plan.start_monday
        while monday <= plan.final_monday:
            rank = 0
            for dyad in self.dyads.values():
                spec = dyad.spec
                last_day = min(spec.consent_end, plan.horizon_last_day)
                had_decision = spec.consent_start <= monday - dt.timedelta(days=1) and last_day >= monday - dt.timedelta(days=7)
                if not had_decision:
                    continue
                when = at(monday, T_CLOSING) + rank * CLOSING_SPACING
                rank += 1
                if (spec.group_id, monday) in faults.withhold_closing:
                    continue
                add(when, "closing", spec.group_id, monday)
            update_time = at(monday, faults.delay_updates.get(monday, T_UPDATE))
            if first_registration is not None and first_registration < at(monday, T_UPDATE):
                if monday not in faults.skip_updates:
                    add(update_time, "update", None, monday)
                if monday in faults.resend_updates:
                    add(at(monday, faults.resend_updates[monday]), "update", None, monday)
            monday += dt.timedelta(days=7)

        if plan.pull_alarms and first_registration is not None:
            day = first_registration.date()
            while day <= plan.final_monday:
                for clock in PULL_TIMES:
                    when = at(day, clock)
                    if first_registration < when <= plan.end_time:
                        add(when, "pull", None, None)
                day += dt.timedelta(days=1)

        for when, callback in self.hooks.timed:
            add(when, "hook", None, callback)
        events.sort()
        return events

    def expected_decisions(self) -> dict[tuple[str, str], int]:
        """Decisions the calendar schedules per (dyad, agent) within the horizon (API-Spec §3)."""
        counts: dict[tuple[str, str], int] = Counter()
        for dyad in self.dyads.values():
            for decision in dyad.decisions.values():
                counts[(dyad.group_id, decision.decision_type)] += 1
        return dict(counts)

    @property
    def events(self) -> list[_Event]:
        return list(self._events)

    # ----------------------------------------------------------------- run

    def run(self) -> "SimulationResult":
        for event in self._events:
            self._advance(event.time)
            self.now = event.time
            getattr(self, f"_on_{event.kind}")(event)
        self._advance(self.plan.end_time, inclusive=True)
        return SimulationResult(self)

    def _advance(self, until: dt.datetime, inclusive: bool = False) -> None:
        step, on_tick = self.hooks.tick_every, self.hooks.on_tick
        if step is None or on_tick is None:
            return
        if self._next_tick is None:
            midnight = at(until.date(), dt.time.min)
            self._next_tick = midnight + step * math.ceil((until - midnight) / step)
            if not inclusive and self._next_tick == until:
                return
        while self._next_tick < until or (inclusive and self._next_tick <= until):
            on_tick(self._next_tick)
            self._next_tick += step

    # ------------------------------------------------------------ handlers

    def _on_register(self, event: _Event) -> None:
        spec = self.dyads[event.group_id].spec
        self._call("POST", "register_group", spec.registration_payload(), group_id=spec.group_id)

    def _on_upload(self, event: _Event) -> None:
        decision: Decision = event.data
        self._send_snapshot(self.dyads[event.group_id], event.time, decision.kind, decision)

    def _on_bad_upload(self, event: _Event) -> None:
        decision: Decision = event.data
        dyad = self.dyads[event.group_id]
        data = dyad.snapshot(event.time, decision.kind, self.plan.analytics_gap_rate)
        data["aya_diary_mood"] = 7  # injected: out of the 1-5 range (UD-400-RANGE, F-U1)
        self._call(
            "POST",
            "upload_data",
            {"group_id": dyad.group_id, "timestamp": self._ts(event.time), "data": data},
            expect={400},
            group_id=dyad.group_id,
            kind=decision.kind,
            note="injected diary item 7",
        )

    def _on_closing(self, event: _Event) -> None:
        self._send_snapshot(self.dyads[event.group_id], event.time, CLOSING, None)

    def _send_snapshot(self, dyad: DyadHost, t: dt.datetime, kind: str, decision: Decision | None) -> None:
        data = dyad.snapshot(t, kind, self.plan.analytics_gap_rate)
        issues = self.auditor.check(dyad.group_id, t, kind, data)
        if issues:
            self._violation("snapshot", dyad.group_id, t, issues, kind=kind)
        record = self._call(
            "POST",
            "upload_data",
            {"group_id": dyad.group_id, "timestamp": self._ts(t), "data": data},
            group_id=dyad.group_id,
            kind=kind,
            decision=decision,
        )
        if record.status == 201:
            self.auditor.note_upload_accepted(dyad.group_id, t, data)

    def _on_action(self, event: _Event) -> None:
        decision: Decision = event.data
        dyad = self.dyads[event.group_id]
        issues = self.auditor.check_action(dyad.group_id, event.time)
        if issues:
            self._violation("upload_before_action", dyad.group_id, event.time, issues, decision=decision)
        payload = {
            "group_id": dyad.group_id,
            "timestamp": self._ts(event.time),
            "decision_idx": decision.decision_idx,
            "decision_type": decision.decision_type,
        }
        record = self._call("POST", "action", payload, group_id=dyad.group_id, decision=decision)
        body = record.response or {}
        decision.status = record.status
        if record.status in (200, 201) and body.get("action") in (0, 1):
            decision.source = "api"
            decision.action = int(body["action"])
            decision.action_prob = body.get("action_prob")
            decision.warmup = body.get("warmup")
            decision.rid = body.get("rid")
            decision.state_len = len(body["state"]) if isinstance(body.get("state"), list) else None
            decision.model_param_len = len(body["model_param"]) if isinstance(body.get("model_param"), list) else None
            self._check_action_body(decision, body)
        else:
            self._host_coin(decision, "host_fallback")  # F-A1
        self._after_decision(dyad, decision)

    def _on_host_decision(self, event: _Event) -> None:
        decision: Decision = event.data
        self._host_coin(decision, "host_skip")
        self._after_decision(self.dyads[event.group_id], decision)

    def _host_coin(self, decision: Decision, source: str) -> None:
        r = _rng(self.plan.seed, decision.group_id, "host_coin", decision.decision_type, decision.decision_idx)
        decision.source = source
        decision.action = int(r.random() < 0.5)
        decision.action_prob = 0.5
        decision.warmup = None

    def _after_decision(self, dyad: DyadHost, decision: Decision) -> None:
        if decision.decision_type == "dyad_game":
            week = decision.study_week
            dyad.game_by_week[week] = decision.action
            self.auditor.note_game(dyad.group_id, week, decision.action)
            from_week = self.faults.stuck_game.get(dyad.group_id)
            if (
                from_week is not None
                and dyad.stuck_week is None
                and week >= max(2, from_week)
                and dyad.game_by_week.get(week - 1) not in (None, decision.action)
            ):
                dyad.stuck_week = week
                self.auditor.note_stuck(dyad.group_id, week)
        else:
            decision.burden_before = dyad.burden_at(decision.role, decision.time)

    def _on_delivery(self, event: _Event) -> None:
        decision: Decision = event.data
        dyad = self.dyads[event.group_id]
        success = _rng(self.plan.seed, dyad.group_id, "deliver", decision.decision_type, decision.decision_idx).random()
        decision.delivered = decision.action == 1 and success < self.plan.delivery_success
        dyad.record_delivery(decision.role, event.time, decision.delivered)
        mtw = (decision.day, decision.slot) if decision.role == "aya" else None
        self.auditor.note_delivery(dyad.group_id, decision.role, event.time, decision.delivered, mtw)

    def _on_mtw(self, event: _Event) -> None:
        day, slot = event.data
        self.dyads[event.group_id].resolve_mtw(day, slot, self.plan.outcome_model)

    def _on_end_of_day(self, event: _Event) -> None:
        self.dyads[event.group_id].resolve_app_opens(event.data, self.plan.outcome_model)

    def _on_survey(self, event: _Event) -> None:
        self.dyads[event.group_id].resolve_survey(event.time, self.plan.outcome_model)
        self.auditor.note_survey(event.group_id, event.time)

    def _on_update(self, event: _Event) -> None:
        record = self._call("POST", "update", {"timestamp": self._ts(event.time)}, note=f"week {event.data}")
        self.update_rids.append((event.time, record.rid, record.status))
        if record.status == 202 and record.rid and self.hooks.wait_for_update is not None:
            self.hooks.wait_for_update(record.rid, event.time)

    def _on_pull(self, event: _Event) -> None:
        record = self._call("GET", "monitor_alarms", None)
        body = record.response or {}
        if record.status == 200 and isinstance(body.get("alarms"), list):
            for alarm in body["alarms"]:
                event_id = alarm.get("event_id") if isinstance(alarm, dict) else None
                if event_id and event_id not in self.alarms:
                    self.alarms[event_id] = {**alarm, "pulled_at": event.time.isoformat()}

    def _on_hook(self, event: _Event) -> None:
        event.data(event.time)

    # ------------------------------------------------------------- calls

    def _ts(self, when: dt.datetime) -> str:
        return format_timestamp(when, self.plan.tz_aware_timestamps)

    def _call(
        self,
        method: str,
        endpoint: str,
        payload: dict | None,
        *,
        expect: set[int] | None = None,
        group_id: str | None = None,
        decision: Decision | None = None,
        kind: str | None = None,
        note: str | None = None,
    ) -> RequestRecord:
        expect = expect or {EXPECTED_STATUS[endpoint]}
        try:
            status, body = self.transport.request(method, ENDPOINT_PATHS[endpoint], payload)
        except TransportError as exc:
            status, body, note = None, None, f"transport error: {exc}"
        self._seq += 1
        body = body if isinstance(body, dict) else None
        ok = status in expect or (endpoint, status) in self.faults.allowed_statuses
        record = RequestRecord(
            seq=self._seq,
            logical_time=self.now,
            method=method,
            endpoint=endpoint,
            status=status,
            title=body.get("title") if body else None,
            message=body.get("message") if body else None,
            rid=body.get("rid") if body else None,
            expected=ok,
            group_id=group_id,
            decision_type=decision.decision_type if decision else None,
            decision_idx=decision.decision_idx if decision else None,
            kind=kind,
            note=note,
            request=payload if self.plan.record_payloads else None,
            response=body if (self.plan.record_payloads or endpoint in ("action", "update", "monitor_alarms")) else None,
        )
        self.records.append(record)
        if self.log is not None:
            self.log(
                f"{record.logical_time} {method} {endpoint} -> {status} {record.title or ''}"
                f"{' [UNEXPECTED]' if not ok else ''}"
            )
        if status is not None:
            self._check_envelope(record, body)
        if not ok:
            self.unexpected.append(record)
            if self.plan.strict:
                raise UnexpectedStatusError(
                    f"{method} {endpoint} at {self.now}: status {status} not in {sorted(expect)}; "
                    f"title={record.title!r} message={record.message!r}"
                )
        if self.hooks.after_request is not None:
            self.hooks.after_request(record, self)
        return record

    def _check_envelope(self, record: RequestRecord, body: dict | None) -> None:
        """API-Spec §2 preamble: five always-present fields, ``status`` echoes HTTP, nonempty ``rid``."""
        problems = []
        if body is None:
            problems.append("response is not a JSON object")
        else:
            problems += [f"missing envelope field {k!r}" for k in ENVELOPE_KEYS if k not in body]
            if body.get("status") != record.status:
                problems.append(f"body status {body.get('status')!r} != HTTP {record.status}")
            if not (isinstance(body.get("rid"), str) and body.get("rid")):
                problems.append("rid empty or missing")
        if problems:
            self._violation("envelope", record.group_id, record.logical_time, problems, endpoint=record.endpoint)

    def _check_action_body(self, decision: Decision, body: dict) -> None:
        """API-Spec §2.2 response fields of a 2xx /action."""
        problems = []
        for name in ("group_id", "decision_type", "decision_idx"):
            if body.get(name) != getattr(decision, name):
                problems.append(f"{name} echo {body.get(name)!r} != {getattr(decision, name)!r}")
        pi = body.get("action_prob")
        if not (_is_number(pi) and 0.0 <= pi <= 1.0):
            problems.append(f"action_prob {pi!r} is not a probability")
        if not isinstance(body.get("warmup"), bool):
            problems.append(f"warmup {body.get('warmup')!r} is not a bool")
        elif body["warmup"]:
            if pi != 0.5:
                problems.append(f"warm-up decision with action_prob {pi!r} != 0.5")
            if body.get("model_param") is not None:
                problems.append("warm-up decision with a non-null model_param")
        if problems:
            self._violation("action_response", decision.group_id, decision.time, problems, decision=decision)

    def _violation(self, what: str, group_id: str | None, when: dt.datetime | None, issues: list[str], **extra) -> None:
        entry = {
            "what": what,
            "group_id": group_id,
            "logical_time": when.isoformat() if when else None,
            "issues": list(issues),
        }
        decision = extra.pop("decision", None)
        if decision is not None:
            entry.update(decision_type=decision.decision_type, decision_idx=decision.decision_idx)
        entry.update({k: v for k, v in extra.items() if v is not None})
        self.violations.append(entry)
        if self.plan.strict:
            error = SnapshotValidationError if what == "snapshot" else ContractViolationError
            raise error(json.dumps(entry, default=str))


# ------------------------------------------------------------------ results


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def q(p: float) -> float:
        k = (len(ordered) - 1) * p
        lo, hi = math.floor(k), math.ceil(k)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)

    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "min": ordered[0],
        "p05": q(0.05),
        "p25": q(0.25),
        "p50": q(0.5),
        "p75": q(0.75),
        "p95": q(0.95),
        "max": ordered[-1],
    }


class SimulationResult:
    """What one run produced: request log, decisions, pulled alarms, unexpected statuses, rule violations."""

    def __init__(self, simulator: HostSimulator):
        self.simulator = simulator
        self.plan = simulator.plan
        self.records = simulator.records
        self.unexpected = simulator.unexpected
        self.violations = simulator.violations
        self.alarms = simulator.alarms
        self.decisions = [d for dyad in simulator.dyads.values() for d in dyad.decisions.values()]
        self.decisions.sort(key=lambda d: (d.time, d.group_id, d.decision_type))

    # ------------------------------------------------------------- counts

    def counts_by_endpoint(self) -> dict[str, dict[str, int]]:
        out: dict[str, Counter] = defaultdict(Counter)
        for record in self.records:
            out[record.endpoint][str(record.status)] += 1
        return {endpoint: dict(sorted(c.items())) for endpoint, c in sorted(out.items())}

    def ok_count(self, endpoint: str) -> int:
        return sum(1 for r in self.records if r.endpoint == endpoint and r.status is not None and 200 <= r.status < 300)

    def decisions_by_source(self) -> dict[str, int]:
        return dict(Counter(d.source or "not_reached" for d in self.decisions))

    def decision_table(self) -> dict[str, dict[int, dict[str, int]]]:
        """Per agent and calendar week: decisions, warm-up, learned and host-side (F-A1) counts."""
        table: dict[str, dict[int, Counter]] = {agent: defaultdict(Counter) for agent in AGENTS}
        for d in self.decisions:
            cell = table[d.decision_type][d.calendar_week]
            cell["n"] += 1
            if d.source != "api":
                cell["host_local"] += 1
            elif d.warmup:
                cell["warmup"] += 1
            else:
                cell["learned"] += 1
        return {agent: {week: dict(c) for week, c in sorted(weeks.items())} for agent, weeks in table.items()}

    def pi_table(self) -> dict[str, dict[int, dict]]:
        """Per agent and calendar week: distribution of ``action_prob`` over learned decisions."""
        values: dict[str, dict[int, list[float]]] = {agent: defaultdict(list) for agent in AGENTS}
        for d in self.decisions:
            if d.source == "api" and d.warmup is False and _is_number(d.action_prob):
                values[d.decision_type][d.calendar_week].append(float(d.action_prob))
        return {agent: {week: _quantiles(v) for week, v in sorted(weeks.items())} for agent, weeks in values.items()}

    def alarms_by_item(self) -> dict[str, dict[str, int]]:
        out: dict[str, Counter] = defaultdict(Counter)
        for alarm in self.alarms.values():
            out[str(alarm.get("item_id"))][str(alarm.get("severity"))] += 1
        return {item: dict(c) for item, c in sorted(out.items())}

    def calendar_mismatches(self) -> list[dict]:
        """(dyad, agent) pairs whose API decisions differ from the calendar minus host-side decisions."""
        expected = self.simulator.expected_decisions()
        made = Counter((d.group_id, d.decision_type) for d in self.decisions if d.source == "api")
        local = Counter((d.group_id, d.decision_type) for d in self.decisions if d.source in ("host_skip", "host_fallback"))
        out = []
        for key, n in sorted(expected.items()):
            if made[key] + local[key] != n:
                out.append({"group_id": key[0], "decision_type": key[1], "expected": n, "api": made[key], "host_local": local[key]})
        return out

    def legacy_counts(self) -> dict:
        """The summary dict of the former ``run_simulation`` (``add_group``/``action``/... and ``errors``)."""
        return {
            "add_group": self.ok_count("register_group"),
            "action": self.ok_count("action"),
            "upload_data": self.ok_count("upload_data"),
            "update": self.ok_count("update"),
            "monitor_alarms": self.ok_count("monitor_alarms"),
            "errors": [r.to_json() for r in self.unexpected] + list(self.violations),
        }

    def summary(self) -> dict:
        return {
            "plan": self.plan.to_json(),
            "faults": self.simulator.faults.to_json(),
            "n_requests": len(self.records),
            "requests": self.counts_by_endpoint(),
            "decisions": {
                "expected_by_calendar": sum(self.simulator.expected_decisions().values()),
                "by_source": self.decisions_by_source(),
                "by_agent_week": self.decision_table(),
                "calendar_mismatches": self.calendar_mismatches(),
            },
            "pi_learned": self.pi_table(),
            "updates": [
                {"logical_time": t.isoformat(), "rid": rid, "status": status}
                for t, rid, status in self.simulator.update_rids
            ],
            "alarms_pulled": {"n": len(self.alarms), "by_item": self.alarms_by_item()},
            "unexpected": [r.to_json() for r in self.unexpected],
            "violations": list(self.violations),
        }


# ---------------------------------------------------------- legacy wrapper


def run_simulation(
    client,
    base_date: dt.date | None = None,
    num_weeks: int = 4,
    num_dyads: int | None = None,
    verbose: bool = False,
    group_prefix: str = "",
    seed: int = 42,
    strict: bool = False,
) -> dict:
    """
    Drive an in-process app (``app.test_client()``) through ``num_weeks``
    calendar weeks with ``num_dyads`` dyads enrolled one a week. Returns the
    former summary dict ``{"add_group", "action", "upload_data", "update",
    "monitor_alarms", "errors"}`` (used by the analysis scripts in ``tools/``).
    """
    plan = TrialPlan(
        num_dyads=5 if num_dyads is None else num_dyads,
        num_weeks=num_weeks,
        start_date=base_date or dt.date(2025, 1, 5),
        seed=seed,
        group_prefix=group_prefix,
        strict=strict,
    )
    simulator = HostSimulator(FlaskClientTransport(client), plan, log=print if verbose else None)
    return simulator.run().legacy_counts()


def iter_dyad_snapshots(simulator: HostSimulator) -> Iterable[tuple[str, dt.datetime, str, dict]]:
    """``(group_id, time, kind, data)`` of every upload a finished run sent (requires ``record_payloads``)."""
    for record in simulator.records:
        if record.endpoint == "upload_data" and record.request is not None:
            yield record.group_id, record.logical_time, record.kind, record.request["data"]
