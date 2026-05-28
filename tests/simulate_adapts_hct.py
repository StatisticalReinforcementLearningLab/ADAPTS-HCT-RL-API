"""
ADAPTS-HCT protocol-faithful study replay.

The replay simulates:
- 25 dyads recruited at roughly one per week
- 100 active calendar days per dyad
- weekly Sunday model updates
- Monday game decisions
- Monday-Saturday AYA AM/PM and care-partner AM message decisions
- delayed upload of outcomes via /upload_data
"""

from __future__ import annotations

import datetime
import random
from dataclasses import dataclass, field
from typing import Generator


NUM_DYADS = 25
DAYS_ACTIVE = 100
WEEKS_ACTIVE = (DAYS_ACTIVE + 6) // 7
DEFAULT_NUM_WEEKS = NUM_DYADS + WEEKS_ACTIVE
MONDAY_TO_SATURDAY = range(0, 6)
WARMUP_DYAD_COUNT = 5


def _make_timestamp(base_date: datetime.date, hour: int, minute: int = 0) -> str:
    dt = datetime.datetime.combine(base_date, datetime.time(hour, minute))
    return dt.isoformat()


def _parse_timestamp(value: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(value)


@dataclass
class DyadState:
    group_id: str
    recruit_date: datetime.date
    consent_end_date: datetime.date
    aya_latent: float
    cp_latent: float
    relationship_latent: float
    warmup: bool = False
    global_decision_index: int = 0
    aya_decision_index: int = 0
    cp_decision_index: int = 0
    game_decision_index: int = 0
    current_game_on: int = 0
    prior_game_action: str | int = "miss"
    notifications: list[dict] = field(default_factory=list)
    medication_reports: dict[tuple[datetime.date, str], dict] = field(default_factory=dict)
    app_opens: dict[str, dict[datetime.date, bool]] = field(
        default_factory=lambda: {"aya": {}, "cp": {}}
    )
    diaries: dict[str, dict[datetime.date, dict]] = field(
        default_factory=lambda: {"aya": {}, "cp": {}}
    )
    weekly_surveys: dict[int, dict] = field(default_factory=dict)


class ProtocolTrialSimulator:
    def __init__(
        self,
        base_date: datetime.date,
        num_weeks: int = DEFAULT_NUM_WEEKS,
        num_dyads: int = NUM_DYADS,
        seed: int | None = 42,
        group_prefix: str = "",
    ):
        self.base_date = base_date
        self.num_weeks = num_weeks
        self.num_dyads = num_dyads
        self.group_prefix = group_prefix
        self.rng = random.Random(seed)
        self.dyads = self._build_dyads()
        self.pending_uploads: list[dict] = []

    def iter_schedule_events(self) -> Generator[dict, None, None]:
        add_groups = []
        timeline = []

        for dyad_idx, dyad in enumerate(self.dyads):
            add_groups.append(
                {
                    "type": "add_group",
                    "timestamp": _make_timestamp(dyad.recruit_date, 0, 0),
                    "payload": {
                        "group_id": dyad.group_id,
                        "member_list": [f"aya_{dyad_idx + 1:03d}", f"cp_{dyad_idx + 1:03d}"],
                        "consent_start_date": dyad.recruit_date.isoformat(),
                        "consent_end_date": dyad.consent_end_date.isoformat(),
                        "warmup": dyad.warmup,
                    },
                }
            )

        for week in range(self.num_weeks):
            sunday = self.base_date + datetime.timedelta(weeks=week)
            monday = sunday + datetime.timedelta(days=1)

            timeline.append(
                {
                    "type": "update",
                    "timestamp": _make_timestamp(sunday, 3, 0),
                    "payload": {
                        "timestamp": _make_timestamp(sunday, 3, 0),
                        "callback_url": "http://localhost:5000/callback",
                    },
                }
            )

            for dyad in self.dyads:
                if not self._is_active_in_week(dyad, week):
                    continue

                timeline.append(
                    {
                        "type": "action",
                        "timestamp": _make_timestamp(monday, 6, 0),
                        "group_id": dyad.group_id,
                        "decision_type": "dyad_game",
                        "week": week,
                        "day_offset": 0,
                        "slot": None,
                    }
                )

                for day_offset in MONDAY_TO_SATURDAY:
                    current_date = monday + datetime.timedelta(days=day_offset)
                    if current_date > dyad.consent_end_date:
                        continue

                    timeline.append(
                        {
                            "type": "action",
                            "timestamp": _make_timestamp(current_date, 9, 0),
                            "group_id": dyad.group_id,
                            "decision_type": "aya_message",
                            "week": week,
                            "day_offset": day_offset,
                            "slot": "am",
                        }
                    )
                    timeline.append(
                        {
                            "type": "action",
                            "timestamp": _make_timestamp(current_date, 9, 5),
                            "group_id": dyad.group_id,
                            "decision_type": "cp_message",
                            "week": week,
                            "day_offset": day_offset,
                            "slot": None,
                        }
                    )
                    timeline.append(
                        {
                            "type": "action",
                            "timestamp": _make_timestamp(current_date, 21, 0),
                            "group_id": dyad.group_id,
                            "decision_type": "aya_message",
                            "week": week,
                            "day_offset": day_offset,
                            "slot": "pm",
                        }
                    )

        add_groups.sort(key=lambda event: event["payload"]["group_id"])
        timeline.sort(key=lambda event: (event["timestamp"], event.get("group_id", "")))

        for event in add_groups:
            yield event
        for event in timeline:
            yield event

    def build_action_payload(self, event: dict) -> dict:
        dyad = self._dyad(event["group_id"])
        event_dt = _parse_timestamp(event["timestamp"])
        self._ensure_histories(dyad, event_dt.date())

        decision_type = event["decision_type"]
        dyad.global_decision_index += 1
        decision_idx = dyad.global_decision_index

        if decision_type == "dyad_game":
            dyad.game_decision_index += 1
            context = self._build_game_context(dyad, event_dt.date())
        elif decision_type == "cp_message":
            dyad.cp_decision_index += 1
            context = self._build_cp_context(dyad, event_dt.date())
        else:
            dyad.aya_decision_index += 1
            context = self._build_aya_context(dyad, event_dt.date(), event["slot"])

        return {
            "group_id": dyad.group_id,
            "timestamp": event["timestamp"],
            "decision_idx": decision_idx,
            "decision_type": decision_type,
            "context": context,
        }

    def schedule_upload(self, payload: dict, response_json: dict):
        dyad = self._dyad(payload["group_id"])
        decision_type = payload["decision_type"]
        action = int(response_json["action"])
        event_dt = _parse_timestamp(payload["timestamp"])
        due_dt = self._upload_due_datetime(decision_type, event_dt)

        if action == 1:
            role = "aya" if decision_type == "aya_message" else "cp" if decision_type == "cp_message" else "game"
            dyad.notifications.append(
                {"role": role, "decision_type": decision_type, "timestamp": payload["timestamp"]}
            )
        if decision_type == "dyad_game":
            dyad.current_game_on = action

        self.pending_uploads.append(
            {
                "due_timestamp": due_dt.isoformat(),
                "action_payload": payload,
                "action_response": response_json,
            }
        )
        self.pending_uploads.sort(key=lambda item: item["due_timestamp"])

    def pop_due_uploads(self, current_timestamp: str) -> list[dict]:
        current_dt = _parse_timestamp(current_timestamp)
        due = []
        remaining = []
        for item in self.pending_uploads:
            if _parse_timestamp(item["due_timestamp"]) <= current_dt:
                due.append(self._build_upload_payload(item))
            else:
                remaining.append(item)
        self.pending_uploads = remaining
        return due

    def flush_all_uploads(self) -> list[dict]:
        items = [self._build_upload_payload(item) for item in self.pending_uploads]
        self.pending_uploads = []
        return items

    def _build_upload_payload(self, pending_item: dict) -> dict:
        action_payload = pending_item["action_payload"]
        action_response = pending_item["action_response"]
        dyad = self._dyad(action_payload["group_id"])
        due_dt = _parse_timestamp(pending_item["due_timestamp"])
        self._ensure_histories(dyad, due_dt.date())
        outcome = self._generate_outcome(
            dyad,
            action_payload["decision_type"],
            action_payload["context"],
            int(action_response["action"]),
            due_dt.date(),
        )
        return {
            "group_id": action_payload["group_id"],
            "decision_idx": action_payload["decision_idx"],
            "decision_type": action_payload["decision_type"],
            "timestamp": due_dt.isoformat(),
            "data": {
                "context": action_payload["context"],
                "action": action_response["action"],
                "action_prob": action_response["action_prob"],
                "state": action_response["state"],
                "outcome": outcome,
            },
        }

    def _build_dyads(self) -> list[DyadState]:
        dyads = []
        for idx in range(self.num_dyads):
            recruit_date = self.base_date + datetime.timedelta(weeks=idx)
            consent_end_date = recruit_date + datetime.timedelta(days=DAYS_ACTIVE - 1)
            dyads.append(
                DyadState(
                    group_id=f"{self.group_prefix}dyad_{idx + 1:03d}",
                    recruit_date=recruit_date,
                    consent_end_date=consent_end_date,
                    aya_latent=self.rng.uniform(0.35, 0.85),
                    cp_latent=self.rng.uniform(0.35, 0.85),
                    relationship_latent=self.rng.uniform(2.0, 4.5),
                    warmup=False,  # cohort warmup removed — EB does per-dyad week-1 instead
                )
            )
        return dyads

    def _dyad(self, group_id: str) -> DyadState:
        return next(dyad for dyad in self.dyads if dyad.group_id == group_id)

    def _is_active_in_week(self, dyad: DyadState, week: int) -> bool:
        week_start = self.base_date + datetime.timedelta(weeks=week)
        week_end = week_start + datetime.timedelta(days=6)
        return dyad.recruit_date <= week_end and week_start <= dyad.consent_end_date

    def _upload_due_datetime(self, decision_type: str, event_dt: datetime.datetime) -> datetime.datetime:
        if decision_type == "aya_message":
            return event_dt + datetime.timedelta(hours=1)
        if decision_type == "cp_message":
            return event_dt + datetime.timedelta(hours=12)
        return datetime.datetime.combine(
            event_dt.date() + datetime.timedelta(days=6),
            datetime.time(18, 0),
        )

    def _ensure_histories(self, dyad: DyadState, up_to_date: datetime.date):
        current_date = dyad.recruit_date
        if dyad.app_opens["aya"]:
            current_date = max(dyad.app_opens["aya"].keys()) + datetime.timedelta(days=1)

        while current_date <= up_to_date:
            self._generate_daily_history(dyad, current_date)
            if current_date.weekday() == 6:
                self._generate_weekly_survey(dyad, current_date)
            current_date += datetime.timedelta(days=1)

    def _generate_daily_history(self, dyad: DyadState, current_date: datetime.date):
        for role, latent in (("aya", dyad.aya_latent), ("cp", dyad.cp_latent)):
            if current_date in dyad.app_opens[role]:
                continue
            recent_notifications = self._count_notifications(
                dyad,
                role=role,
                window_start=datetime.datetime.combine(current_date - datetime.timedelta(days=2), datetime.time.min),
                window_end=datetime.datetime.combine(current_date, datetime.time.min),
            )
            open_prob = min(0.95, max(0.1, latent + 0.03 * recent_notifications))
            diary_prob = min(0.9, max(0.05, latent + 0.02 * recent_notifications))
            opened = self.rng.random() < open_prob
            completed = opened and (self.rng.random() < diary_prob)
            dyad.app_opens[role][current_date] = opened
            if completed:
                dyad.diaries[role][current_date] = {
                    "completed": True,
                    "mood": round(min(5.0, max(1.0, 2.5 + latent * 2 + self.rng.uniform(-0.8, 0.8))), 2),
                    "physical": round(min(5.0, max(1.0, 2.0 + latent * 2 + self.rng.uniform(-0.8, 0.8))), 2),
                    "score": round(min(5.0, max(1.0, 2.2 + latent * 2 + self.rng.uniform(-0.6, 0.6))), 2),
                }
            else:
                dyad.diaries[role][current_date] = {
                    "completed": False,
                    "mood": "miss",
                    "physical": "miss",
                    "score": 0.0,
                }

    def _generate_weekly_survey(self, dyad: DyadState, sunday_date: datetime.date):
        week_in_study = self._week_in_study(dyad, sunday_date)
        if week_in_study in dyad.weekly_surveys:
            return
        # Survey completion: clipped at [0.80, 0.95] so the weekly miss rate is
        # ~20% (down from the previous ~50%). REL's reward signal is too weak
        # under heavy missingness — see Brainstorming/slides.tex for the m, √v
        # diagnostic that motivated the change.
        cp_completed = self.rng.random() < min(0.95, max(0.80, dyad.cp_latent))
        aya_completed = self.rng.random() < min(0.95, max(0.80, dyad.aya_latent))
        # Action effect: game_on adds +1.0 to relationship_score, game_off
        # adds 0 (no penalty — was -0.5 in earlier versions). Expected weekly
        # reward gap ≈ +1.0 · Pr(survey completed) ≈ +0.85. REL pool has
        # only ~350 obs against D = 14, so its action-contrast variance v
        # stays ~0.1; we need m ≥ ~0.85 for the smooth-allocation MC to land
        # close to L_max for late-recruited dyads.
        game_bonus = 1.0 if dyad.current_game_on == 1 else 0.0
        cp_score = round(
            min(5.0, max(1.0, dyad.relationship_latent + game_bonus + self.rng.uniform(-0.7, 0.7))),
            2,
        )
        aya_score = round(
            min(5.0, max(1.0, dyad.relationship_latent + game_bonus + self.rng.uniform(-0.7, 0.7))),
            2,
        )
        dyad.weekly_surveys[week_in_study] = {
            "cp_completed": cp_completed,
            "aya_completed": aya_completed,
            "cp_score": cp_score if cp_completed else "miss",
            "aya_score": aya_score if aya_completed else "miss",
        }

    def _build_aya_context(self, dyad: DyadState, current_date: datetime.date, slot: str) -> dict:
        prior_date = current_date if slot == "pm" else current_date - datetime.timedelta(days=1)
        prior_slot = "am" if slot == "pm" else "pm"
        prior_report = dyad.medication_reports.get((prior_date, prior_slot), {"med_adherence": "miss"})
        prev_day = current_date - datetime.timedelta(days=1)
        prior_week = self._week_in_study(dyad, current_date - datetime.timedelta(days=7))
        survey = dyad.weekly_surveys.get(prior_week, {"cp_score": "miss", "aya_score": "miss"})
        return {
            "slot": slot,
            "agent_decision_index": dyad.aya_decision_index,
            "day_in_study": self._day_in_study(dyad, current_date),
            "week_in_study": self._week_in_study(dyad, current_date),
            "prior_med_adherence": prior_report["med_adherence"],
            "aya_diary": self._diary_block(dyad, "aya", prev_day),
            "relationship_quality_cp": survey["cp_score"],
            "relationship_quality_aya": survey["aya_score"],
            "aya_app_engagement": self._engagement_level(dyad, "aya", current_date),
            "aya_app_burden": round(self._notification_burden(dyad, "aya", current_date), 3),
            "aya_missing_rate_7d": round(self._missing_rate(dyad, "aya", current_date), 3),
            "current_game_on": dyad.current_game_on,
        }

    def _build_cp_context(self, dyad: DyadState, current_date: datetime.date) -> dict:
        prev_day = current_date - datetime.timedelta(days=1)
        prior_week = self._week_in_study(dyad, current_date - datetime.timedelta(days=7))
        survey = dyad.weekly_surveys.get(prior_week, {"cp_score": "miss", "aya_score": "miss"})
        return {
            "agent_decision_index": dyad.cp_decision_index,
            "day_in_study": self._day_in_study(dyad, current_date),
            "week_in_study": self._week_in_study(dyad, current_date),
            "cp_diary": self._diary_block(dyad, "cp", prev_day),
            "cp_app_engagement": self._engagement_level(dyad, "cp", current_date),
            "cp_app_burden": round(self._notification_burden(dyad, "cp", current_date), 3),
            "cp_missing_rate_7d": round(self._missing_rate(dyad, "cp", current_date), 3),
            "relationship_quality_cp": survey["cp_score"],
            "relationship_quality_aya": survey["aya_score"],
            "current_game_on": dyad.current_game_on,
        }

    def _build_game_context(self, dyad: DyadState, current_date: datetime.date) -> dict:
        prev_week = self._week_in_study(dyad, current_date - datetime.timedelta(days=7))
        survey = dyad.weekly_surveys.get(prev_week, {"cp_score": "miss", "aya_score": "miss"})
        prior_game_action = dyad.current_game_on if dyad.game_decision_index > 1 else "miss"
        return {
            "agent_decision_index": dyad.game_decision_index,
            "week_in_study": self._week_in_study(dyad, current_date),
            "relationship_quality_aya": survey["aya_score"],
            "relationship_quality_cp": survey["cp_score"],
            "aya_app_engagement": self._engagement_level(dyad, "aya", current_date),
            "cp_app_engagement": self._engagement_level(dyad, "cp", current_date),
            "aya_app_burden": round(self._notification_burden(dyad, "aya", current_date), 3),
            "cp_app_burden": round(self._notification_burden(dyad, "cp", current_date), 3),
            "prior_game_action": prior_game_action,
            "aya_diary_summary": round(self._diary_summary(dyad, "aya", current_date), 3),
            "cp_diary_summary": round(self._diary_summary(dyad, "cp", current_date), 3),
        }

    def _generate_outcome(
        self,
        dyad: DyadState,
        decision_type: str,
        context: dict,
        action: int,
        due_date: datetime.date,
    ) -> dict:
        if decision_type == "aya_message":
            # Action effect on adherence probability. Boost = 0.70 → net Δr ≈
            # +0.45 under the 4-tier {0,1,2,3} reward (which penalises a=1 by
            # +1 reward when prompted_by_message=True). With this boost the
            # learned θ[action] is large enough (~0.45) for the smooth-allocation
            # to saturate near L_max = 0.8 for late-recruited dyads.
            adherence_prob = min(
                0.95,
                max(0.05, 0.05 + 0.7 * action + 0.1 * dyad.aya_latent
                    - 0.05 * context["aya_missing_rate_7d"]),
            )
            # Report rate clipped to [0.70, 0.95] (avg ~0.80) so the AYA
            # reward column is informative for the learner.
            report_prob = min(0.95, max(0.70, 0.65 + 0.05 * context["aya_app_engagement"]))
            if self.rng.random() > report_prob:
                med_adherence = "miss"
                prompted = False
            else:
                med_adherence = 1 if self.rng.random() < adherence_prob else 0
                prompted = bool(action == 1 and med_adherence == 1 and self.rng.random() < 0.75)
            dyad.medication_reports[(due_date, context["slot"])] = {
                "med_adherence": med_adherence,
                "prompted_by_message": prompted,
            }
            return {
                "med_adherence": med_adherence,
                "prompted_by_message": prompted,
            }

        if decision_type == "cp_message":
            diary = dyad.diaries["cp"].get(due_date, {"completed": False, "score": 0.0})
            # VERIFICATION TEST: send-CP-message bumps the diary score by 1.0
            # (clipped at 5.0) when the diary was completed. Net expected
            # reward gain is ≈ 1.0 * Pr(completed) ≈ 0.5–0.8 per a=1.
            base_score = float(diary["score"])
            boosted = min(5.0, base_score + 1.0 * action) if diary["completed"] else 0.0
            return {
                "daily_diary_completed": bool(diary["completed"]),
                "daily_diary_score": boosted,
            }

        week_idx = self._week_in_study(dyad, due_date)
        survey = dyad.weekly_surveys.get(
            week_idx,
            {"cp_completed": False, "cp_score": 0.0},
        )
        return {
            "weekly_survey_completed": bool(survey["cp_completed"]),
            "weekly_relationship_score": float(survey["cp_score"]) if survey["cp_completed"] else 0.0,
        }

    def _day_in_study(self, dyad: DyadState, date_value: datetime.date) -> int:
        return (date_value - dyad.recruit_date).days + 1

    def _week_in_study(self, dyad: DyadState, date_value: datetime.date) -> int:
        if date_value < dyad.recruit_date:
            return 0
        days = (date_value - dyad.recruit_date).days
        return (days // 7) + 1

    def _diary_block(self, dyad: DyadState, role: str, date_value: datetime.date) -> dict:
        diary = dyad.diaries[role].get(date_value, {"completed": False, "mood": "miss", "physical": "miss"})
        return {
            "mood": diary["mood"] if diary["completed"] else "miss",
            "physical": diary["physical"] if diary["completed"] else "miss",
        }

    def _engagement_level(self, dyad: DyadState, role: str, current_date: datetime.date) -> int:
        d1 = current_date - datetime.timedelta(days=1)
        d2 = current_date - datetime.timedelta(days=2)
        d3 = current_date - datetime.timedelta(days=3)
        opened_d1 = dyad.app_opens[role].get(d1, False)
        opened_d2 = dyad.app_opens[role].get(d2, False)
        opened_d3 = dyad.app_opens[role].get(d3, False)
        if not opened_d1 and not opened_d2 and not opened_d3:
            return 1
        if not opened_d1 and (opened_d2 or opened_d3):
            return 2
        if opened_d1 and not dyad.diaries[role].get(d1, {"completed": False})["completed"]:
            return 3
        return 4

    def _notification_burden(self, dyad: DyadState, role: str, current_date: datetime.date) -> float:
        burden = 0.0
        current_dt = datetime.datetime.combine(current_date, datetime.time.max)
        for notification in dyad.notifications:
            if notification["role"] != role:
                continue
            sent_dt = _parse_timestamp(notification["timestamp"])
            delta_days = max(0.0, (current_dt - sent_dt).total_seconds() / 86400.0)
            if delta_days <= 7.0:
                burden += 1.0 / (1.0 + delta_days)
        return burden

    def _count_notifications(
        self,
        dyad: DyadState,
        role: str,
        window_start: datetime.datetime,
        window_end: datetime.datetime,
    ) -> int:
        count = 0
        for notification in dyad.notifications:
            if notification["role"] != role:
                continue
            sent_dt = _parse_timestamp(notification["timestamp"])
            if window_start <= sent_dt <= window_end:
                count += 1
        return count

    def _diary_summary(self, dyad: DyadState, role: str, current_date: datetime.date) -> float:
        """
        Average completed diary score over the prior 7 days, normalized to [0, 1].
        Returns 0.0 if no diary was completed in that window. System-side
        missingness — value is always defined.
        """
        scores: list[float] = []
        for offset in range(1, 8):
            date_value = current_date - datetime.timedelta(days=offset)
            diary = dyad.diaries[role].get(date_value, {"completed": False})
            if not diary["completed"]:
                continue
            try:
                scores.append(float(diary.get("score", 0.0)))
            except (TypeError, ValueError):
                continue
        if not scores:
            return 0.0
        return max(0.0, min(1.0, sum(scores) / (5.0 * len(scores))))

    def _missing_rate(self, dyad: DyadState, role: str, current_date: datetime.date) -> float:
        dates = [current_date - datetime.timedelta(days=offset) for offset in range(1, 8)]
        total = len(dates)
        missing = 0
        for date_value in dates:
            diary = dyad.diaries[role].get(date_value, {"completed": False})
            if not diary["completed"]:
                missing += 1
        if role == "aya":
            med_missing = 0
            med_total = 0
            for date_value in dates:
                for slot in ("am", "pm"):
                    med_total += 1
                    report = dyad.medication_reports.get((date_value, slot), {"med_adherence": "miss"})
                    if report["med_adherence"] == "miss":
                        med_missing += 1
            return (missing + med_missing) / (total + med_total)
        return missing / total


def iter_simulation_events(
    base_date: datetime.date,
    num_weeks: int = DEFAULT_NUM_WEEKS,
    num_dyads: int = NUM_DYADS,
    seed: int | None = 42,
    group_prefix: str = "",
) -> Generator[dict, None, None]:
    simulator = ProtocolTrialSimulator(
        base_date=base_date,
        num_weeks=num_weeks,
        num_dyads=num_dyads,
        seed=seed,
        group_prefix=group_prefix,
    )
    for event in simulator.iter_schedule_events():
        if event["type"] != "action":
            yield event
            continue
        payload = simulator.build_action_payload(event)
        yield {"type": "action", "timestamp": event["timestamp"], "payload": payload}


def run_simulation(
    client,
    base_date: datetime.date | None = None,
    num_weeks: int = 4,
    num_dyads: int | None = None,
    verbose: bool = False,
    group_prefix: str = "",
):
    if base_date is None:
        base_date = datetime.date(2025, 1, 5)
    if num_dyads is None:
        num_dyads = min(5, NUM_DYADS)

    simulator = ProtocolTrialSimulator(
        base_date=base_date,
        num_weeks=num_weeks,
        num_dyads=num_dyads,
        seed=42,
        group_prefix=group_prefix,
    )
    results = {"add_group": 0, "action": 0, "upload_data": 0, "update": 0, "errors": []}

    def submit_uploads(payloads: list[dict]):
        for upload_payload in payloads:
            upload_response = client.post("/api/v1/upload_data", json=upload_payload)
            if upload_response.status_code in (200, 201):
                results["upload_data"] += 1
            else:
                results["errors"].append(
                    {
                        "type": "upload_data",
                        "status": upload_response.status_code,
                        "body": upload_response.get_json(),
                    }
                )

    for event in simulator.iter_schedule_events():
        submit_uploads(simulator.pop_due_uploads(event["timestamp"]))

        if event["type"] == "add_group":
            payload = event["payload"]
            if verbose:
                print(f"[API] add_group  group_id={payload.get('group_id')}")
            response = client.post("/api/v1/add_group", json=payload)
            if response.status_code in (200, 201):
                results["add_group"] += 1
            else:
                results["errors"].append(
                    {"type": "add_group", "status": response.status_code, "body": response.get_json()}
                )
            continue

        if event["type"] == "update":
            payload = event["payload"]
            if verbose:
                print(f"[API] update  timestamp={payload.get('timestamp')}")
            response = client.post("/api/v1/update", json=payload)
            if response.status_code in (200, 202):
                results["update"] += 1
            else:
                results["errors"].append(
                    {"type": "update", "status": response.status_code, "body": response.get_json()}
                )
            continue

        payload = simulator.build_action_payload(event)
        response = client.post("/api/v1/action", json=payload)
        if response.status_code in (200, 201):
            results["action"] += 1
            response_json = response.get_json()
            simulator.schedule_upload(payload, response_json)
            if verbose:
                print(
                    f"[API] action  group_id={payload['group_id']} "
                    f"decision_type={payload['decision_type']} "
                    f"decision_idx={payload['decision_idx']}  "
                    f"action={response_json.get('action')}  action_prob={response_json.get('action_prob')}"
                )
        else:
            results["errors"].append(
                {"type": "action", "status": response.status_code, "body": response.get_json()}
            )

    submit_uploads(simulator.flush_all_uploads())
    return results
