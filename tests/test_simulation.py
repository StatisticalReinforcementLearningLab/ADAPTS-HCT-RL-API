"""
Tests of the host simulator (``tests/simulate_adapts_hct.py``) and the
fault table (``tools/fault_scenarios.py``).

Part (a) needs no server: the snapshot builder, schedule, validator and
host-side faults are checked against API-Spec §2.3, §3 and §4.1 with the
:class:`OfflineTransport`.

Part (b) is an in-process end-to-end run (``FlaskClientTransport`` over a
``config.TestingConfig`` app): 6 dyads, two of them with non-Monday starts,
over 4 calendar weeks, with two injected host faults.
"""

from __future__ import annotations

import datetime as dt
import http.server
import json
import threading
from collections import Counter

import pytest

from tests.simulate_adapts_hct import (
    CLOSING,
    GAMMA,
    MISS,
    PRE_AYA_AM,
    PRE_AYA_PM,
    PRE_CP,
    PRE_GAME,
    SNAPSHOT_FIELDS,
    DyadHost,
    FlaskClientTransport,
    HostFaults,
    HostSimulator,
    HttpTransport,
    OfflineTransport,
    SimulatorHooks,
    SnapshotValidationError,
    TransportError,
    TrialPlan,
    UnexpectedStatusError,
    check_snapshot_rules,
    consent_end_for,
    run_simulation,
    study_week,
    week1_start,
)

MON = dt.date(2026, 11, 2)  # a Monday
TUE = dt.date(2026, 11, 3)
WED = dt.date(2026, 11, 4)


def _run(enrollment, weeks, *, seed=5, faults=None, action_fn=None, hooks=None, **plan_kwargs):
    plan = TrialPlan(
        enrollment=tuple(enrollment),
        num_weeks=weeks,
        seed=seed,
        record_payloads=True,
        **plan_kwargs,
    )
    simulator = HostSimulator(OfflineTransport(seed=seed, action_fn=action_fn), plan, faults, hooks)
    return simulator, simulator.run()


def _uploads(result, group_id):
    """Accepted uploads of one dyad: list of (time, kind, data)."""
    return [
        (r.logical_time, r.kind, r.request["data"])
        for r in result.records
        if r.endpoint == "upload_data" and r.group_id == group_id and r.status == 201
    ]


def _game_by_week(action_fn_week):
    """An action_fn: dyad_game -> ``action_fn_week(week)``, messages always 1."""

    def fn(payload):
        if payload["decision_type"] == "dyad_game":
            return action_fn_week(payload["decision_idx"])
        return 1

    return fn


# =============================================================== part (a)


class TestCalendarAndRegistration:
    def test_consent_window_covers_partial_week_and_14_weeks(self):
        for start in (MON + dt.timedelta(days=k) for k in range(7)):
            end = consent_end_for(start)
            assert end.weekday() == 6
            assert study_week(start, end) == 14
            assert study_week(start, end + dt.timedelta(days=1)) == 15
            assert 98 <= (end - start).days + 1 <= 104

    def test_registration_precedes_first_decision(self):
        simulator, result = _run([WED], 2)
        reg = [r for r in result.records if r.endpoint == "register_group"]
        assert len(reg) == 1 and reg[0].status == 201
        first_upload = min(r.logical_time for r in result.records if r.endpoint == "upload_data")
        assert reg[0].logical_time < first_upload
        spec = simulator.dyads["dyad_001"].spec
        assert spec.registration_payload() == {
            "group_id": "dyad_001",
            "member_list": ["cp_001", "aya_001"],
            "consent_start_date": "2026-11-04",
            "consent_end_date": consent_end_for(WED).isoformat(),
        }

    def test_partial_week_has_no_game_and_three_daily_decisions(self):
        simulator, result = _run([TUE], 2)
        dyad = simulator.dyads["dyad_001"]
        partial = [d for d in dyad.decisions.values() if d.study_week == 0]
        assert len(partial) == 6 * 3  # Tuesday..Sunday
        assert {d.decision_type for d in partial} == {"aya_message", "cp_message"}
        assert {(d.decision_type, d.slot, d.time.time()) for d in partial} == {
            ("aya_message", "am", dt.time(7, 30)),
            ("cp_message", "am", dt.time(7, 35)),
            ("aya_message", "pm", dt.time(19, 30)),
        }
        for t, kind, data in _uploads(result, "dyad_001"):
            if study_week(TUE, t.date()) == 0:
                assert data["week_in_study"] == 0
                assert data["current_game_on"] == MISS and data["prior_game_action"] == MISS
        games = [d for d in dyad.decisions.values() if d.decision_type == "dyad_game"]
        assert [(g.day, g.decision_idx, g.time.time()) for g in games] == [(dt.date(2026, 11, 9), 1, dt.time(6, 0))]

    def test_decision_idx_counters(self):
        simulator, _ = _run([WED], 3)
        dyad = simulator.dyads["dyad_001"]
        by_type = {t: sorted(d for (dt_, d) in dyad.decisions if dt_ == t) for t in ("aya_message", "cp_message", "dyad_game")}
        n_days = (dt.date(2026, 11, 22) - WED).days + 1  # horizon: calendar weeks 0..2
        assert by_type["aya_message"] == list(range(1, 2 * n_days + 1))
        assert by_type["cp_message"] == list(range(1, n_days + 1))
        assert by_type["dyad_game"] == [1, 2]  # decision_idx = study week (API-Spec §3)
        am = [d for d in dyad.decisions.values() if d.decision_type == "aya_message" and d.slot == "am"]
        assert all(d.decision_idx % 2 == 1 for d in am)

    def test_horizon_counts_and_updates(self):
        simulator, result = _run([MON, MON + dt.timedelta(days=9)], 3)
        counts = result.counts_by_endpoint()
        expected = simulator.expected_decisions()
        assert expected[("dyad_001", "aya_message")] == 42 and expected[("dyad_001", "dyad_game")] == 3
        assert expected[("dyad_002", "aya_message")] == 24 and expected[("dyad_002", "dyad_game")] == 1
        assert counts["action"] == {"201": sum(expected.values())}
        # /update every Monday once a dyad is registered (registration is Sunday 12:00), incl. the final Monday.
        updates = [r.logical_time for r in result.records if r.endpoint == "update"]
        assert updates == [dt.datetime.combine(MON + dt.timedelta(days=7 * k), dt.time(3, 0)) for k in range(4)]
        pulls = [r.logical_time for r in result.records if r.endpoint == "monitor_alarms"]
        assert all(t.time() in (dt.time(5, 50), dt.time(17, 50)) for t in pulls)
        assert len(pulls) == len(set(pulls)) == 1 + 2 * 21 + 1  # registration day 17:50, 21 days, final Monday 05:50
        assert result.calendar_mismatches() == []


class TestMondaySequence:
    def test_monday_and_weekday_call_order(self):
        _, result = _run([MON], 3)
        monday = dt.date(2026, 11, 16)

        def sequence(day):
            return [
                (r.endpoint, r.kind or r.decision_type)
                for r in result.records
                if r.logical_time.date() == day and r.endpoint != "register_group"
            ]

        def label(r):
            return r.kind if r.endpoint == "upload_data" else r.decision_type

        assert [
            (r.endpoint, label(r)) for r in result.records if r.logical_time.date() == monday
        ] == [
            ("upload_data", CLOSING),
            ("update", None),
            ("monitor_alarms", None),
            ("upload_data", PRE_GAME),
            ("action", "dyad_game"),
            ("upload_data", PRE_AYA_AM),
            ("action", "aya_message"),
            ("upload_data", PRE_CP),
            ("action", "cp_message"),
            ("monitor_alarms", None),
            ("upload_data", PRE_AYA_PM),
            ("action", "aya_message"),
        ]
        tuesday = [(r.endpoint, label(r)) for r in result.records if r.logical_time.date() == monday + dt.timedelta(days=1)]
        assert [e for e in tuesday if e[0] == "upload_data"] == [
            ("upload_data", PRE_AYA_AM),
            ("upload_data", PRE_CP),
            ("upload_data", PRE_AYA_PM),
        ]
        assert sequence(monday)[0] == ("upload_data", CLOSING)
        closing = next(r for r in result.records if r.logical_time.date() == monday and r.kind == CLOSING)
        assert dt.time(0, 0) <= closing.logical_time.time() <= dt.time(2, 45)

    def test_every_action_preceded_by_exactly_one_upload_2_to_5_minutes_earlier(self):
        _, result = _run([MON, WED], 3)
        assert not [v for v in result.violations if v["what"] == "upload_before_action"]
        last_upload = {}
        uploads_since_action = Counter()
        for r in result.records:
            if r.endpoint == "upload_data" and r.status == 201:
                last_upload[r.group_id] = r.logical_time
                uploads_since_action[r.group_id] += 1
            elif r.endpoint == "action":
                lead = r.logical_time - last_upload[r.group_id]
                assert dt.timedelta(minutes=2) <= lead < dt.timedelta(minutes=5), (r.to_json(), lead)
                # Monday's dyad_game action also follows that night's week-closing upload.
                allowed = 2 if r.decision_type == "dyad_game" else 1
                assert 1 <= uploads_since_action[r.group_id] <= allowed
                uploads_since_action[r.group_id] = 0


class TestSnapshotFields:
    def test_snapshot_is_full_and_ordered(self):
        _, result = _run([WED], 2)
        for _t, _kind, data in _uploads(result, "dyad_001"):
            assert list(data) == list(SNAPSHOT_FIELDS)
            assert len(data) == 26

    def test_cold_start(self):
        _, result = _run([WED], 1)
        t, kind, data = _uploads(result, "dyad_001")[0]
        assert kind == PRE_AYA_AM and t.date() == WED
        assert data == {
            "day_in_study": 1,
            "week_in_study": 0,
            "slot": "am",
            "aya_diary_mood": MISS,
            "aya_diary_physical": MISS,
            "aya_app_engagement": MISS,
            "aya_app_burden": 0.0,
            "aya_missing_rate_7d": 0.0,
            "previous_med_adherence": MISS,
            "prompted_by_message": False,
            "aya_messages_delivered_week": 0,
            "cp_diary_mood": MISS,
            "cp_app_engagement": MISS,
            "cp_app_burden": 0.0,
            "cp_missing_rate_7d": 0.0,
            "daily_diary_completed": False,
            "cp_prompted_by_message": False,
            "cp_messages_delivered_week": 0,
            "relationship_quality_aya": MISS,
            "relationship_quality_cp": MISS,
            "current_game_on": MISS,
            "prior_game_action": MISS,
            "aya_diary_summary": MISS,
            "cp_diary_summary": MISS,
            "weekly_survey_completed": False,
            "weekly_relationship_score": 0.0,
        }

    def test_burden_recurrence_closed_form_when_every_message_is_delivered(self):
        _, result = _run([MON], 2, action_fn=lambda payload: 1)
        g_a, g_c = GAMMA["aya"], GAMMA["cp"]
        checked = 0
        for t, kind, data in _uploads(result, "dyad_001"):
            n = data["day_in_study"]
            if kind in (PRE_GAME, PRE_AYA_AM, PRE_CP):
                k_aya, k_cp = 2 * (n - 1), n - 1  # deliveries folded in so far
            elif kind == PRE_AYA_PM:
                k_aya, k_cp = 2 * (n - 1) + 1, n
            else:  # CLOSING on day n: every decision of days 1..n-1
                k_aya, k_cp = 2 * (n - 1), n - 1
            assert data["aya_app_burden"] == pytest.approx((1 - g_a**k_aya) / (1 - g_a), abs=1e-12)
            assert data["cp_app_burden"] == pytest.approx((1 - g_c**k_cp) / (1 - g_c), abs=1e-12)
            checked += 1
        assert checked > 40

    def test_burden_stays_zero_without_deliveries(self):
        _, result = _run([MON], 2, action_fn=lambda payload: 0)
        for _t, _kind, data in _uploads(result, "dyad_001"):
            assert data["aya_app_burden"] == 0.0 and data["cp_app_burden"] == 0.0
            assert data["aya_messages_delivered_week"] == 0 and data["cp_messages_delivered_week"] == 0
            assert data["prompted_by_message"] is False and data["cp_prompted_by_message"] is False

    def test_failed_deliveries_do_not_advance_burden(self):
        simulator, result = _run([MON], 2, action_fn=lambda payload: 1, delivery_success=0.5)
        dyad = simulator.dyads["dyad_001"]
        flags = [d.delivered for d in sorted(dyad.decisions.values(), key=lambda d: d.time) if d.decision_type == "aya_message"]
        assert True in flags and False in flags
        b = 0.0
        expected_by_time = []
        for d in sorted((d for d in dyad.decisions.values() if d.decision_type == "aya_message"), key=lambda d: d.time):
            expected_by_time.append((d.time, b))
            b = GAMMA["aya"] * b + (1.0 if d.delivered else 0.0)
        for t, kind, data in _uploads(result, "dyad_001"):
            if kind == PRE_AYA_AM or kind == PRE_AYA_PM:
                decision_time = next(dt_ for dt_, _ in expected_by_time if dt_ > t)
                b_expected = dict(expected_by_time)[decision_time]
                assert data["aya_app_burden"] == pytest.approx(b_expected, abs=1e-12)

    def test_missing_rate_denominators_and_values(self):
        simulator, result = _run([MON], 3, seed=11)
        dyad = simulator.dyads["dyad_001"]

        def day(n):
            return MON + dt.timedelta(days=n - 1)

        def adh_missing(d, slot):
            return dyad.adherence.get((d, slot), MISS) == MISS

        def diary_missing(d):
            return dyad.aya_diary.get(d, (MISS, MISS)) == (MISS, MISS)

        seen_full = False
        for t, kind, data in _uploads(result, "dyad_001"):
            n = data["day_in_study"]
            prev_days = [day(k) for k in range(max(1, n - 7), n)]
            if kind == PRE_AYA_PM:
                am_days = [day(k) for k in range(max(1, n - 6), n + 1)]
            else:
                am_days = prev_days
            e = len(am_days) + 2 * len(prev_days)
            m = sum(adh_missing(d, "am") for d in am_days) + sum(adh_missing(d, "pm") + diary_missing(d) for d in prev_days)
            assert data["aya_missing_rate_7d"] == pytest.approx(m / e if e else 0.0, abs=1e-12), (t, kind)
            cp_days = [day(k) for k in range(max(1, n - 7), n)]
            cp_m = sum(dyad.cp_diary.get(d, MISS) == MISS for d in cp_days)
            assert data["cp_missing_rate_7d"] == pytest.approx(cp_m / len(cp_days) if cp_days else 0.0, abs=1e-12)
            if n >= 8 and kind == PRE_AYA_AM:
                assert e == 21 and len(cp_days) == 7
                seen_full = True
            if n == 2 and kind == PRE_AYA_AM:
                assert e == 3 and len(cp_days) == 1
            if n == 2 and kind == PRE_AYA_PM:
                assert e == 4
        assert seen_full
        rates = {data["aya_missing_rate_7d"] for _, _, data in _uploads(result, "dyad_001")}
        assert len(rates) > 3  # the latent model produces misses

    def test_diary_fields_and_summaries(self):
        simulator, result = _run([MON], 3, seed=3)
        dyad = simulator.dyads["dyad_001"]
        for t, kind, data in _uploads(result, "dyad_001"):
            yesterday = t.date() - dt.timedelta(days=1)
            mood, physical = dyad.aya_diary.get(yesterday, (MISS, MISS)) if yesterday >= MON else (MISS, MISS)
            assert (data["aya_diary_mood"], data["aya_diary_physical"]) == (mood, physical)
            cp_mood = dyad.cp_diary.get(yesterday, MISS) if yesterday >= MON else MISS
            assert data["cp_diary_mood"] == cp_mood
            assert data["daily_diary_completed"] is (cp_mood != MISS)
            window = [yesterday - dt.timedelta(days=k) for k in range(7)]
            moods = [dyad.aya_diary.get(d, (MISS, MISS))[0] for d in window if d >= MON]
            moods = [m for m in moods if m != MISS]
            expected = sum((m - 1) / 4 for m in moods) / len(moods) if moods else MISS
            assert data["aya_diary_summary"] == (pytest.approx(expected, abs=1e-12) if moods else MISS)

    def test_closing_upload_carries_final_week_values(self):
        simulator, result = _run([MON], 3, seed=2)
        dyad = simulator.dyads["dyad_001"]
        uploads = _uploads(result, "dyad_001")
        monday = dt.date(2026, 11, 9)
        closing = [u for u in uploads if u[1] == CLOSING and u[0].date() == monday]
        assert len(closing) == 1
        t, _, data = closing[0]
        week_decisions = [d for d in dyad.decisions.values() if MON <= d.day < monday]
        assert data["aya_messages_delivered_week"] == sum(bool(d.delivered) for d in week_decisions if d.decision_type == "aya_message")
        assert data["cp_messages_delivered_week"] == sum(bool(d.delivered) for d in week_decisions if d.decision_type == "cp_message")
        assert data["aya_messages_delivered_week"] > 0
        assert data["current_game_on"] == MISS
        assert data["prior_game_action"] == dyad.game_by_week[1]  # pre-dyad_game convention of week 2
        assert data["week_in_study"] == 2 and data["slot"] == "am"
        sunday = monday - dt.timedelta(days=1)
        assert data["previous_med_adherence"] == dyad.adherence[(sunday, "pm")]
        assert data["aya_diary_mood"] == dyad.aya_diary[sunday][0]
        survey_time, aya_score, cp_score = max((s for s in dyad.surveys if s[0] < t), key=lambda s: s[0])
        assert survey_time.date() == sunday
        assert data["relationship_quality_aya"] == aya_score  # first survey of a Monday starter: no older value to cache
        assert data["relationship_quality_cp"] == cp_score
        assert data["weekly_survey_completed"] is (aya_score != MISS and cp_score != MISS)
        pre_game = next(u for u in uploads if u[1] == PRE_GAME and u[0].date() == monday)[2]
        assert pre_game["aya_messages_delivered_week"] == 0 and pre_game["cp_messages_delivered_week"] == 0
        assert pre_game["current_game_on"] == MISS
        assert pre_game["relationship_quality_aya"] == data["relationship_quality_aya"]
        pre_am = next(u for u in uploads if u[1] == PRE_AYA_AM and u[0].date() == monday)[2]
        assert pre_am["current_game_on"] == dyad.game_by_week[2]

    def test_closing_upload_only_for_dyads_active_last_week(self):
        _, result = _run([MON, dt.date(2026, 11, 9)], 2)
        closings = [(r.group_id, r.logical_time.date()) for r in result.records if r.kind == CLOSING]
        assert closings == [
            ("dyad_001", dt.date(2026, 11, 9)),
            ("dyad_001", dt.date(2026, 11, 16)),
            ("dyad_002", dt.date(2026, 11, 16)),
        ]

    def test_game_fields_follow_the_weekly_rules(self):
        simulator, result = _run([TUE], 5, action_fn=_game_by_week(lambda week: week % 2))
        dyad = simulator.dyads["dyad_001"]
        assert dyad.game_by_week == {1: 1, 2: 0, 3: 1, 4: 0}
        for t, kind, data in _uploads(result, "dyad_001"):
            week = data["week_in_study"]
            if week == 0:
                expected = (MISS, MISS)
            elif kind in (CLOSING, PRE_GAME):
                expected = (MISS, dyad.game_by_week.get(week - 1, MISS) if week >= 2 else MISS)
            else:
                expected = (week % 2, (week - 1) % 2 if week >= 2 else MISS)
            assert (data["current_game_on"], data["prior_game_action"]) == expected, (t, kind)

    def test_relationship_survey_cached_and_scored(self):
        simulator, result = _run([MON], 5, seed=8)
        dyad = simulator.dyads["dyad_001"]
        assert [s[0].time() for s in dyad.surveys] == [dt.time(21, 0)] * 5
        assert all(s[0].weekday() == 6 for s in dyad.surveys)
        for t, kind, data in _uploads(result, "dyad_001"):
            if data["weekly_survey_completed"]:
                assert data["weekly_relationship_score"] == pytest.approx(
                    0.5 * (data["relationship_quality_aya"] + data["relationship_quality_cp"])
                )
            else:
                assert data["weekly_relationship_score"] == 0.0
            for name in ("relationship_quality_aya", "relationship_quality_cp"):
                value = data[name]
                assert value == MISS or (1 <= value <= 5 and (value * 4) == int(value * 4))
        first_week = [d for t, k, d in _uploads(result, "dyad_001") if d["week_in_study"] == 1]
        assert all(d["relationship_quality_aya"] == MISS for d in first_week)  # no survey yet (Monday start)

    def test_engagement_buckets(self):
        simulator, result = _run([MON], 3, seed=4, analytics_gap_rate=0.0)
        dyad = simulator.dyads["dyad_001"]
        for t, kind, data in _uploads(result, "dyad_001"):
            d = t.date()
            if d == MON:
                assert data["aya_app_engagement"] == MISS
                continue
            opened = [dyad.opens["aya"].get(d - dt.timedelta(days=k), False) for k in (1, 2, 3)]
            if not any(opened):
                expected = 1
            elif not opened[0]:
                expected = 2
            else:
                mood, physical = dyad.aya_diary[d - dt.timedelta(days=1)]
                expected = 4 if (mood, physical) != (MISS, MISS) else 3
            assert data["aya_app_engagement"] == expected

    def test_deterministic_given_seed(self):
        _, first = _run([MON, WED], 3, seed=21)
        _, second = _run([MON, WED], 3, seed=21)
        _, other = _run([MON, WED], 3, seed=22)

        def trace(result):
            return [(r.endpoint, r.logical_time, r.status, json.dumps(r.request, sort_keys=True)) for r in result.records]

        assert trace(first) == trace(second)
        assert trace(first) != trace(other)


class TestValidator:
    def _valid(self):
        simulator, result = _run([WED], 3, seed=9)
        t, kind, data = next(u for u in _uploads(result, "dyad_001") if u[1] == PRE_AYA_PM and u[2]["day_in_study"] == 14)
        assert check_snapshot_rules(data, consent_start=WED, t=t, kind=kind) == []
        return t, kind, data

    @pytest.mark.parametrize(
        "change, fragment",
        [
            ({"aya_diary_mood": 7}, "aya_diary_mood"),
            ({"aya_diary_mood": 3.5}, "aya_diary_mood"),
            ({"cp_app_engagement": 0}, "cp_app_engagement"),
            ({"slot": "am"}, "slot"),
            ({"day_in_study": 3}, "day_in_study"),
            ({"week_in_study": 5}, "week_in_study"),
            ({"current_game_on": MISS}, "current_game_on"),
            ({"prior_game_action": MISS}, "prior_game_action"),
            ({"aya_missing_rate_7d": 0.123}, "aya_missing_rate_7d"),
            ({"cp_missing_rate_7d": 0.5}, "cp_missing_rate_7d"),
            ({"aya_app_burden": 20.0}, "aya_app_burden"),
            ({"aya_messages_delivered_week": 15}, "delivered-week"),
            ({"weekly_survey_completed": True, "relationship_quality_cp": MISS}, "weekly_survey_completed"),
            ({"weekly_relationship_score": 4.0, "weekly_survey_completed": False}, "weekly_relationship_score"),
            ({"daily_diary_completed": True, "cp_diary_mood": MISS}, "daily_diary_completed"),
            ({"prompted_by_message": True, "previous_med_adherence": MISS}, "prompted_by_message"),
            ({"aya_diary_summary": 0.33}, "aya_diary_summary"),
            ({"extra_field": 1}, "unknown field"),
        ],
    )
    def test_static_rules_reject(self, change, fragment):
        t, kind, data = self._valid()
        bad = {**data, **change}
        issues = check_snapshot_rules(bad, consent_start=WED, t=t, kind=kind)
        assert issues and any(fragment in issue for issue in issues), issues

    def test_missing_field_rejected(self):
        t, kind, data = self._valid()
        bad = dict(data)
        del bad["cp_prompted_by_message"]
        assert any("missing field 'cp_prompted_by_message'" in i for i in check_snapshot_rules(bad, consent_start=WED, t=t, kind=kind))

    def test_closing_rules(self):
        t, kind, data = self._valid()
        monday = dt.datetime(2026, 11, 16, 0, 30)
        closing = {**data, "day_in_study": 13, "week_in_study": 2, "slot": "am", "current_game_on": 1}
        issues = check_snapshot_rules(closing, consent_start=WED, t=monday, kind=CLOSING)
        assert any("current_game_on must be 'miss'" in i for i in issues)
        late = check_snapshot_rules({**closing, "current_game_on": MISS}, consent_start=WED, t=monday.replace(hour=3), kind=CLOSING)
        assert any("outside 00:00-02:45" in i for i in late)

    def test_auditor_catches_a_broken_burden_recurrence(self, monkeypatch):
        original = DyadHost.burden_at

        def drifting(self, role, t):
            value = original(self, role, t)
            return value + 0.01 if value > 0 else value

        monkeypatch.setattr(DyadHost, "burden_at", drifting)
        with pytest.raises(SnapshotValidationError, match="recurrence"):
            _run([MON], 1, action_fn=lambda payload: 1)


class TestHostFaults:
    def test_skip_action_uses_host_coin_and_keeps_counters(self):
        key = ("dyad_001", "aya_message", 5)
        simulator, result = _run([MON], 2, faults=HostFaults(skip_actions={key}))
        dyad = simulator.dyads["dyad_001"]
        decision = dyad.decisions[("aya_message", 5)]
        assert decision.source == "host_skip" and decision.action in (0, 1) and decision.action_prob == 0.5
        calls = [r for r in result.records if r.decision_type == "aya_message" and r.group_id == "dyad_001"]
        assert 5 not in {r.decision_idx for r in calls}
        assert {4, 6} <= {r.decision_idx for r in calls}
        # The skipped pair sends no upload either: one upload fewer than the calendar implies.
        n_uploads = sum(1 for r in result.records if r.endpoint == "upload_data")
        n_closing = sum(1 for r in result.records if r.kind == CLOSING)
        assert n_uploads == sum(simulator.expected_decisions().values()) - 1 + n_closing
        assert result.calendar_mismatches() == []
        assert decision.delivered is (decision.action == 1)

    def test_withhold_closing(self):
        monday = dt.date(2026, 11, 16)
        _, result = _run([MON, MON], 3, faults=HostFaults(withhold_closing={("dyad_001", monday)}))
        closings = {(r.group_id, r.logical_time.date()) for r in result.records if r.kind == CLOSING}
        assert ("dyad_001", monday) not in closings
        assert ("dyad_002", monday) in closings and ("dyad_001", dt.date(2026, 11, 9)) in closings

    def test_bad_diary_upload_rejected_then_resent(self):
        key = ("dyad_001", "aya_message", 7)
        _, result = _run([MON], 2, faults=HostFaults(bad_diary_uploads={key}))
        assert result.unexpected == []
        bad = [r for r in result.records if r.endpoint == "upload_data" and r.status == 400]
        assert len(bad) == 1 and bad[0].request["data"]["aya_diary_mood"] == 7 and bad[0].expected
        i = result.records.index(bad[0])
        resent, action = result.records[i + 1], result.records[i + 2]
        assert resent.endpoint == "upload_data" and resent.status == 201 and resent.request["data"]["aya_diary_mood"] != 7
        assert action.endpoint == "action" and action.decision_idx == 7
        assert resent.logical_time - bad[0].logical_time == dt.timedelta(seconds=60)

    def test_skip_delay_and_resend_updates(self):
        w1, w2, w3 = (dt.date(2026, 11, 9), dt.date(2026, 11, 16), dt.date(2026, 11, 23))
        faults = HostFaults(skip_updates={w1}, delay_updates={w2: dt.time(6, 0)}, resend_updates={w3: dt.time(4, 0)})
        _, result = _run([MON], 4, faults=faults)
        updates = [r.logical_time for r in result.records if r.endpoint == "update"]
        assert updates == [
            dt.datetime(2026, 11, 2, 3, 0),
            dt.datetime(2026, 11, 16, 6, 0),
            dt.datetime(2026, 11, 23, 3, 0),
            dt.datetime(2026, 11, 23, 4, 0),
            dt.datetime(2026, 11, 30, 3, 0),
        ]
        delayed = [r for r in result.records if r.logical_time == dt.datetime(2026, 11, 16, 6, 0)]
        assert [r.endpoint for r in delayed] == ["update", "action"]  # update first, then the 06:00 game

    def test_stuck_game_reports_last_weeks_value(self):
        faults = HostFaults(stuck_game={"dyad_001": 2})
        simulator, result = _run([MON], 4, faults=faults, action_fn=_game_by_week(lambda week: week % 2))
        dyad = simulator.dyads["dyad_001"]
        assert dyad.stuck_week == 2
        assert result.violations == []
        for t, kind, data in _uploads(result, "dyad_001"):
            if kind in (CLOSING, PRE_GAME):
                continue
            week = data["week_in_study"]
            expected = dyad.game_by_week[week - 1] if week == 2 else dyad.game_by_week[week]
            assert data["current_game_on"] == expected

    def test_stuck_game_waits_for_a_change(self):
        faults = HostFaults(stuck_game={"dyad_001": 2})
        simulator, _ = _run([MON], 4, faults=faults, action_fn=_game_by_week(lambda week: int(week >= 3)))
        assert simulator.dyads["dyad_001"].stuck_week == 3  # weeks 1-2 were both 0

    def test_tuesday_start(self):
        simulator, result = _run([TUE], 2)
        dyad = simulator.dyads["dyad_001"]
        assert dyad.spec.consent_start.weekday() == 1 and dyad.spec.week1_start == dt.date(2026, 11, 9)
        first = [u for u in _uploads(result, "dyad_001") if u[2]["day_in_study"] <= 6]
        assert {u[2]["week_in_study"] for u in first} == {0}
        assert result.violations == [] and result.unexpected == []


class _Flaky(OfflineTransport):
    """Answers the n-th /action with 500."""

    def __init__(self, fail_at: int, **kwargs):
        super().__init__(**kwargs)
        self.fail_at = fail_at
        self.n_actions = 0

    def request(self, method, path, json_body=None):
        if path.endswith("/action"):
            self.n_actions += 1
            if self.n_actions == self.fail_at:
                return self._envelope(500, "Internal Error", "The decision could not be persisted.", action=None)
        return super().request(method, path, json_body)


class TestStrictness:
    def test_unexpected_status_raises_in_strict_mode(self):
        plan = TrialPlan(enrollment=(MON,), num_weeks=1, seed=1)
        with pytest.raises(UnexpectedStatusError, match="status 500"):
            HostSimulator(_Flaky(3), plan).run()

    def test_non_strict_records_and_falls_back_to_host_coin(self):
        plan = TrialPlan(enrollment=(MON,), num_weeks=1, seed=1, strict=False)
        simulator = HostSimulator(_Flaky(3), plan)
        result = simulator.run()
        assert len(result.unexpected) == 1 and result.unexpected[0].status == 500
        sources = Counter(d.source for d in result.decisions)
        assert sources["host_fallback"] == 1
        assert result.calendar_mismatches() == []
        assert result.legacy_counts()["errors"][0]["status"] == 500

    def test_allow_list_accepts_listed_status(self):
        plan = TrialPlan(enrollment=(MON,), num_weeks=1, seed=1)
        result = HostSimulator(_Flaky(3), plan, HostFaults(allowed_statuses={("action", 500)})).run()
        assert result.unexpected == []

    def test_envelope_violation_is_reported(self):
        class NoRid(OfflineTransport):
            def request(self, method, path, json_body=None):
                status, body = super().request(method, path, json_body)
                if path.endswith("/register_group"):
                    body = {k: v for k, v in body.items() if k != "rid"}
                return status, body

        plan = TrialPlan(enrollment=(MON,), num_weeks=1, seed=1, strict=False)
        result = HostSimulator(NoRid(), plan).run()
        assert any(v["what"] == "envelope" and v.get("endpoint") == "register_group" for v in result.violations)


class TestHooksAndTransports:
    def test_ticks_on_a_fixed_logical_grid(self):
        ticks = []
        hooks = SimulatorHooks(tick_every=dt.timedelta(minutes=15), on_tick=ticks.append)
        simulator, result = _run([MON], 1, hooks=hooks)
        steps = {b - a for a, b in zip(ticks, ticks[1:])}
        assert steps == {dt.timedelta(minutes=15)}
        assert ticks[0] >= simulator.dyads["dyad_001"].spec.registration_time
        assert ticks[-1] == simulator.plan.end_time
        assert all(t.minute % 15 == 0 and t.second == 0 for t in ticks)

    def test_wait_for_update_and_timed_hooks(self):
        waited, fired = [], []
        hooks = SimulatorHooks(
            wait_for_update=lambda rid, when: waited.append((rid, when)),
            timed=[(dt.datetime(2026, 11, 4, 12, 0), fired.append)],
        )
        _, result = _run([MON], 1, hooks=hooks)
        assert waited == [(r.rid, r.logical_time) for r in result.records if r.endpoint == "update"]
        assert fired == [dt.datetime(2026, 11, 4, 12, 0)]

    def test_tz_aware_timestamps(self):
        _, result = _run([MON], 1, tz_aware_timestamps=True)
        stamp = next(r.request["timestamp"] for r in result.records if r.endpoint == "action")
        assert stamp.endswith("-05:00")  # EST in November
        assert dt.datetime.fromisoformat(stamp).replace(tzinfo=None) == dt.datetime(2026, 11, 2, 6, 0)

    def test_http_transport_maps_status_and_json(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                payload = json.dumps({"status": 201, "echo": body}).encode()
                self.send_response(201)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):  # noqa: N802
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            transport = HttpTransport(f"http://127.0.0.1:{server.server_address[1]}", timeout=5)
            assert transport.request("POST", "/api/v1/action", {"a": 1}) == (201, {"status": 201, "echo": {"a": 1}})
            assert transport.request("GET", "/api/v1/monitor/alarms") == (404, None)
        finally:
            server.shutdown()
            server.server_close()
        with pytest.raises(TransportError):
            HttpTransport(f"http://127.0.0.1:{server.server_address[1]}", timeout=2).request("GET", "/x")

    def test_legacy_run_simulation_wrapper(self):
        offline = OfflineTransport(seed=3)

        class FakeResponse:
            def __init__(self, status, body):
                self.status_code, self._body = status, body

            def get_json(self, silent=False):
                return self._body

        class FakeClient:
            def open(self, path, method="GET", json=None):
                return FakeResponse(*offline.request(method, path, json))

        out = run_simulation(FakeClient(), base_date=MON, num_weeks=2, num_dyads=2)
        assert out["add_group"] == 2 and out["errors"] == []
        assert out["action"] == 2 * 7 * 3 + 2 + 6 * 3  # dyad 1: 14 days + 2 games; dyad 2 (Tuesday): 6 partial days
        assert out["update"] == 3 and out["upload_data"] > out["action"]


class TestFaultScenarios:
    """The Monitoring-Spec §11.3 table resolves against the default live plan and its host faults behave."""

    @pytest.fixture
    def plan(self):
        return TrialPlan(num_dyads=10, num_weeks=16, seed=42, strict=True)

    def test_learning_starts_before_every_fault(self, plan):
        from tools.fault_scenarios import SCENARIOS

        starts = sorted(week1_start(d) for d in plan.consent_starts())
        fifth = starts[4]
        first_learning_week = plan.calendar_week(fifth) + 1  # |E_{W-1}| >= 5 (contract D2)
        assert first_learning_week == 6
        for scenario in SCENARIOS.values():
            whens = [h.when for h in scenario.host if h.when] + [a.when for a in scenario.server_actions]
            if scenario.server_env_window:
                whens.append(scenario.server_env_window.start)
            if scenario.db_tamper:
                whens.append(scenario.db_tamper.after)
            if scenario.monitor_pause:
                whens.append(scenario.monitor_pause.start)
            assert all(w.week >= first_learning_week + 2 for w in whens), scenario.name
            assert scenario.min_weeks <= plan.num_weeks
            # The run must last until the fault's follow-up checks (C5 Tuesday, C1 two weeks) have run.
            if whens or scenario.extra_dyads:
                last = max([w.week for w in whens] + [d.when.week for d in scenario.extra_dyads])
                assert scenario.min_weeks >= last + 2, scenario.name
        # The clean baseline places nothing, so a short smoke run (e.g. 8 weeks) is allowed.
        assert SCENARIOS["baseline"].min_weeks <= 8

    def test_every_scenario_resolves_and_runs_host_side(self, plan):
        from tools.fault_scenarios import SCENARIOS, plan_with_extra_dyads, resolve_host_faults

        for scenario in SCENARIOS.values():
            run_plan = plan_with_extra_dyads(plan, scenario)
            faults = resolve_host_faults(scenario, run_plan)
            simulator = HostSimulator(OfflineTransport(seed=1), run_plan, faults)
            for gid, decision_type, idx in faults.skip_actions | faults.bad_diary_uploads:
                decision = simulator.dyads[gid].decisions[(decision_type, idx)]
                spec = next(h for h in scenario.host if h.kind in ("skip_action", "bad_diary"))
                assert decision.time == spec.when.resolve(run_plan)
            if not scenario.host and not scenario.extra_dyads:
                continue
            result = simulator.run()
            assert result.unexpected == [] and result.violations == [], scenario.name
            assert result.calendar_mismatches() == [], scenario.name

    def test_extra_dyad_scenario_adds_a_tuesday_start(self, plan):
        from tools.fault_scenarios import SCENARIOS, plan_with_extra_dyads

        run_plan = plan_with_extra_dyads(plan, SCENARIOS["tuesday_start"])
        assert len(run_plan.consent_starts()) == 11
        assert run_plan.consent_starts()[-1].weekday() == 1

    def test_judge_alarms(self):
        from tools.fault_scenarios import SCENARIOS, judge_alarms

        raised = {"C1": {"yellow": 2}, "A6": {"yellow": 1}, "G0": {"green": 5}, "B11": {"yellow": 1}, "L2": {"green": 3}}
        assert judge_alarms(SCENARIOS["baseline"], raised) == {"missing": [], "unexplained": ["B11"], "chance": ["A6"]}
        assert judge_alarms(SCENARIOS["withhold_closing"], raised) == {"missing": [], "unexplained": [], "chance": ["A6"]}
        assert judge_alarms(SCENARIOS["skip_update"], raised)["missing"] == ["C11", "C7"]


# =============================================================== part (b)

E2E_START = dt.date(2026, 11, 2)  # Monday of calendar week 0
E2E_ENROLLMENT = (
    E2E_START,  # dyad_001 Monday
    E2E_START,  # dyad_002 Monday
    E2E_START + dt.timedelta(days=2),  # dyad_003 Wednesday (partial week)
    E2E_START,  # dyad_004 Monday
    E2E_START,  # dyad_005 Monday
    E2E_START + dt.timedelta(days=11),  # dyad_006 Friday of week 1 (partial week)
)
E2E_SKIP = ("dyad_001", "cp_message", 10)
E2E_BAD_DIARY = ("dyad_003", "aya_message", 9)


@pytest.fixture(scope="module")
def e2e():
    """One in-process 4-week run shared by the part (b) tests."""
    import config
    from app import create_app, db

    cfg = type("SimulationTestConfig", (config.TestingConfig,), {})
    app = create_app(cfg)
    ctx = app.app_context()
    ctx.push()
    try:
        plan = TrialPlan(enrollment=E2E_ENROLLMENT, num_weeks=4, seed=17, strict=False, record_payloads=True)
        faults = HostFaults(skip_actions={E2E_SKIP}, bad_diary_uploads={E2E_BAD_DIARY})
        simulator = HostSimulator(FlaskClientTransport(app.test_client()), plan, faults)
        result = simulator.run()
        yield app, simulator, result
    finally:
        db.session.remove()
        db.drop_all()
        ctx.pop()


class TestEndToEndInProcess:
    def test_every_response_is_2xx_except_injected_faults(self, e2e):
        _, _, result = e2e
        assert [r.to_json() for r in result.unexpected] == []
        non_2xx = [r for r in result.records if not (200 <= (r.status or 0) < 300)]
        assert [(r.endpoint, r.status, r.group_id, r.note) for r in non_2xx] == [
            ("upload_data", 400, "dyad_003", "injected diary item 7")
        ]
        assert non_2xx[0].message == "aya_diary_mood must be between 1 and 5; got 7."

    def test_no_host_side_contract_violations(self, e2e):
        _, _, result = e2e
        assert result.violations == []

    def test_every_action_has_a_fresh_preceding_upload(self, e2e):
        _, _, result = e2e
        last_upload: dict[str, dt.datetime] = {}
        n_actions = 0
        for record in result.records:
            if record.endpoint == "upload_data" and record.status == 201:
                last_upload[record.group_id] = record.logical_time
            elif record.endpoint == "action":
                lead = record.logical_time - last_upload[record.group_id]
                assert dt.timedelta(minutes=2) <= lead < dt.timedelta(minutes=5)
                n_actions += 1
        assert n_actions == sum(1 for d in result.decisions if d.source == "api")

    def test_decision_counts_match_the_calendar(self, e2e):
        app, simulator, result = e2e
        from app.models import Action

        expected = {
            # Monday starts: 28 days x (2 AYA + 1 CP) + 4 Monday games.
            **{(g, "aya_message"): 56 for g in ("dyad_001", "dyad_002", "dyad_004", "dyad_005")},
            **{(g, "cp_message"): 28 for g in ("dyad_001", "dyad_002", "dyad_004", "dyad_005")},
            **{(g, "dyad_game"): 4 for g in ("dyad_001", "dyad_002", "dyad_004", "dyad_005")},
            # Wednesday start: 26 days, games from week 1 (3 Mondays).
            ("dyad_003", "aya_message"): 52,
            ("dyad_003", "cp_message"): 26,
            ("dyad_003", "dyad_game"): 3,
            # Friday-of-week-1 start: 17 days, 2 Mondays.
            ("dyad_006", "aya_message"): 34,
            ("dyad_006", "cp_message"): 17,
            ("dyad_006", "dyad_game"): 2,
        }
        assert simulator.expected_decisions() == expected
        assert result.calendar_mismatches() == []
        api = Counter((d.group_id, d.decision_type) for d in result.decisions if d.source == "api")
        stored = Counter((a.group_id, a.decision_type) for a in Action.query.all())
        assert api == stored
        assert stored[("dyad_001", "cp_message")] == 27  # one CP pair skipped by the host (F-A1)
        assert sum(stored.values()) == sum(expected.values()) - 1

    def test_updates_and_alarm_pulls(self, e2e):
        _, _, result = e2e
        updates = [r for r in result.records if r.endpoint == "update"]
        assert [r.logical_time for r in updates] == [
            dt.datetime.combine(E2E_START + dt.timedelta(days=7 * k), dt.time(3, 0)) for k in range(5)
        ]
        assert {r.status for r in updates} == {202}
        pulls = [r for r in result.records if r.endpoint == "monitor_alarms"]
        assert len(pulls) == 1 + 2 * 28 + 1 and {r.status for r in pulls} == {200}
        for r in pulls:
            assert isinstance(r.response.get("alarms"), list) and r.response.get("count") == len(r.response["alarms"])

    def test_warmup_rules(self, e2e):
        _, simulator, result = e2e
        for d in result.decisions:
            if d.source != "api":
                continue
            if d.study_week <= 1 or d.calendar_week <= 1:
                # Partial week / week 1 (D3) and cohort warm-up: fewer than 5 dyads completed week 1 before week 2 (D2).
                assert d.warmup is True and d.action_prob == 0.5, (d.group_id, d.decision_type, d.decision_idx)
            assert d.action_prob is not None and 0.2 - 1e-12 <= d.action_prob <= 0.8 + 1e-12

    def test_learning_engages_after_the_cohort_warm_up(self, e2e):
        _, _, result = e2e
        learned = Counter(
            d.decision_type for d in result.decisions if d.source == "api" and d.warmup is False and d.calendar_week >= 2
        )
        assert set(learned) == {"aya_message", "cp_message", "dyad_game"}, learned
        for d in result.decisions:
            if d.source == "api" and d.warmup is False:
                assert d.calendar_week >= 2 and d.study_week >= 2
                assert d.model_param_len and d.state_len

    def test_alarm_pulls_carry_the_injected_upload_error_and_no_red_alarm(self, e2e):
        _, _, result = e2e
        by_item = result.alarms_by_item()
        # The rejected diary upload is logged after the response (Monitoring-Spec §6.4, UD-400-RANGE) and collected by a pull.
        assert by_item.get("UD-400-RANGE") == {"yellow": 1}
        red = {item: sev for item, sev in by_item.items() if sev.get("red")}
        assert red == {}
        for alarm in result.alarms.values():
            assert set(alarm) >= {"event_id", "item_id", "severity", "owner", "raised_at", "message", "detail"}
            assert alarm["owner"] in ("host", "rl") and alarm["severity"] in ("red", "yellow", "green")
            assert dt.datetime.fromisoformat(alarm["raised_at"]).utcoffset() is not None  # Detroit offset (API-Spec §2.5)

    def test_server_state_matches_the_host_log(self, e2e):
        app, _, result = e2e
        from app.models import DataUpload, UploadQuarantine

        accepted = sum(1 for r in result.records if r.endpoint == "upload_data" and r.status == 201)
        assert DataUpload.query.count() == accepted
        quarantined = UploadQuarantine.query.all()
        assert len(quarantined) == 1 and json.loads(quarantined[0].raw_body.decode())["data"]["aya_diary_mood"] == 7
