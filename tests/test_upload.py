"""
``POST /api/v1/upload_data`` (API-Spec §2.3, §4.1; Monitoring-Spec §6.4,
§6.4.1, §7 F-U1; contract D4, D18): every 400 kind and message, quarantine
content, 404 / 503 / 500 answers, the envelope-only response shape and the
Detroit-local timestamp handling.
"""

from __future__ import annotations

import datetime
import hashlib
import logging

import pytest
from sqlalchemy.exc import OperationalError

from app.extensions import db
from app.models import DataUpload, UploadQuarantine
from tests.conftest import (
    ENVELOPE_KEYS,
    assert_envelope,
    full_snapshot,
    raw_json,
    register_group,
    snapshot_for,
    upload,
)

GID = "dyad_up"
TS = "2026-01-07T07:25:00"  # Wednesday of week 1 for a 2026-01-05 consent: day 3, am


def _body(**data_overrides):
    return {"group_id": GID, "timestamp": TS, "data": snapshot_for("2026-01-05", TS, **data_overrides)}


def _post(client, body):
    return client.post("/api/v1/upload_data", json=body)


def _post_raw(client, raw: bytes):
    return client.post("/api/v1/upload_data", data=raw, content_type="application/json")


@pytest.fixture
def registered(client):
    assert register_group(client, GID).status_code == 201
    return client


# ------------------------------------------------------------------ success


def test_valid_snapshot_is_accepted_and_stored_verbatim(registered, app):
    body = _body(aya_diary_mood=4, relationship_quality_cp=3.5, aya_diary_summary=0.55)
    response = _post(registered, body)
    out = assert_envelope(response, 201, "Success", "Snapshot accepted.")
    assert set(out) == set(ENVELOPE_KEYS), "upload_data answers with the envelope only"
    row = DataUpload.query.filter_by(rid=out["rid"]).one()
    assert row.group_id == GID
    assert row.data == body["data"]
    assert row.request_timestamp == datetime.datetime(2026, 1, 7, 7, 25)
    assert UploadQuarantine.query.count() == 0


def test_json_null_is_miss_where_miss_is_allowed(registered):
    body = _body(aya_diary_mood=None, previous_med_adherence=None, relationship_quality_aya=None,
                 current_game_on=None, cp_diary_summary=None, aya_app_engagement=None)
    assert_envelope(_post(registered, body), 201, "Success", "Snapshot accepted.")


def test_aware_timestamp_is_converted_to_detroit_local(registered):
    # 12:25Z is 07:25 EST: still the "am" upload of day 3.
    body = {"group_id": GID, "timestamp": "2026-01-07T12:25:00Z", "data": snapshot_for("2026-01-05", TS)}
    out = assert_envelope(_post(registered, body), 201)
    row = DataUpload.query.filter_by(rid=out["rid"]).one()
    assert row.request_timestamp == datetime.datetime(2026, 1, 7, 7, 25)


def test_partial_week_upload_is_week_zero(client):
    register_group(client, "dyad_thu", start="2026-01-08")  # Thursday; week 1 starts 2026-01-12
    assert upload(client, "dyad_thu", "2026-01-08T08:00:00").status_code == 201
    r = upload(client, "dyad_thu", "2026-01-08T19:00:00")
    assert r.status_code == 201
    r = upload(client, "dyad_thu", "2026-01-12T01:00:00")  # week-closing upload: Monday of week 1
    assert r.status_code == 201


# ----------------------------------------------------------- 400: PARAM


@pytest.mark.parametrize(
    "body, message",
    [
        (None, "group_id is required."),
        ({"timestamp": TS, "data": {}}, "group_id is required."),
        ({"group_id": 5, "timestamp": TS, "data": {}}, "group_id must be a string."),
        ({"group_id": GID, "data": {}}, "timestamp is required."),
        ({"group_id": GID, "timestamp": 1736252700, "data": {}}, "timestamp must be a string or datetime object."),
        ({"group_id": GID, "timestamp": "yesterday", "data": {}},
         "timestamp must be an ISO-8601 timestamp; got 'yesterday'."),
        ({"group_id": GID, "timestamp": TS}, "data is required."),
        ({"group_id": GID, "timestamp": TS, "data": [1, 2]}, "data must be a dictionary."),
    ],
)
def test_param_errors(registered, body, message):
    response = registered.post("/api/v1/upload_data", json=body) if body is not None else \
        registered.post("/api/v1/upload_data", data=b"", content_type="application/json")
    out = assert_envelope(response, 400, "Invalid Parameter", message)
    assert set(out) == set(ENVELOPE_KEYS)


def test_non_json_body_is_param_error(registered):
    response = registered.post("/api/v1/upload_data", data=b"not json at all", content_type="text/plain")
    assert_envelope(response, 400, "Invalid Parameter", "group_id is required.")


# ------------------------------------------------- 400: UNKNOWN / MISSINGFIELD


def test_unknown_field_is_rejected(registered):
    body = _body()
    body["data"]["daily_diary_score"] = 3.0  # removed from the contract
    assert_envelope(_post(registered, body), 400, "Invalid Parameter", "unknown field 'daily_diary_score'.")


def test_missing_field_is_rejected(registered):
    body = _body()
    del body["data"]["aya_messages_delivered_week"]
    assert_envelope(
        _post(registered, body), 400, "Invalid Parameter",
        "aya_messages_delivered_week is required (full snapshot).",
    )


def test_unknown_is_reported_before_missing(registered):
    body = _body()
    del body["data"]["slot"]
    body["data"]["sloot"] = "am"
    assert_envelope(_post(registered, body), 400, "Invalid Parameter", "unknown field 'sloot'.")


# --------------------------------------------------------- 400: TYPE


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("aya_diary_mood", "n/a", "aya_diary_mood: must be an integer from 1 through 5, or 'miss'."),
        ("aya_diary_mood", 3.5, "aya_diary_mood: must be an integer from 1 through 5, or 'miss'."),
        ("aya_diary_mood", True, "aya_diary_mood: must be an integer from 1 through 5, or 'miss'."),
        ("cp_diary_mood", "4", "cp_diary_mood: must be an integer from 1 through 5, or 'miss'."),
        ("day_in_study", True, "day_in_study: must be a positive integer."),
        ("day_in_study", 0, "day_in_study: must be a positive integer."),
        ("day_in_study", 3.0, "day_in_study: must be a positive integer."),
        ("week_in_study", -1, "week_in_study: must be a non-negative integer."),
        ("aya_messages_delivered_week", 1.5, "aya_messages_delivered_week: must be a non-negative integer."),
        ("slot", "noon", "slot: must be 'am' or 'pm'."),
        ("aya_app_burden", None, "aya_app_burden: must be a non-negative number."),
        ("aya_app_burden", -0.5, "aya_app_burden: must be a non-negative number."),
        ("aya_app_burden", "miss", "aya_app_burden: must be a non-negative number."),
        ("weekly_relationship_score", False, "weekly_relationship_score: must be a non-negative number."),
        ("aya_missing_rate_7d", "miss", "aya_missing_rate_7d: must be a number between 0 and 1."),
        ("previous_med_adherence", True, "previous_med_adherence: must be 0, 1, or 'miss'."),
        ("previous_med_adherence", 0.5, "previous_med_adherence: must be 0, 1, or 'miss'."),
        ("prompted_by_message", 1, "prompted_by_message: must be true or false."),
        ("daily_diary_completed", None, "daily_diary_completed: must be true or false."),
        ("aya_app_engagement", 2.5, "aya_app_engagement: must be one of 1, 2, 3, 4, or 'miss'."),
        ("relationship_quality_aya", "high", "relationship_quality_aya: must be a number from 1 through 5, or 'miss'."),
        ("aya_diary_summary", "x", "aya_diary_summary: must be a number between 0 and 1, or 'miss'."),
    ],
)
def test_type_errors(registered, field, value, message):
    assert_envelope(_post(registered, _body(**{field: value})), 400, "Invalid Parameter", message)


def test_non_finite_number_is_type_error(registered):
    raw = raw_json(_body()).replace(b'"aya_missing_rate_7d":0.0', b'"aya_missing_rate_7d":NaN')
    assert b"NaN" in raw
    assert_envelope(_post_raw(registered, raw), 400, "Invalid Parameter",
                    "aya_missing_rate_7d: must be a number between 0 and 1.")


# -------------------------------------------------------- 400: RANGE


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("aya_diary_mood", 7, "aya_diary_mood must be between 1 and 5; got 7."),
        ("aya_diary_physical", 0, "aya_diary_physical must be between 1 and 5; got 0."),
        ("cp_diary_mood", 6, "cp_diary_mood must be between 1 and 5; got 6."),
        ("relationship_quality_cp", 5.5, "relationship_quality_cp must be between 1 and 5; got 5.5."),
        ("relationship_quality_aya", 0.5, "relationship_quality_aya must be between 1 and 5; got 0.5."),
        ("aya_missing_rate_7d", 1.2, "aya_missing_rate_7d must be between 0 and 1; got 1.2."),
        ("cp_missing_rate_7d", -0.1, "cp_missing_rate_7d must be between 0 and 1; got -0.1."),
        ("cp_diary_summary", 2, "cp_diary_summary must be between 0 and 1; got 2."),
        ("cp_app_engagement", 5, "cp_app_engagement must be between 1 and 4; got 5."),
        ("current_game_on", 2, "current_game_on must be between 0 and 1; got 2."),
        ("prior_game_action", -1, "prior_game_action must be between 0 and 1; got -1."),
    ],
)
def test_range_errors(registered, field, value, message):
    assert_envelope(_post(registered, _body(**{field: value})), 400, "Invalid Parameter", message)


def test_first_wrong_field_in_schema_order_is_reported(registered):
    body = _body(aya_diary_mood=7, cp_diary_mood="bad")
    assert_envelope(_post(registered, body), 400, "Invalid Parameter",
                    "aya_diary_mood must be between 1 and 5; got 7.")


# ---------------------------------------------------- 400: INCONSISTENT


def test_day_in_study_must_match_consent_date_and_timestamp(registered):
    assert_envelope(
        _post(registered, _body(day_in_study=4)), 400, "Invalid Parameter",
        "day_in_study does not match timestamp: expected 3, got 4.",
    )


def test_week_in_study_must_match_monday_weeks(registered):
    assert_envelope(
        _post(registered, _body(week_in_study=0)), 400, "Invalid Parameter",
        "week_in_study does not match day_in_study: expected 1, got 0.",
    )


def test_week_in_study_uses_the_api_spec_monday_formula(client):
    # Thursday consent: day 5 (Monday 2026-01-12) is week 1, although
    # floor((5 - 1) / 7) + 1 would also give 1; day 8 (Thursday 2026-01-15) is
    # still week 1, where the outdated Monitoring-Spec formula gives 2.
    register_group(client, "dyad_thu", start="2026-01-08")
    assert upload(client, "dyad_thu", "2026-01-15T08:00:00").status_code == 201
    r = upload(client, "dyad_thu", "2026-01-15T08:05:00", week_in_study=2)
    assert_envelope(r, 400, "Invalid Parameter", "week_in_study does not match day_in_study: expected 1, got 2.")


def test_slot_must_match_local_time_of_day(registered):
    assert_envelope(
        _post(registered, _body(slot="pm")), 400, "Invalid Parameter",
        "slot does not match timestamp: expected 'am', got 'pm'.",
    )
    body = {"group_id": GID, "timestamp": "2026-01-07T19:25:00",
            "data": snapshot_for("2026-01-05", "2026-01-07T19:25:00", slot="am")}
    assert_envelope(_post(registered, body), 400, "Invalid Parameter",
                    "slot does not match timestamp: expected 'pm', got 'am'.")


def test_slot_boundary_follows_upload_slot_noon_hour(make_app):
    app = make_app(UPLOAD_SLOT_NOON_HOUR=14)
    client = app.test_client()
    register_group(client, GID)
    body = {"group_id": GID, "timestamp": "2026-01-07T13:00:00",
            "data": snapshot_for("2026-01-05", "2026-01-07T13:00:00", noon_hour=14)}
    assert body["data"]["slot"] == "am"
    assert_envelope(_post(client, body), 201)


# ------------------------------------------------------- F-U1 quarantine


def test_rejected_body_is_quarantined_verbatim(registered):
    body = _body(aya_diary_mood=7)
    raw = raw_json(body)
    out = assert_envelope(_post_raw(registered, raw), 400)
    row = UploadQuarantine.query.one()
    assert row.rid == out["rid"]
    assert row.group_id == GID
    assert row.raw_body == raw
    assert row.validation_error == "aya_diary_mood must be between 1 and 5; got 7."
    assert row.payload_sha256 == hashlib.sha256(raw).hexdigest()
    assert row.payload_bytes == len(raw)
    assert isinstance(row.received_at, datetime.datetime)
    assert DataUpload.query.count() == 0


def test_every_400_kind_is_quarantined(registered):
    bodies = [
        b"",  # PARAM
        raw_json({"group_id": GID, "timestamp": TS, "data": {**full_snapshot(), "x": 1}}),  # UNKNOWN
        raw_json({"group_id": GID, "timestamp": TS, "data": {k: v for k, v in full_snapshot().items() if k != "slot"}}),
        raw_json(_body(slot=3)),  # TYPE
        raw_json(_body(aya_diary_mood=9)),  # RANGE
        raw_json(_body(day_in_study=9)),  # INCONSISTENT
    ]
    rids = []
    for raw in bodies:
        rids.append(assert_envelope(_post_raw(registered, raw), 400)["rid"])
    rows = UploadQuarantine.query.order_by(UploadQuarantine.id).all()
    assert [r.rid for r in rows] == rids
    assert [r.raw_body for r in rows] == bodies
    assert rows[0].group_id is None


def test_unreadable_group_id_is_quarantined_without_it(registered):
    raw = raw_json({"group_id": 12, "timestamp": TS, "data": full_snapshot()})
    assert_envelope(_post_raw(registered, raw), 400, "Invalid Parameter", "group_id must be a string.")
    row = UploadQuarantine.query.one()
    assert row.group_id is None
    assert row.raw_body == raw


def test_non_utf8_rejected_body_is_quarantined_byte_for_byte(registered):
    raw = b'{"group_id":"dyad_up","data":"\xff\xfe"}'
    assert_envelope(_post_raw(registered, raw), 400)
    row = UploadQuarantine.query.one()
    assert row.raw_body == raw
    assert row.payload_sha256 == hashlib.sha256(raw).hexdigest()
    assert row.payload_bytes == len(raw)


def test_rejected_body_never_reaches_the_logs(registered, caplog):
    marker = "PHI-MARKER-93d1"
    body = _body(aya_diary_mood=7)
    body["data"]["x_" + marker] = 1  # unknown field carrying the marker
    caplog.set_level(logging.DEBUG)
    assert_envelope(_post(registered, body), 400)
    assert UploadQuarantine.query.count() == 1
    assert marker.encode() in UploadQuarantine.query.one().raw_body
    assert marker not in caplog.text


def test_quarantine_failure_does_not_change_the_answer(registered, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("quarantine store down")

    monkeypatch.setattr("app.routes.data.UploadQuarantine", broken)
    assert_envelope(_post(registered, _body(aya_diary_mood=7)), 400, "Invalid Parameter",
                    "aya_diary_mood must be between 1 and 5; got 7.")


def test_accepted_and_404_uploads_are_not_quarantined(registered):
    assert_envelope(_post(registered, _body()), 201)
    body = _body()
    body["group_id"] = "dyad_nope"
    assert_envelope(_post(registered, body), 404)
    assert UploadQuarantine.query.count() == 0


# ------------------------------------------------------- 404 / 503 / 500


def test_unknown_group_is_404(client):
    body = _body()
    body["group_id"] = "dyad_999"
    out = assert_envelope(_post(client, body), 404, "Unknown Group", "group_id dyad_999 is not registered.")
    assert set(out) == set(ENVELOPE_KEYS)


def test_database_write_failure_is_503(registered, monkeypatch):
    def fail_commit():
        raise OperationalError("INSERT INTO data_uploads", {}, Exception("disk I/O error"))

    monkeypatch.setattr(db.session, "commit", fail_commit)
    assert_envelope(_post(registered, _body()), 503, "Service Unavailable", "Database write failed; retry.")
    monkeypatch.undo()
    assert DataUpload.query.count() == 0


def test_database_read_failure_is_503(registered, monkeypatch):
    class BrokenGroup:
        class query:  # noqa: N801 - mimics the Flask-SQLAlchemy attribute
            @staticmethod
            def filter_by(**kwargs):
                raise OperationalError("SELECT", {}, Exception("database is locked"))

    monkeypatch.setattr("app.routes.data.Group", BrokenGroup)
    assert_envelope(_post(registered, _body()), 503, "Service Unavailable", "Database write failed; retry.")


def test_unexpected_failure_is_500(registered, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr("app.routes.data.check_snapshot_consistency", boom)
    out = assert_envelope(_post(registered, _body()), 500, "Internal Error", "Internal server error.")
    assert set(out) == set(ENVELOPE_KEYS)
