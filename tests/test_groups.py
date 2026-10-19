"""
``POST /api/v1/register_group`` (alias ``/add_group``) per API-Spec §2.1 and
Monitoring-Spec §6.2, §9.6: both 201 messages, 409 Member Mismatch, the 400
messages with the offending field echoed as null, 503 / 500, the persisted
``rid`` and one ``group_history`` row per successful call.
"""

from __future__ import annotations

import pytest
from sqlalchemy.exc import OperationalError

from app.extensions import db
from app.models import Group, GroupHistory
from app.routes.group import check_fields
from tests.conftest import assert_envelope

GID = "dyad_007"
BODY = {
    "group_id": GID,
    "member_list": ["cp_007", "aya_007"],
    "consent_start_date": "2026-05-25",
    "consent_end_date": "2026-09-02",
}
ECHO_KEYS = ("group_id", "member_list", "consent_start_date", "consent_end_date")


def _register(client, **overrides):
    return client.post("/api/v1/register_group", json={**BODY, **overrides})


# ----------------------------------------------------------- check_fields


def test_check_fields_valid():
    assert check_fields(dict(BODY)) == (True, "", None)


@pytest.mark.parametrize(
    "body, message, field",
    [
        (None, "Request body is required.", None),
        ([1, 2], "Request body is required.", None),
        ({k: v for k, v in BODY.items() if k != "group_id"}, "group_id is required.", "group_id"),
        ({k: v for k, v in BODY.items() if k != "member_list"}, "member_list is required.", "member_list"),
        ({k: v for k, v in BODY.items() if k != "consent_start_date"}, "consent_start_date is required.",
         "consent_start_date"),
        ({k: v for k, v in BODY.items() if k != "consent_end_date"}, "consent_end_date is required.",
         "consent_end_date"),
        ({**BODY, "consent_start_date": "2026-05-32"},
         "consent_start_date must be a date in YYYY-MM-DD format; got '2026-05-32'.", "consent_start_date"),
        ({**BODY, "consent_end_date": "09/02/2026"},
         "consent_end_date must be a date in YYYY-MM-DD format; got '09/02/2026'.", "consent_end_date"),
        ({**BODY, "consent_start_date": "20260525"},
         "consent_start_date must be a date in YYYY-MM-DD format; got '20260525'.", "consent_start_date"),
        ({**BODY, "consent_start_date": 20260525},
         "consent_start_date must be a date in YYYY-MM-DD format; got 20260525.", "consent_start_date"),
    ],
)
def test_check_fields_errors(body, message, field):
    assert check_fields(body) == (False, message, field)


# ------------------------------------------------------------------ 201


def test_new_registration(client, app):
    out = assert_envelope(_register(client), 201, "Success", "Group registered successfully.")
    for key in ECHO_KEYS:
        assert out[key] == BODY[key]
    group = Group.query.filter_by(group_id=GID).one()
    assert group.rid == out["rid"]
    assert group.group_info == {k: BODY[k] for k in ECHO_KEYS[1:]}
    history = GroupHistory.query.all()
    assert len(history) == 1
    assert history[0].rid == out["rid"]
    assert history[0].member_list == BODY["member_list"]
    assert history[0].consent_start_date == "2026-05-25"
    assert history[0].consent_end_date == "2026-09-02"


def test_reregistration_updates_consent_window_and_appends_history(client, app):
    first = assert_envelope(_register(client), 201)
    # Same members in another order: a match; the recorded members stand.
    second = assert_envelope(
        _register(client, member_list=["aya_007", "cp_007"], consent_start_date="2026-06-01",
                  consent_end_date="2026-09-10"),
        201, "Success", "Group consent window updated.",
    )
    assert second["member_list"] == BODY["member_list"]
    assert second["consent_start_date"] == "2026-06-01"
    assert second["consent_end_date"] == "2026-09-10"

    group = Group.query.filter_by(group_id=GID).one()
    assert group.group_info["consent_start_date"] == "2026-06-01"
    assert group.group_info["consent_end_date"] == "2026-09-10"
    assert group.group_info["member_list"] == BODY["member_list"]
    assert group.rid == second["rid"] != first["rid"]
    assert Group.query.count() == 1

    history = GroupHistory.query.order_by(GroupHistory.id).all()
    assert [h.rid for h in history] == [first["rid"], second["rid"]]
    assert [h.consent_start_date for h in history] == ["2026-05-25", "2026-06-01"]
    assert history[1].member_list == BODY["member_list"]


def test_identical_reregistration_is_accepted(client):
    assert_envelope(_register(client), 201, "Success", "Group registered successfully.")
    assert_envelope(_register(client), 201, "Success", "Group consent window updated.")
    assert GroupHistory.query.count() == 2


def test_add_group_alias_still_works(client):
    out = assert_envelope(client.post("/api/v1/add_group", json=BODY), 201, "Success",
                          "Group registered successfully.")
    assert out["group_id"] == GID
    assert GroupHistory.query.count() == 1


def test_register_group_has_no_warmup_field(client):
    out = assert_envelope(_register(client, warmup=True), 201)
    assert "warmup" not in out
    assert "warmup" not in Group.query.filter_by(group_id=GID).one().group_info


# ------------------------------------------------------------------ 409


def test_member_mismatch_is_409_and_changes_nothing(client):
    first = assert_envelope(_register(client), 201)
    out = assert_envelope(
        _register(client, member_list=["cp_007", "aya_999"], consent_start_date="2026-06-01"),
        409, "Member Mismatch",
        "group_id dyad_007 is already registered with members [cp_007, aya_007]; "
        "membership cannot be changed via /register_group.",
    )
    # The submitted (rejected) list and the other parsed fields are echoed.
    assert out["member_list"] == ["cp_007", "aya_999"]
    assert out["group_id"] == GID
    assert out["consent_start_date"] == "2026-06-01"
    assert out["consent_end_date"] == BODY["consent_end_date"]

    group = Group.query.filter_by(group_id=GID).one()
    assert group.group_info["member_list"] == BODY["member_list"]
    assert group.group_info["consent_start_date"] == "2026-05-25"
    assert group.rid == first["rid"]
    assert GroupHistory.query.count() == 1


def test_member_count_change_is_a_mismatch(client):
    assert_envelope(_register(client), 201)
    assert_envelope(_register(client, member_list=["cp_007", "aya_007", "cp_007"]), 409, "Member Mismatch")


# ------------------------------------------------------------------ 400


def test_missing_body_echoes_all_null(client):
    response = client.post("/api/v1/register_group", data=b"", content_type="application/json")
    out = assert_envelope(response, 400, "Invalid Parameter", "Request body is required.")
    for key in ECHO_KEYS:
        assert out[key] is None


def test_malformed_date_is_null_and_other_fields_echo(client):
    out = assert_envelope(
        _register(client, consent_start_date="2026-05-32"), 400, "Invalid Parameter",
        "consent_start_date must be a date in YYYY-MM-DD format; got '2026-05-32'.",
    )
    assert out["consent_start_date"] is None
    assert out["group_id"] == GID
    assert out["member_list"] == BODY["member_list"]
    assert out["consent_end_date"] == BODY["consent_end_date"]
    assert Group.query.count() == 0
    assert GroupHistory.query.count() == 0


@pytest.mark.parametrize("field", ECHO_KEYS)
def test_missing_field_is_named_and_null(client, field):
    body = {k: v for k, v in BODY.items() if k != field}
    out = assert_envelope(client.post("/api/v1/register_group", json=body), 400, "Invalid Parameter",
                          f"{field} is required.")
    assert out[field] is None
    for key in ECHO_KEYS:
        if key != field:
            assert out[key] == BODY[key]


def test_non_string_group_id_and_non_list_members_are_rejected(client):
    out = assert_envelope(_register(client, group_id=7), 400, "Invalid Parameter", "group_id must be a string.")
    assert out["group_id"] is None
    out = assert_envelope(_register(client, member_list="cp_007,aya_007"), 400, "Invalid Parameter",
                          "member_list must be a list.")
    assert out["member_list"] is None


# ------------------------------------------------------------- 503 / 500


def test_database_write_failure_is_503_and_saves_nothing(client, monkeypatch):
    def fail_commit():
        raise OperationalError("INSERT INTO groups", {}, Exception("disk full"))

    monkeypatch.setattr(db.session, "commit", fail_commit)
    out = assert_envelope(_register(client), 503, "Service Unavailable", "Database write failed; retry with backoff.")
    for key in ECHO_KEYS:
        assert out[key] == BODY[key]
    monkeypatch.undo()
    assert Group.query.count() == 0
    assert GroupHistory.query.count() == 0


def test_database_read_failure_is_503(client, monkeypatch):
    class BrokenGroup:
        class query:  # noqa: N801
            @staticmethod
            def filter_by(**kwargs):
                raise OperationalError("SELECT", {}, Exception("database is locked"))

    monkeypatch.setattr("app.routes.group.Group", BrokenGroup)
    assert_envelope(_register(client), 503, "Service Unavailable", "Database write failed; retry with backoff.")


def test_unexpected_failure_is_500_with_echo(client, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr("app.routes.group.GroupHistory", boom)
    out = assert_envelope(_register(client), 500, "Internal Error", "Internal server error.")
    assert out["group_id"] == GID
    monkeypatch.undo()
    assert Group.query.count() == 0
