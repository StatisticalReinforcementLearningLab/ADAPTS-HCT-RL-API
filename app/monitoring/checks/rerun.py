"""
C5: the week's update can be rerun exactly (Monitoring-Spec §5.C; contract D14).

For every completed update of week W (each ``/update`` rid, including a
second call in the same week), on Tuesday 02:00:

1. copy the API database into a temporary in-memory SQLite database, keeping
   only what existed when the update started (``model_update_requests.started_at``,
   server clock): uploads received by then, decisions minted before, the
   learner rows, policy versions and frozen settings written before, consent
   dates from ``group_history`` as of then, and ``study_data`` rows derived
   before (rows derived by this update or later are dropped and derived
   again from the uploads received before the update started, exactly as the
   update did);
2. rerun :func:`app.learner.run_weekly_update` there with the same rid and
   request timestamp;
3. compare every number the update recorded — per agent and dyad theta-hat
   and Sigma-hat (``local_fit``), Theta-hat, Sigma_0 and tau-hat^2 (``hyper``),
   alpha^EB and Sigma^EB (``posterior``) — and each policy version's status,
   exactly. Each decision is reproduced separately by A4.

A version is in force for week W only if its ``published_at_study`` is at or
before Monday 05:45 (contract D13), and the REL version in force feeds the
AYA and CP targets; the rerun therefore reproduces the original's side of
that deadline (on time: zero elapsed time; late: past the deadline).
"""

from __future__ import annotations

import datetime

from app.extensions import db
from app.monitoring import catalog
from app.monitoring.alarms import describe_exception
from app.monitoring.checks.common import CheckContext, deadline_spec, update_event_time, update_week, updates_of_week
from app.monitoring.findings import Finding, capped, dedup_key
from app.study_calendar import week_monday, weekly_deadline

#: Tables the learner and the reward derivation never read; not copied.
SKIPPED_TABLES = frozenset({
    "monitor_alarms", "api_error_events", "monitor_runs", "update_weeks", "monitor_pulls", "upload_quarantine",
})


def _cutoff_rows(table: str, rows: list[dict], started: datetime.datetime, req) -> list[dict]:
    """The rows of ``table`` as they were when the update started (see the module docstring)."""
    def before(col, inclusive=False):
        out = []
        for r in rows:
            value = r.get(col)
            if value is None or (value <= started if inclusive else value < started):
                out.append(r)
        return out

    if table == "data_uploads":
        return before("created_at", inclusive=True)
    if table == "actions":
        return before("timestamp")
    if table == "study_data":
        return [r for r in rows if r.get("derived_at") is not None and r["derived_at"] < started
                and (r.get("created_at") is None or r["created_at"] < started)]
    if table == "model_parameters":
        return [r for r in before("timestamp") if r.get("update_rid") != req.rid]
    if table == "policy_versions":
        return [r for r in before("created_at") if r.get("update_rid") != req.rid]
    if table in ("hyperparameter_sets", "noise_init"):
        return before("created_at")
    if table == "model_update_requests":
        return [r for r in rows if r.get("created_at") is None or r["created_at"] <= req.created_at]
    if table == "group_history":
        return before("recorded_at", inclusive=True)
    return rows


def _groups_as_of(groups: list[dict], history: list[dict], started: datetime.datetime) -> list[dict]:
    """``groups`` rows registered by ``started`` with consent dates from the latest history row by then."""
    latest: dict[str, dict] = {}
    for h in sorted(history, key=lambda r: (r.get("recorded_at") or datetime.datetime.min, r.get("id") or 0)):
        latest[h["group_id"]] = h
    out = []
    for g in groups:
        if g.get("created_at") is not None and g["created_at"] > started:
            continue
        h = latest.get(g["group_id"])
        if h is not None:
            info = dict(g.get("group_info") or {})
            info["consent_start_date"] = h["consent_start_date"]
            info["consent_end_date"] = h["consent_end_date"]
            g = {**g, "group_info": info}
        out.append(g)
    return out


def _snapshot(req, started: datetime.datetime) -> dict[str, list[dict]]:
    from sqlalchemy import select

    data: dict[str, list[dict]] = {}
    for table in db.metadata.sorted_tables:
        if table.name in SKIPPED_TABLES:
            continue
        rows = [dict(r) for r in db.session.execute(select(table)).mappings()]
        data[table.name] = _cutoff_rows(table.name, rows, started, req)
    if "groups" in data:
        data["groups"] = _groups_as_of(data["groups"], data.get("group_history", []), started)
    return data


def _payload(rows) -> dict:
    out = {}
    for row in rows:
        meta = row.metadata_json or {}
        out[(row.decision_type, row.snapshot_type, row.group_id)] = {
            "theta": row.theta,
            "covariance": row.covariance,
            "tau2": meta.get("tau2") if row.snapshot_type == "hyper" else None,
        }
    return out


def rerun_update(req) -> dict:
    """
    Rerun one completed update on a copy of the database (module docstring).
    Returns ``{"rows": {(decision_type, snapshot_type, group_id): {...}},
    "versions": {decision_type: status}, "summary": {...}}``.
    """
    from flask import Flask, current_app
    from sqlalchemy import insert

    from app import learner
    from app.clock import server_now
    from app.models import ModelParameters, PolicyVersion
    from app.reward_derivation import derive_study_data

    main = current_app._get_current_object()
    started = req.started_at or req.created_at
    data = _snapshot(req, started)

    monday = update_week(req)
    deadline = weekly_deadline(monday, deadline_spec("MONITOR_WEEK_READY_BY", "MON 05:45"))
    rel = PolicyVersion.query.filter_by(update_rid=req.rid, decision_type="dyad_game").first()
    late = rel is not None and rel.published_at_study is not None and rel.published_at_study > deadline
    if late:
        # Elapsed time past the deadline, as in the original run.
        rerun_started = server_now() - (deadline - req.request_timestamp) - datetime.timedelta(days=1)
    else:
        # A start in the future: the learner clamps the elapsed time to zero.
        rerun_started = server_now() + datetime.timedelta(days=1)

    copy = Flask("adapts_c5_rerun")
    copy.config.update({k: v for k, v in main.config.items() if not k.startswith("SQLALCHEMY_")})
    copy.config.update(
        SQLALCHEMY_DATABASE_URI="sqlite://",
        SQLALCHEMY_ENGINE_OPTIONS={},
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        FAULT_INJECTION_ENABLED=False,
    )
    copy.hyperparams = main.hyperparams
    copy.sampler = getattr(main, "sampler", None)
    copy.rl_algorithm = getattr(main, "rl_algorithm", None)
    db.init_app(copy)
    try:
        with copy.app_context():
            db.create_all()
            for table in db.metadata.sorted_tables:
                rows = data.get(table.name)
                if rows:
                    db.session.execute(insert(table), rows)
            db.session.commit()
            derive_study_data(copy, cutoff=started)
            summary = learner.run_weekly_update(req.rid, req.request_timestamp, rerun_started)
            rows = ModelParameters.query.filter_by(update_rid=req.rid).all()
            result = {
                "rows": _payload(rows),
                "versions": {v.decision_type: v.status for v in PolicyVersion.query.filter_by(update_rid=req.rid)},
                "summary": {"status": (summary or {}).get("status"), "learning": (summary or {}).get("learning")},
            }
            db.session.remove()
    finally:
        try:
            with copy.app_context():
                db.engine.dispose()
        except Exception:  # noqa: BLE001 - cleanup only
            pass
    return result


def _max_abs_diff(a, b):
    try:
        import numpy as np

        x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if x.shape != y.shape:
            return None
        return float(np.nanmax(np.abs(x - y))) if x.size else 0.0
    except Exception:  # noqa: BLE001
        return None


def compare(original: dict, rerun: dict) -> list[dict]:
    """Every difference between the original and the rerun results (exact comparison)."""
    diffs = []
    keys = sorted(set(original["rows"]) | set(rerun["rows"]), key=lambda k: tuple(str(x) for x in k))
    for key in keys:
        dt, snap, gid = key
        if key not in rerun["rows"]:
            diffs.append({"agent": dt, "snapshot": snap, "group_id": gid, "kind": "missing_in_rerun"})
            continue
        if key not in original["rows"]:
            diffs.append({"agent": dt, "snapshot": snap, "group_id": gid, "kind": "extra_in_rerun"})
            continue
        for field in ("theta", "covariance", "tau2"):
            a, b = original["rows"][key][field], rerun["rows"][key][field]
            if a != b:
                diffs.append({"agent": dt, "snapshot": snap, "group_id": gid, "kind": "value", "field": field,
                              "max_abs_diff": _max_abs_diff(a, b)})
    for dt in sorted(set(original["versions"]) | set(rerun["versions"])):
        a, b = original["versions"].get(dt), rerun["versions"].get(dt)
        if a != b:
            diffs.append({"agent": dt, "kind": "version_status", "original": a, "rerun": b})
    return diffs


def _previous_week_failed(monday: datetime.date) -> bool:
    from app.models import MonitorAlarm

    prev = (monday - datetime.timedelta(days=7)).isoformat()
    return (
        db.session.query(MonitorAlarm.id)
        .filter(MonitorAlarm.item_id == "C5", MonitorAlarm.dedup_key.like(f"C5:{prev}:%"))
        .first()
        is not None
    )


def c5_for_update(req, at: datetime.datetime | None = None) -> list[Finding]:
    """C5 for one update rid: a finding when the rerun differs or cannot be done."""
    from app.models import ModelParameters, PolicyVersion

    if req.status != "completed":
        return []
    original_rows = ModelParameters.query.filter_by(update_rid=req.rid).all()
    if not original_rows:
        return []  # nothing was fitted (F-U3): nothing to reproduce
    original = {
        "rows": _payload(original_rows),
        "versions": {v.decision_type: v.status for v in PolicyVersion.query.filter_by(update_rid=req.rid)},
    }
    monday = update_week(req)
    red = _previous_week_failed(monday)
    when = at or update_event_time(req)
    try:
        rerun = rerun_update(req)
    except Exception as exc:  # noqa: BLE001 - a rerun that cannot run fails C5
        db.session.rollback()
        return [Finding(
            "C5",
            f"The update {req.rid} of the week of {monday.isoformat()} could not be rerun ({type(exc).__name__}).",
            dedup_key("C5", monday, req.rid),
            at=when,
            detail={"study_week_monday": monday, "update_rid": req.rid, "kind": "rerun_error",
                    "error": describe_exception(exc), "red_after_two_weeks": red},
            rid=req.rid,
            severity=catalog.RED if red else None,
        )]
    diffs = compare(original, rerun)
    if not diffs:
        return []
    return [Finding(
        "C5",
        f"Rerunning the update {req.rid} of the week of {monday.isoformat()} gave {len(diffs)} different results.",
        dedup_key("C5", monday, req.rid),
        at=when,
        detail={"study_week_monday": monday, "update_rid": req.rid, "kind": "mismatch", "n_differences": len(diffs),
                "differences": capped(diffs, 50), "red_after_two_weeks": red},
        rid=req.rid,
        value=float(len(diffs)),
        severity=catalog.RED if red else None,
    )]


def check_c5(ctx: CheckContext) -> list[Finding]:
    """C5 (Tuesday 02:00): rerun every completed update of the current week."""
    monday = week_monday(ctx.now)
    out: list[Finding] = []
    for req in updates_of_week(monday, until=ctx.now):
        out.extend(c5_for_update(req, at=ctx.now))
    return out


__all__ = ["rerun_update", "compare", "c5_for_update", "check_c5"]
