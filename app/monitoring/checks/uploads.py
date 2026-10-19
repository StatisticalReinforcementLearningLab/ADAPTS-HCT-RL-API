"""
Registration, upload and reward checks (Monitoring-Spec §5.B): B11, B5, B8, B3.
"""

from __future__ import annotations

import collections
import datetime

from app.monitoring import catalog
from app.monitoring.checks.common import (
    AGENT_ORDER,
    CheckContext,
    ConsentTimeline,
    actions_on_day,
    active_in_week,
    day_bounds,
    deadline_spec,
    finished_updates,
    is_missing,
    lookback_start,
    start_dates,
    update_event_time,
    update_week,
    updates_of_week,
    week_closing_close,
)
from app.monitoring.findings import Finding, capped, dedup_key
from app.study_calendar import cohort_warmup_active, per_dyad_warmup, study_week, weekly_deadline

# ------------------------------------------------------------------- B11


def first_update_of_week(monday: datetime.date):
    reqs = updates_of_week(monday)
    return reqs[0] if reqs else None


def missing_closing_uploads(monday: datetime.date, first_req, timeline: ConsentTimeline | None = None) -> list[str]:
    """
    Dyads active in the week before ``monday`` (consent dates as recorded when
    the week's first ``/update`` arrived) with no upload stamped between
    Monday 00:00 and that update's ``request_timestamp`` and received before
    it (Monitoring-Spec B11; API-Spec §2.3 week-closing upload).
    """
    from app.models import DataUpload

    timeline = timeline or ConsentTimeline.load()
    consents = timeline.as_of(first_req.created_at)
    prev = monday - datetime.timedelta(days=7)
    active = sorted(gid for gid, c in consents.items() if active_in_week(c, prev))
    if not active:
        return []
    start = datetime.datetime.combine(monday, datetime.time.min)
    q = DataUpload.query.with_entities(DataUpload.group_id).filter(
        DataUpload.group_id.in_(active),
        DataUpload.request_timestamp >= start,
        DataUpload.request_timestamp <= first_req.request_timestamp,
    )
    if first_req.created_at is not None:
        q = q.filter(DataUpload.created_at <= first_req.created_at)
    have = {gid for (gid,) in q.distinct()}
    return [gid for gid in active if gid not in have]


def b11_for_week(monday: datetime.date, first_req=None, timeline: ConsentTimeline | None = None):
    """
    B11 at the arrival of week ``monday``'s first ``/update``: a yellow alarm
    for dyads missing their week-closing upload, a red one for dyads that also
    missed it the week before. Returns ``(findings, missing_group_ids)``.
    """
    first_req = first_req or first_update_of_week(monday)
    if first_req is None:
        return [], []
    timeline = timeline or ConsentTimeline.load()
    missing = missing_closing_uploads(monday, first_req, timeline)
    if not missing:
        return [], []
    prev = monday - datetime.timedelta(days=7)
    prev_req = first_update_of_week(prev)
    prev_missing = set(missing_closing_uploads(prev, prev_req, timeline)) if prev_req is not None else set()
    repeat = [g for g in missing if g in prev_missing]
    first_time = [g for g in missing if g not in prev_missing]
    at = first_req.request_timestamp
    hhmm = at.strftime("%H:%M")
    out = []
    if first_time:
        out.append(Finding(
            "B11",
            f"Week-closing upload missing at the {hhmm} update for {len(first_time)} dyads.",
            dedup_key("B11", monday, "yellow"),
            at=at,
            detail={"study_week_monday": monday, "group_ids": first_time, "update_rid": first_req.rid},
            rid=first_req.rid,
            value=float(len(first_time)),
        ))
    if repeat:
        out.append(Finding(
            "B11",
            f"Week-closing upload missing at the {hhmm} update for {len(repeat)} dyads for the second week in a row.",
            dedup_key("B11", monday, "red"),
            at=at,
            detail={"study_week_monday": monday, "group_ids": repeat, "update_rid": first_req.rid,
                    "also_missing_week": prev},
            rid=first_req.rid,
            value=float(len(repeat)),
            severity=catalog.RED,
        ))
    return out, missing


def check_b11(ctx: CheckContext) -> list[Finding]:
    """B11 for every week whose first ``/update`` arrived in the window."""
    from app.models import ModelUpdateRequests

    reqs = (
        ModelUpdateRequests.query.filter(
            ModelUpdateRequests.request_timestamp > ctx.window_start,
            ModelUpdateRequests.request_timestamp <= ctx.window_end,
        )
        .order_by(ModelUpdateRequests.request_timestamp.asc(), ModelUpdateRequests.id.asc())
        .all()
    )
    timeline = ConsentTimeline.load()
    out: list[Finding] = []
    done = set()
    for req in reqs:
        monday = update_week(req)
        if monday in done:
            continue
        done.add(monday)
        first = first_update_of_week(monday)
        if first is None or first.id != req.id:
            continue
        findings, _ = b11_for_week(monday, first, timeline)
        out.extend(findings)
    return out


# -------------------------------------------------------------------- B5


def _eligible_version(decision_type: str, monday: datetime.date, rids: list[str]):
    """
    The version of ``decision_type`` that governs week ``monday`` given the
    updates ``rids`` of that week so far (contract D13): the latest published
    one of the week published by the deadline, else the latest published
    version of an earlier week (F-U2).
    """
    from app.models import PolicyVersion

    deadline = weekly_deadline(monday, deadline_spec("MONITOR_WEEK_READY_BY", "MON 05:45"))
    base = PolicyVersion.query.filter(
        PolicyVersion.decision_type == decision_type,
        PolicyVersion.status == PolicyVersion.STATUS_PUBLISHED,
        PolicyVersion.published_at_study.isnot(None),
    )
    version = None
    if rids:
        version = (
            base.filter(
                PolicyVersion.study_week_monday == monday,
                PolicyVersion.update_rid.in_(rids),
                PolicyVersion.published_at_study <= deadline,
            )
            .order_by(PolicyVersion.id.desc())
            .first()
        )
    if version is None:
        version = (
            base.filter(PolicyVersion.study_week_monday < monday)
            .order_by(PolicyVersion.study_week_monday.desc(), PolicyVersion.id.desc())
            .first()
        )
    return version


def b5_for_update(req, timeline: ConsentTimeline | None = None) -> list[Finding]:
    """
    B5 at an update of week W: each dyad active in W and past its first week
    (no cohort warm-up in W) that has no posterior for an agent in the
    version governing W after this update. The cause goes in ``detail``.
    """
    from app.extensions import db
    from app.models import ModelParameters, PolicyVersion

    if req.status == "processing":
        return []
    from app.hyperparams import get_hyperparams

    hp = get_hyperparams()
    monday = update_week(req)
    timeline = timeline or ConsentTimeline.load()
    consents = timeline.as_of(req.created_at)
    if cohort_warmup_active(monday, start_dates(consents), n_warm=hp.n_warm):
        return []
    required = sorted(
        gid for gid, c in consents.items()
        if active_in_week(c, monday) and 1 <= study_week(c.start, monday) <= hp.trial_weeks
        and not per_dyad_warmup(c.start, monday, hp.t_warm)
    )
    if not required:
        return []
    rids = [r.rid for r in updates_of_week(monday) if (r.request_timestamp, r.id) <= (req.request_timestamp, req.id)]
    own = {v.decision_type: v for v in PolicyVersion.query.filter_by(update_rid=req.rid).all()}
    out: list[Finding] = []
    for dt in AGENT_ORDER:
        version = _eligible_version(dt, monday, rids)
        with_posterior, with_fit = set(), set()
        if version is not None:
            for gid, snap in db.session.query(ModelParameters.group_id, ModelParameters.snapshot_type).filter(
                ModelParameters.policy_version_id == version.id,
                ModelParameters.snapshot_type.in_(("posterior", "local_fit")),
            ):
                (with_posterior if snap == "posterior" else with_fit).add(gid)
        rows = []
        for gid in required:
            if gid in with_posterior:
                continue
            if version is None:
                cause = "no_policy_version"
            elif gid in with_fit:
                cause = "local_fit_without_posterior"
            else:
                cause = "no_local_fit"
            own_version = own.get(dt)
            if own_version is not None and own_version.status != PolicyVersion.STATUS_PUBLISHED:
                cause += f"; this update's version was {own_version.status}"
            rows.append({"group_id": gid, "cause": cause})
        if not rows:
            continue
        out.append(Finding(
            "B5",
            f"{len(rows)} dyads have no {dt} policy for the week of {monday.isoformat()}; their decisions use the "
            f"50/50 fallback (F-A3).",
            dedup_key("B5", req.rid, dt),
            at=update_event_time(req),
            detail={"study_week_monday": monday, "agent": dt, "update_rid": req.rid,
                    "policy_version_id": version.id if version is not None else None,
                    "policy_week": version.study_week_monday if version is not None else None,
                    "group_ids": [r["group_id"] for r in rows], "dyads": capped(rows, 50)},
            decision_type=dt,
            rid=req.rid,
            value=float(len(rows)),
            fallback_used="F-A3",
        ))
    return out


def check_b5(ctx: CheckContext) -> list[Finding]:
    timeline = ConsentTimeline.load()
    out: list[Finding] = []
    for req in finished_updates(lookback_start(ctx), ctx.window_end):
        out.extend(b5_for_update(req, timeline))
    return out


# -------------------------------------------------------------------- B8


def check_b8(ctx: CheckContext) -> list[Finding]:
    """
    B8 (daily, previous day): decisions with no upload from the same dyad in
    the 30 minutes before them (including the Monday REL decision), and dyads
    with more than one week-closing upload (Monday 00:00 to the week's
    first update).
    """
    from app.models import DataUpload

    day = (ctx.now - datetime.timedelta(days=1)).date()
    window = datetime.timedelta(minutes=catalog.B8_UPLOAD_MINUTES)
    no_upload = []
    for action in actions_on_day(day):
        found = (
            DataUpload.query.with_entities(DataUpload.id)
            .filter(
                DataUpload.group_id == action.group_id,
                DataUpload.request_timestamp >= action.request_timestamp - window,
                DataUpload.request_timestamp <= action.request_timestamp,
            )
            .first()
        )
        if found is None:
            no_upload.append({"rid": action.rid, "group_id": action.group_id, "decision_type": action.decision_type,
                              "decision_idx": action.decision_idx, "request_timestamp": action.request_timestamp})
    duplicates = []
    if day.weekday() == 0:
        start = datetime.datetime.combine(day, datetime.time.min)
        close = week_closing_close(day)
        counts = collections.Counter(
            gid for (gid,) in DataUpload.query.with_entities(DataUpload.group_id).filter(
                DataUpload.request_timestamp >= start, DataUpload.request_timestamp <= close
            )
        )
        duplicates = [{"group_id": g, "n_week_closing_uploads": n} for g, n in sorted(counts.items()) if n > 1]
    if not no_upload and not duplicates:
        return []
    parts = []
    if no_upload:
        parts.append(f"{len(no_upload)} decisions without an upload in the {catalog.B8_UPLOAD_MINUTES} minutes before")
    if duplicates:
        parts.append(f"{len(duplicates)} dyads with more than one week-closing upload")
    return [Finding(
        "B8",
        f"Uploads on {day.isoformat()}: " + "; ".join(parts) + ".",
        dedup_key("B8", day),
        at=ctx.now,
        detail={"day": day, "minutes": catalog.B8_UPLOAD_MINUTES, "n_without_upload": len(no_upload),
                "decisions_without_upload": capped(no_upload, 50), "duplicate_week_closing": duplicates,
                "group_ids": sorted({r["group_id"] for r in no_upload} | {r["group_id"] for r in duplicates})},
        value=float(len(no_upload) + len(duplicates)),
    )]


# -------------------------------------------------------------------- B3


def _miss_shares(uploads) -> dict[str, float]:
    counts: collections.Counter = collections.Counter()
    fields: set[str] = set()
    for upload in uploads:
        data = upload.data if isinstance(upload.data, dict) else {}
        fields.update(data)
        for key, value in data.items():
            if is_missing(value):
                counts[key] += 1
    n = len(uploads)
    return {f: counts[f] / n for f in fields} if n else {}


def check_b3(ctx: CheckContext) -> list[Finding]:
    """B3 (daily): per upload field, yesterday's share of ``"miss"`` against the previous 7 days; |change| > 0.15."""
    from app.models import DataUpload

    day = (ctx.now - datetime.timedelta(days=1)).date()
    start, end = day_bounds(day)
    base_start = start - datetime.timedelta(days=catalog.B3_BASELINE_DAYS)

    def uploads(a, b):
        return DataUpload.query.filter(DataUpload.request_timestamp >= a, DataUpload.request_timestamp < b).all()

    today, base = uploads(start, end), uploads(base_start, start)
    if not today or not base:
        return []
    s_today, s_base = _miss_shares(today), _miss_shares(base)
    rows = []
    for field in sorted(set(s_today) | set(s_base)):
        a, b = s_today.get(field, 0.0), s_base.get(field, 0.0)
        if abs(a - b) > catalog.B3_MISS_DELTA:
            rows.append({"field": field, "share_day": round(a, 6), "share_previous_7d": round(b, 6),
                         "change": round(a - b, 6)})
    if not rows:
        return []
    return [Finding(
        "B3",
        f"The share of 'miss' changed by more than {catalog.B3_MISS_DELTA} on {day.isoformat()} for "
        f"{len(rows)} upload fields.",
        dedup_key("B3", day),
        at=ctx.now,
        detail={"day": day, "n_uploads_day": len(today), "n_uploads_previous_7d": len(base), "fields": rows},
        value=max(abs(r["change"]) for r in rows),
        threshold=catalog.B3_MISS_DELTA,
    )]


__all__ = [
    "first_update_of_week",
    "missing_closing_uploads",
    "b11_for_week",
    "check_b11",
    "b5_for_update",
    "check_b5",
    "check_b8",
    "check_b3",
]
