"""
The weekly update watch (Monitoring-Spec §5.C, §7 F-U2/F-U4, §9.4): C11, C7,
C1, C3, C9, C10, the F-U4 timeout and the ``update_weeks`` bookkeeping.

The host calls ``/update`` every Monday at 03:00 (API-Spec §3). Week ``W`` is
*ready* when, by Monday 05:45 (``MONITOR_WEEK_READY_BY``), an update of the
week completed and published every valid policy (contract D13: a version is
eligible iff ``published_at_study`` is at or before the deadline). An update
is *required* for ``W`` when at least one dyad was active in the week before
(it has outcomes to learn from and owes a week-closing upload).

``update_weeks.status``: ``waiting`` (no call yet), ``running`` (latest call
processing), ``ready`` (a call completed by the deadline; stays ready),
``failed`` (the latest call failed, or completed after the deadline),
``stuck`` (the latest call timed out, F-U4).
"""

from __future__ import annotations

import datetime

from app.extensions import db
from app.monitoring import catalog
from app.monitoring.alarms import sanitize_error
from app.monitoring.checks.common import (
    CheckContext,
    active_in_week,
    actions_between,
    consents_at_logical,
    deadline_spec,
    finished_updates,
    lookback_start,
    update_event_time,
    update_week,
    updates_of_week,
)
from app.monitoring.findings import Finding, capped, dedup_key, jsonable
from app.study_calendar import week_monday, weekly_deadline

STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
TIMEOUT = "timeout"

WEEK_WAITING = "waiting"
WEEK_RUNNING = "running"
WEEK_READY = "ready"
WEEK_FAILED = "failed"
WEEK_STUCK = "stuck"


def ready_deadline(monday: datetime.date) -> datetime.datetime:
    return weekly_deadline(monday, deadline_spec("MONITOR_WEEK_READY_BY", "MON 05:45"))


def expected_by(monday: datetime.date) -> datetime.datetime:
    return weekly_deadline(monday, deadline_spec("MONITOR_UPDATE_EXPECTED_BY", "MON 03:30"))


def timeout_minutes() -> int:
    from flask import current_app

    return int(current_app.config.get("MONITOR_UPDATE_TIMEOUT_MIN", 30))


def update_required(monday: datetime.date, at: datetime.datetime) -> bool:
    """An update is expected for week ``monday`` iff some dyad (as recorded by ``at``) was active the week before."""
    prev = monday - datetime.timedelta(days=7)
    return any(active_in_week(c, prev) for c in consents_at_logical(at).values())


def versions_of(rid: str) -> list:
    from app.models import PolicyVersion

    return PolicyVersion.query.filter_by(update_rid=rid).order_by(PolicyVersion.id.asc()).all()


# ------------------------------------------------------------ update_weeks


def ensure_week(monday: datetime.date):
    """The ``update_weeks`` row of ``monday`` (created ``waiting`` if absent)."""
    from sqlalchemy.exc import IntegrityError

    from app.models import UpdateWeek

    row = UpdateWeek.query.filter_by(study_week_monday=monday).first()
    if row is not None:
        return row
    row = UpdateWeek(study_week_monday=monday, rids=[], status=WEEK_WAITING)
    db.session.add(row)
    try:
        db.session.flush()
    except IntegrityError:
        db.session.rollback()
        row = UpdateWeek.query.filter_by(study_week_monday=monday).first()
    return row


def record_received(req, missing: list[str] | None = None) -> None:
    """``/update`` arrived: append its rid, mark the week running (unless ready), keep the first call time."""
    row = ensure_week(update_week(req))
    rids = list(row.rids or [])
    if req.rid not in rids:
        rids.append(req.rid)
        row.rids = rids
    if row.first_call_at is None or req.request_timestamp < row.first_call_at:
        row.first_call_at = req.request_timestamp
    if row.status != WEEK_READY:
        row.status = WEEK_RUNNING
    if missing is not None and row.missing_closing_uploads is None:
        row.missing_closing_uploads = list(missing)
    db.session.commit()


def record_finished(req) -> None:
    """An update ended: ``ready`` when it completed by the deadline, else ``failed`` / ``stuck``."""
    monday = update_week(req)
    row = ensure_week(monday)
    rids = list(row.rids or [])
    if req.rid not in rids:
        rids.append(req.rid)
        row.rids = rids
    if row.first_call_at is None:
        row.first_call_at = req.request_timestamp
    finished = update_event_time(req) if req.status == STATUS_COMPLETED else None
    if finished is not None:
        row.finished_at = finished
    if row.status == WEEK_READY:
        db.session.commit()
        return
    deadline = ready_deadline(monday)
    if req.status == STATUS_COMPLETED and req.published_at_study is not None and req.published_at_study <= deadline \
            and not any(v.status == "failed" for v in versions_of(req.rid)):
        row.status = WEEK_READY
        row.ready_at = req.published_at_study
    elif req.status == STATUS_FAILED and (req.error_message or "") == TIMEOUT:
        row.status = WEEK_STUCK
    elif req.status in (STATUS_FAILED, STATUS_COMPLETED):
        row.status = WEEK_FAILED
    db.session.commit()


def mark_timeouts(now: datetime.datetime) -> list[str]:
    """
    F-U4: every update still ``processing`` more than
    ``MONITOR_UPDATE_TIMEOUT_MIN`` minutes after its request time (study clock)
    is marked ``failed`` with ``error_message = "timeout"``; its week becomes
    ``stuck``. C7 reports it at 05:45; the host may send ``/update`` again.
    """
    from app.clock import server_now
    from app.models import ModelUpdateRequests

    limit = now - datetime.timedelta(minutes=timeout_minutes())
    reqs = (
        ModelUpdateRequests.query.filter(
            ModelUpdateRequests.status == STATUS_PROCESSING,
            ModelUpdateRequests.request_timestamp < limit,
        )
        .order_by(ModelUpdateRequests.id.asc())
        .all()
    )
    marked = []
    for req in reqs:
        req.status = STATUS_FAILED
        req.error_message = TIMEOUT
        req.completed_at = server_now()
        marked.append(req.rid)
    if marked:
        db.session.commit()
        for req in reqs:
            record_finished(req)
    sync_running_weeks()
    return marked


def sync_running_weeks() -> None:
    """
    A week still marked ``running`` whose calls have all ended (e.g. the update
    thread could not start, so no finish hook ran) takes the state of its
    latest call.
    """
    from app.models import UpdateWeek

    for row in UpdateWeek.query.filter_by(status=WEEK_RUNNING).all():
        reqs = updates_of_week(row.study_week_monday)
        if reqs and all(r.status != STATUS_PROCESSING for r in reqs):
            record_finished(reqs[-1])


# -------------------------------------------------------------------- C11


def check_c11(ctx: CheckContext) -> list[Finding]:
    """C11 (Monday 03:30): no ``/update`` of the week has arrived although one is required."""
    monday = week_monday(ctx.now)
    at = expected_by(monday)
    if not update_required(monday, at) or updates_of_week(monday, until=at):
        return []
    return [Finding(
        "C11",
        f"No /update arrived for the week of {monday.isoformat()} by {at.strftime('%H:%M')} Monday; "
        f"the host's weekly trigger did not run.",
        dedup_key("C11", monday),
        at=at,
        detail={"study_week_monday": monday, "expected_by": at},
    )]


# --------------------------------------------------------------------- C7


def week_readiness(monday: datetime.date, deadline: datetime.datetime | None = None) -> dict:
    """
    Readiness of week ``monday`` at its deadline (contract D13): ``{"ready",
    "reason", "rids", "ready_rid", "ready_at", "latest": {...}}``. Reasons
    when not ready: ``never_arrived``, ``running``, ``timeout``, ``failed``,
    ``late``, ``not_all_published``.
    """
    deadline = deadline or ready_deadline(monday)
    reqs = updates_of_week(monday, until=deadline)
    out = {"ready": False, "reason": None, "rids": [r.rid for r in reqs], "ready_rid": None, "ready_at": None,
           "deadline": deadline, "latest": None}
    if not reqs:
        out["reason"] = "never_arrived"
        return out
    for req in reqs:
        if req.status == STATUS_COMPLETED and req.published_at_study is not None \
                and req.published_at_study <= deadline \
                and not any(v.status == "failed" for v in versions_of(req.rid)):
            out.update(ready=True, ready_rid=req.rid, ready_at=req.published_at_study)
            return out
    latest = reqs[-1]
    versions = versions_of(latest.rid)
    if latest.status == STATUS_PROCESSING:
        late = deadline - latest.request_timestamp > datetime.timedelta(minutes=timeout_minutes())
        reason = TIMEOUT if late else "running"
    elif latest.status == STATUS_FAILED:
        reason = TIMEOUT if (latest.error_message or "") == TIMEOUT else "failed"
    elif any(v.status == "failed" for v in versions):
        reason = "not_all_published"
    else:
        reason = "late"
    out["reason"] = reason
    out["latest"] = {
        "rid": latest.rid,
        "status": latest.status,
        "request_timestamp": latest.request_timestamp,
        "published_at_study": latest.published_at_study,
        "error": sanitize_error(latest.error_message),
        "versions": {v.decision_type: v.status for v in versions},
    }
    return out


_C7_TEXT = {
    "never_arrived": "the update never arrived",
    "running": "the update was still running",
    "timeout": "the update was stuck and timed out (F-U4)",
    "failed": "the update failed",
    "late": "the update finished after the deadline",
    "not_all_published": "the update did not publish every required policy",
}


def check_c7(ctx: CheckContext) -> list[Finding]:
    """C7 (Monday 05:45): week not ready; every agent keeps its last valid policy for the whole week (F-U2/F-A3)."""
    monday = week_monday(ctx.now)
    deadline = ready_deadline(monday)
    if not update_required(monday, deadline):
        return []
    state = week_readiness(monday, deadline)
    if state["ready"]:
        return []
    return [Finding(
        "C7",
        f"Week of {monday.isoformat()} not ready at {deadline.strftime('%H:%M')}: {_C7_TEXT[state['reason']]}. "
        f"Last week's policies stay in force for the whole week.",
        dedup_key("C7", monday),
        at=deadline,
        detail={"study_week_monday": monday, **jsonable(state)},
        fallback_used="F-U2/F-A3",
    )]


def record_c7(monday: datetime.date) -> None:
    row = ensure_week(monday)
    row.c7_raised = True
    db.session.commit()


# --------------------------------------------------------------------- C1


def _hyper_rows(rid: str) -> list:
    from app.models import ModelParameters

    return ModelParameters.query.filter_by(update_rid=rid, snapshot_type="hyper").order_by(ModelParameters.id).all()


def _eb_fell_back(meta: dict) -> bool:
    return bool(meta.get("fallback_used")) or meta.get("converged") is False


def c1_agents(req) -> dict[str, str]:
    """``{agent: "failed" | "eb_fallback"}`` of one update (agent ``"all"`` when it failed before any agent ran)."""
    out: dict[str, str] = {}
    if req.status == STATUS_FAILED and (req.error_message or "") != TIMEOUT:
        failed = [v.decision_type for v in versions_of(req.rid) if v.status == "failed"]
        for agent in failed or ["all"]:
            out[agent] = "failed"
    for row in _hyper_rows(req.rid):
        if _eb_fell_back(row.metadata_json or {}):
            out.setdefault(row.decision_type, "eb_fallback")
    return out


def _c1_last_week(monday: datetime.date) -> set[str]:
    agents: set[str] = set()
    for req in updates_of_week(monday - datetime.timedelta(days=7)):
        if req.status == STATUS_PROCESSING:
            continue
        agents.update(c1_agents(req))
    return agents


def c1_for_update(req) -> list[Finding]:
    """
    C1: the update ended with an error (other than C3, which reports itself,
    and the F-U4 timeout, which C7 reports), or the EB optimizer returned a
    non-finite value or hit ``k_max`` so last week's (Theta, Sigma_0) or the
    prior were reused (F-U5). Red when the same agent had a C1 problem the
    week before.
    """
    from app.hyperparams import get_hyperparams

    if req.status == STATUS_PROCESSING:
        return []
    monday = update_week(req)
    prev = _c1_last_week(monday)
    at = update_event_time(req)
    out: list[Finding] = []
    if req.status == STATUS_FAILED and (req.error_message or "") != TIMEOUT:
        failed = [v for v in versions_of(req.rid) if v.status == "failed"]
        targets = [(v.decision_type, sanitize_error(v.reason)) for v in failed] or [("all", None)]
        for agent, reason in targets:
            red = agent in prev or "all" in prev
            out.append(Finding(
                "C1",
                f"Update {req.rid} for the week of {monday.isoformat()} failed"
                + (f" for {agent}" if agent != "all" else "") + "; last valid policies stay in force (F-U2).",
                dedup_key("C1", req.rid, agent, "failed"),
                at=at,
                detail={"kind": "update_failed", "agent": agent, "study_week_monday": monday, "update_rid": req.rid,
                        "error": reason or sanitize_error(req.error_message), "red_after_two_weeks": red},
                decision_type=None if agent == "all" else agent,
                rid=req.rid,
                fallback_used="F-U2",
                severity=catalog.RED if red else None,
            ))
    tol = get_hyperparams().eps_tol
    for row in _hyper_rows(req.rid):
        meta = row.metadata_json or {}
        if not _eb_fell_back(meta):
            continue
        agent = row.decision_type
        red = agent in prev or "all" in prev
        which = "the prior" if meta.get("fallback_used") == "prior" else "last week's (Theta, Sigma_0)"
        out.append(Finding(
            "C1",
            f"EB hyper-update for {agent} fell back to {which} (F-U5).",
            dedup_key("C1", req.rid, agent, "eb_fallback"),
            at=at,
            detail={"kind": "eb_fallback", "agent": agent, "study_week_monday": monday, "update_rid": req.rid,
                    "iterations": meta.get("iterations"), "grad_inf": meta.get("grad_inf"), "tol": tol,
                    "converged": meta.get("converged"), "fallback_used": meta.get("fallback_used"),
                    "termination": sanitize_error(meta.get("termination")), "N_w": meta.get("N_w"),
                    "red_after_two_weeks": red},
            decision_type=agent,
            rid=req.rid,
            value=meta.get("grad_inf") if isinstance(meta.get("grad_inf"), (int, float)) else None,
            threshold=tol,
            fallback_used="F-U5",
            severity=catalog.RED if red else None,
        ))
    return out


def check_c1(ctx: CheckContext) -> list[Finding]:
    out: list[Finding] = []
    for req in finished_updates(lookback_start(ctx), ctx.window_end):
        out.extend(c1_for_update(req))
    return out


# --------------------------------------------------------------------- C3


def _c3_rejected_last_week(decision_type: str, monday: datetime.date) -> bool:
    from app.models import PolicyVersion

    latest = (
        PolicyVersion.query.filter(
            PolicyVersion.decision_type == decision_type,
            PolicyVersion.study_week_monday == monday - datetime.timedelta(days=7),
        )
        .order_by(PolicyVersion.id.desc())
        .first()
    )
    return latest is not None and latest.status == PolicyVersion.STATUS_REJECTED


def c3_for_update(req) -> list[Finding]:
    """
    C3: an agent's new policy was rejected by the validity gate inside the
    update (the gate itself is in the learner, Monitoring-Spec §2.1); that
    agent keeps last week's policy (F-U2). Red when the same agent was
    rejected the week before too.
    """
    from app.models import ModelParameters, PolicyVersion

    out: list[Finding] = []
    for version in versions_of(req.rid):
        if version.status != PolicyVersion.STATUS_REJECTED:
            continue
        hyper = ModelParameters.query.filter_by(
            policy_version_id=version.id, snapshot_type="hyper"
        ).first()
        c3 = ((hyper.metadata_json or {}).get("c3") if hyper is not None else None) or {}
        red = _c3_rejected_last_week(version.decision_type, version.study_week_monday)
        failures = list(c3.get("failures") or [])
        out.append(Finding(
            "C3",
            f"C3 rejected the new {version.decision_type} policy for the week of "
            f"{version.study_week_monday.isoformat()}; last week's policy stays in force (F-U2).",
            dedup_key("C3", req.rid, version.decision_type),
            at=update_event_time(req),
            detail={"agent": version.decision_type, "study_week_monday": version.study_week_monday,
                    "update_rid": req.rid, "policy_version_id": version.id,
                    "failures": capped([sanitize_error(f) for f in failures], 20),
                    "checks": c3.get("checks"), "reason": sanitize_error(version.reason),
                    "red_after_two_weeks": red},
            decision_type=version.decision_type,
            rid=req.rid,
            fallback_used="F-U2",
            severity=catalog.RED if red else None,
        ))
    return out


def check_c3(ctx: CheckContext) -> list[Finding]:
    out: list[Finding] = []
    for req in finished_updates(lookback_start(ctx), ctx.window_end):
        out.extend(c3_for_update(req))
    return out


# --------------------------------------------------------------------- C9


def c9_for_action(action) -> list[Finding]:
    """C9 per decision: after freezing, a decision without a settings fingerprint or with one not in force."""
    from app.hyperparams import in_force_version

    frozen = in_force_version(at=action.request_timestamp)
    if frozen is None or action.hyperparams_hash == frozen.sha256:
        return []
    kind = "missing" if not action.hyperparams_hash else "mismatch"
    return [Finding(
        "C9",
        f"{action.decision_type} decision {action.decision_idx} of {action.group_id} used "
        + ("no settings fingerprint" if kind == "missing" else "a settings fingerprint")
        + f" that does not match approved version {frozen.version}.",
        dedup_key("C9", "action", action.rid),
        at=action.request_timestamp,
        detail={"kind": kind, "recorded_sha256": action.hyperparams_hash, "approved_sha256": frozen.sha256,
                "approved_version": frozen.version},
        group_id=action.group_id,
        decision_type=action.decision_type,
        decision_idx=action.decision_idx,
        rid=action.rid,
    )]


def c9_for_update(req) -> list[Finding]:
    """C9 per fit: every ``model_parameters`` / ``policy_versions`` row of the update carries the approved fingerprint."""
    from app.hyperparams import in_force_version
    from app.models import ModelParameters, PolicyVersion

    frozen = in_force_version(at=req.request_timestamp)
    if frozen is None:
        return []
    bad: dict[str, int] = {}
    for row in ModelParameters.query.filter_by(update_rid=req.rid).all():
        if row.hyperparams_hash != frozen.sha256:
            key = f"{row.snapshot_type}:{row.decision_type}:{'missing' if not row.hyperparams_hash else 'mismatch'}"
            bad[key] = bad.get(key, 0) + 1
    for row in PolicyVersion.query.filter_by(update_rid=req.rid).all():
        if row.hyperparams_hash != frozen.sha256:
            key = f"policy_version:{row.decision_type}:{'missing' if not row.hyperparams_hash else 'mismatch'}"
            bad[key] = bad.get(key, 0) + 1
    if not bad:
        return []
    return [Finding(
        "C9",
        f"Update {req.rid} wrote {sum(bad.values())} fitted records without the settings fingerprint of approved "
        f"version {frozen.version}.",
        dedup_key("C9", "fit", req.rid),
        at=update_event_time(req),
        detail={"update_rid": req.rid, "approved_version": frozen.version, "approved_sha256": frozen.sha256,
                "records": bad},
        rid=req.rid,
        value=float(sum(bad.values())),
    )]


def startup_finding(detail: dict, at: datetime.datetime) -> Finding:
    """C9 at start-up (Hyperparameters.md §2.3): the settings differ from the frozen set; the API refuses to start."""
    detail = dict(detail or {})
    return Finding(
        "C9",
        f"The API refused to start: the resolved settings differ from frozen version {detail.get('frozen_version')} "
        f"({detail.get('n_differences')} differences).",
        dedup_key("C9", "startup", detail.get("frozen_sha256"), detail.get("running_sha256")),
        at=at,
        detail={"kind": "startup_refused", **detail},
    )


def check_c9(ctx: CheckContext) -> list[Finding]:
    out: list[Finding] = []
    for action in actions_between(ctx.window_start, ctx.window_end):
        out.extend(c9_for_action(action))
    for req in finished_updates(lookback_start(ctx), ctx.window_end):
        out.extend(c9_for_update(req))
    return out


# -------------------------------------------------------------------- C10


def c10_finding() -> list[Finding]:
    """
    C10 (once, after the first learning update): the noise-scale record per
    agent from ``noise_init``. Green; yellow when the upper limit was used
    (``clip_used``, or an estimated scale with fewer than two week-1 rewards).
    sigma^2 is fixed by design (main_new.tex §5.6), so the yellow branch only
    fires if ``noise_init`` says the limit was used.
    """
    from app.models import HyperparameterSet, NoiseInit

    rows = NoiseInit.query.order_by(NoiseInit.id.asc()).all()
    if not rows:
        return []
    clipped = [r.decision_type for r in rows
               if r.clip_used or (r.source != "design" and (r.n_week1_rewards or 0) < 2)]
    first = HyperparameterSet.query.order_by(HyperparameterSet.version.asc()).first()
    at = first.in_force_from if first is not None else None
    per_agent = {
        r.decision_type: {
            "n_week1_rewards": r.n_week1_rewards, "s2": r.s2, "sigma_bar_sq": r.sigma_bar_sq,
            "sigma_sq": r.sigma_sq, "tau0_act": r.tau0_act, "tau0_mod": r.tau0_mod,
            "clip_used": bool(r.clip_used), "source": r.source, "study_week_monday": r.study_week_monday,
        }
        for r in rows
    }
    message = "Noise scale set at the first learning update: " + ", ".join(
        f"{dt} sigma^2 = {v['sigma_sq']:.4g} (limit {v['sigma_bar_sq']:.4g})" for dt, v in per_agent.items()
    ) + "."
    if clipped:
        message = f"Noise scale hit its upper limit for {', '.join(clipped)}. " + message
    return [Finding(
        "C10",
        message,
        dedup_key("C10", "noise_init"),
        at=at,
        detail={"agents": per_agent, "limit_used": clipped},
        severity=catalog.YELLOW if clipped else None,
    )]


def check_c10(ctx: CheckContext) -> list[Finding]:
    return c10_finding()


__all__ = [
    "ready_deadline",
    "expected_by",
    "update_required",
    "ensure_week",
    "record_received",
    "record_finished",
    "mark_timeouts",
    "check_c11",
    "week_readiness",
    "check_c7",
    "record_c7",
    "c1_agents",
    "c1_for_update",
    "check_c1",
    "c3_for_update",
    "check_c3",
    "c9_for_action",
    "c9_for_update",
    "startup_finding",
    "check_c9",
    "c10_finding",
    "check_c10",
]
