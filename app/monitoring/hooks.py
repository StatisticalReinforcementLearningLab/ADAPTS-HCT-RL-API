"""
Monitoring inside the API (Monitoring-Spec §2.1, §2.2, §6.9; implementation
contract §3.1, D17).

- :func:`after_response` is the single after-response step, registered by
  ``init_app`` as ``app.after_request`` for every ``/api/v1/*`` answer
  (framework 404 / 405 / 500 included). It maps (endpoint, status, title,
  message) to a §6 code (:mod:`app.monitoring.codes`) and, for every
  non-success answer, every ``AC-200-REPLAY`` and every ``AC-201-*``
  fallback, writes one ``api_error_events`` row: the ``rid``, the ids read
  from the request JSON when readable, and the SHA-256 and size of the
  request body (never the body). Each code except ``AC-200-REPLAY`` also
  becomes an alarm. Endpoints never write codes; this step is the only one.
- :func:`check_action_probability` is A13 inside ``/action``.
- :func:`after_action_committed`, :func:`on_update_received`,
  :func:`on_update_finished`, :func:`on_startup_settings_mismatch` run the
  per-decision, per-update and start-up checks.

Every hook catches everything: a crash is recorded as E7 (itself guarded)
and never changes the API's answer.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import uuid

from app.extensions import db
from app.monitoring import catalog, codes
from app.monitoring.alarms import emit, report_internal_error
from app.monitoring.timeutil import alarm_time, is_host_mode, monitoring_now

logger = logging.getLogger(__name__)

_MAX_ID = 255
_MAX_MESSAGE = 4000


def _clip(value, limit: int = _MAX_ID):
    return None if value is None else str(value)[:limit]


def _prepare_session() -> None:
    """The request is over: discard anything the route left uncommitted before monitoring writes."""
    session = db.session
    if session.new or session.dirty or session.deleted:
        session.rollback()


def _request_ids(payload) -> dict:
    """``group_id`` / ``decision_type`` / ``decision_idx`` of a request body, when readable."""
    if not isinstance(payload, dict):
        return {"group_id": None, "decision_type": None, "decision_idx": None}
    gid, dt, idx = payload.get("group_id"), payload.get("decision_type"), payload.get("decision_idx")
    return {
        "group_id": _clip(gid) if isinstance(gid, str) else None,
        "decision_type": _clip(dt) if isinstance(dt, str) else None,
        "decision_idx": idx if isinstance(idx, int) and not isinstance(idx, bool) and -(2**31) <= idx < 2**31
        else None,
    }


def _answer_time(payload) -> datetime.datetime:
    """When the answer happened on the study clock: the request's ``timestamp`` in host mode, else the server clock."""
    if is_host_mode() and isinstance(payload, dict) and isinstance(payload.get("timestamp"), str):
        from app.clock import to_study_local

        try:
            return to_study_local(payload["timestamp"])
        except ValueError:
            pass
    return monitoring_now()


def record_answer(path: str, status: int, title, message, rid, classified, *, payload, raw: bytes):
    """Write the ``api_error_events`` row of one coded answer and raise its alarm. Returns the event row."""
    from app.models import ApiErrorEvent
    from app.monitoring.checks.records import code_finding, fallback_reason_for

    _prepare_session()
    ids = _request_ids(payload)
    event = ApiErrorEvent(
        event_id=str(uuid.uuid4()),
        code=classified.code,
        endpoint=_clip(path),
        http_status=int(status),
        title=_clip(title),
        message=_clip(message, _MAX_MESSAGE),
        rid=_clip(rid),
        group_id=ids["group_id"],
        decision_type=ids["decision_type"],
        decision_idx=ids["decision_idx"],
        payload_sha256=hashlib.sha256(raw).hexdigest(),
        payload_bytes=len(raw),
        occurred_at=_answer_time(payload),
    )
    db.session.add(event)
    db.session.commit()
    finding = code_finding(event, fallback_reason=fallback_reason_for(event))
    if finding is not None:
        emit(finding, source=catalog.SOURCE_ERROR_CODE)
    return event


def after_response(response):
    """``app.after_request``: log the §6 code of every ``/api/v1/*`` answer that needs one. Never raises."""
    try:
        from flask import request

        path = request.path or ""
        if not path.startswith(codes.API_PREFIX):
            return response
        if request.method == "OPTIONS" and response.status_code == 200:
            return response  # Flask's automatic OPTIONS answer is not an API answer
        body = response.get_json(silent=True) if response.is_json else None
        body = body if isinstance(body, dict) else {}
        title, message, rid = body.get("title"), body.get("message"), body.get("rid")
        classified = codes.classify(path, response.status_code, title, message)
        if not codes.must_log(classified):
            return response
        raw = request.get_data(cache=True) or b""
        payload = request.get_json(silent=True) if raw else None
        record_answer(path, response.status_code, title, message, rid, classified, payload=payload, raw=raw)
    except Exception as exc:  # noqa: BLE001 - never break an answer (Monitoring-Spec §6.9 rule 1)
        report_internal_error("after_response", exc)
    return response


# ------------------------------------------------------------------- /action


def check_action_probability(pi: float, *, learned: bool, context: dict) -> bool:
    """
    A13 inside ``/action`` (Monitoring-Spec §2.1): ``True`` iff ``pi`` is a
    valid probability (learned: finite and within ``[l_min, l_max]`` with a
    1e-12 tolerance; warm-up: exactly 0.5). An invalid ``pi`` raises the red
    A13 alarm; the route then answers with the 50/50 fallback (F-A2).
    Never raises.
    """
    from app.monitoring.checks.decisions import a13_inline_finding, pi_valid

    try:
        from app.hyperparams import get_hyperparams

        hp = get_hyperparams()
        valid = pi_valid(pi, learned=learned, hp=hp)
    except Exception as exc:  # noqa: BLE001
        report_internal_error("check_action_probability", exc)
        try:
            import math

            return math.isfinite(float(pi))
        except (TypeError, ValueError):
            return False
    if not valid:
        try:
            ctx = dict(context or {})
            at = None
            if ctx.get("request_timestamp"):
                from app.clock import to_study_local

                at = to_study_local(ctx["request_timestamp"])
            emit(a13_inline_finding(pi, learned=learned, context=ctx, hp=hp, at=at),
                 source=catalog.SOURCE_INSIDE_ACTION)
        except Exception as exc:  # noqa: BLE001
            report_internal_error("check_action_probability", exc)
    return valid


def _run_guarded(name: str, fn, source: str) -> int:
    try:
        findings = fn()
    except Exception as exc:  # noqa: BLE001
        db.session.rollback()
        report_internal_error(name, exc)
        return 0
    n = 0
    for finding in findings:
        if emit(finding, source=source) is not None:
            n += 1
    return n


def after_action_committed(action_id: int) -> None:
    """A13 (warm-up pi), A8, A15, A17 and the per-decision C9 for a committed decision. Never raises."""
    try:
        from flask import current_app

        from app.hyperparams import get_hyperparams
        from app.models import Action
        from app.monitoring.checks import decisions
        from app.monitoring.checks import update as update_checks

        action = db.session.get(Action, action_id)
        if action is None:
            return
        hp = get_hyperparams()
        streak = int(current_app.config.get("MONITOR_GAME_STREAK_WEEKS", 3))
        checks = (
            ("A13", lambda: decisions.a13_for_action(action, hp)),
            ("A8", lambda: decisions.a8_for_action(action)),
            ("A15", lambda: decisions.a15_for_action(action)),
            ("A17", lambda: decisions.a17_for_action(action, streak=streak)),
            ("C9", lambda: update_checks.c9_for_action(action)),
        )
        for name, fn in checks:
            _run_guarded(f"after_action_committed:{name}", fn, catalog.SOURCE_AFTER_RESPONSE)
    except Exception as exc:  # noqa: BLE001
        report_internal_error("after_action_committed", exc)


# ------------------------------------------------------------------- /update


def _update_request(update_rid: str):
    from app.models import ModelUpdateRequests

    return ModelUpdateRequests.query.filter_by(rid=update_rid).first()


def on_update_received(update_rid: str) -> None:
    """``update_weeks`` bookkeeping and B11 (at the week's first ``/update``) when an update is accepted. Never raises."""
    try:
        from app.monitoring.checks import update as update_checks
        from app.monitoring.checks.common import update_week
        from app.monitoring.checks.uploads import b11_for_week, first_update_of_week

        req = _update_request(update_rid)
        if req is None:
            return
        monday = update_week(req)
        first = first_update_of_week(monday)
        missing = None
        if first is not None and first.id == req.id:
            try:
                findings, missing = b11_for_week(monday, first)
                for finding in findings:
                    emit(finding, source=catalog.SOURCE_AFTER_RESPONSE)
            except Exception as exc:  # noqa: BLE001
                db.session.rollback()
                report_internal_error("on_update_received:B11", exc)
        update_checks.record_received(req, missing)
    except Exception as exc:  # noqa: BLE001
        report_internal_error("on_update_received", exc)


def on_update_finished(update_rid: str, summary: dict) -> None:
    """
    After an update (completed or failed): ``update_weeks``; C3 (the gate's
    rejections), C1 (failure or EB fallback), C9 (fingerprints of the fit),
    B5, C10 (first learning update), L1-L3. Read from the rows the update
    wrote, so the runner finds the same. Never raises.
    """
    try:
        from app.monitoring.checks import diagnostics, uploads
        from app.monitoring.checks import update as update_checks

        req = _update_request(update_rid)
        if req is None:
            return
        try:
            update_checks.record_finished(req)
        except Exception as exc:  # noqa: BLE001
            db.session.rollback()
            report_internal_error("on_update_finished:update_weeks", exc)
        checks = (
            ("C3", lambda: update_checks.c3_for_update(req), catalog.SOURCE_BEFORE_PUBLISH),
            ("C1", lambda: update_checks.c1_for_update(req), catalog.SOURCE_AFTER_RESPONSE),
            ("C9", lambda: update_checks.c9_for_update(req), catalog.SOURCE_AFTER_RESPONSE),
            ("B5", lambda: uploads.b5_for_update(req), catalog.SOURCE_AFTER_RESPONSE),
            ("C10", update_checks.c10_finding, catalog.SOURCE_AFTER_RESPONSE),
            ("L1", lambda: diagnostics.l1_for_update(req), catalog.SOURCE_AFTER_RESPONSE),
            ("L2", lambda: diagnostics.l2_for_update(req), catalog.SOURCE_AFTER_RESPONSE),
            ("L3", lambda: diagnostics.l3_for_update(req), catalog.SOURCE_AFTER_RESPONSE),
        )
        for name, fn, source in checks:
            _run_guarded(f"on_update_finished:{name}", fn, source)
    except Exception as exc:  # noqa: BLE001
        report_internal_error("on_update_finished", exc)


# ------------------------------------------------------------------ start-up


def on_startup_settings_mismatch(detail: dict) -> None:
    """
    C9 at start-up (Hyperparameters.md §2.3): record the red alarm before the
    API refuses to start because the settings differ from the frozen set.
    """
    from app.monitoring.checks.update import startup_finding

    try:
        at = monitoring_now()
        emit(startup_finding(detail, at), source=catalog.SOURCE_STARTUP, raised_at=alarm_time(at))
    except Exception as exc:  # noqa: BLE001
        report_internal_error("on_startup_settings_mismatch", exc)


__all__ = [
    "after_response",
    "record_answer",
    "check_action_probability",
    "after_action_committed",
    "on_update_received",
    "on_update_finished",
    "on_startup_settings_mismatch",
]
