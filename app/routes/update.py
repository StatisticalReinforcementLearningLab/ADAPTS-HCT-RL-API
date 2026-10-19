"""
``POST /api/v1/update`` — start the weekly re-fit (API-Spec §2.4, §5.6, §7.1;
Monitoring-Spec §6.5, §7 F-U3/F-U4; implementation contract D13, D14, D21).

Answers (envelope only; the ``rid`` is the update handle):

- ``202 Update Accepted`` ``Model update started.``: the
  ``model_update_requests`` row (``status = "processing"``, the request
  timestamp in Detroit local time, its study week Monday) is committed, then
  ``monitoring.on_update_received`` runs and the fit starts in a background
  thread (inline when ``UPDATE_RUN_INLINE``, as in the tests).
- ``400 Invalid Parameter``: ``timestamp is required.``; a timestamp that is
  not a string or not ISO-8601 is also rejected.
- ``503 Service Unavailable`` ``Could not start the update; retry.``: the row
  could not be written (nothing recorded).
- ``500 Internal Error`` ``Internal server error.``

The fit (:func:`process_update_request`): ``FAULT_UPDATE_HANG_SECONDS`` (testing
only), optional backup of every table, pre-update reproducibility snapshot,
reward derivation from uploads received before the fit started, then
``app.learner.run_weekly_update`` (or the legacy learner's ``update``). The
row ends ``completed`` (with ``completed_at`` and ``published_at_study`` =
request timestamp + run duration, contract D13) or ``failed`` (with
``error_message``); ``monitoring.on_update_finished`` then gets the summary.
"""

from __future__ import annotations

import datetime
import logging
import os
import shutil
import time
from threading import Thread

from flask import Blueprint, current_app

from app import learner
from app.algorithms.base import RLAlgorithm
from app.algorithms.inf_lsvi_eb import InfLsviEbAlgorithm
from app.clock import server_now
from app.extensions import db
from app.models import ModelParameters, ModelUpdateRequests, StudyData, dump_tables_to_csv
from app.repro_snapshot import save_pre_update_repro_snapshot
from app.reward_derivation import derive_study_data
from app.routes.envelope import (
    call_monitoring,
    envelope,
    new_rid,
    parse_timestamp,
    read_json_body,
    timestamp_problem,
)
from app.study_calendar import week_monday

logger = logging.getLogger(__name__)

update_blueprint = Blueprint("update", __name__)

MSG_ACCEPTED = "Model update started."
MSG_ENQUEUE = "Could not start the update; retry."

STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

#: Learner summary statuses that mark the request failed (API-Spec §7.1).
_FAILED_SUMMARY_STATUSES = frozenset({"failed", "error"})


def backup_tables(app) -> str:
    """
    Back up every table (contract §2: all models, quarantine excluded) as CSV
    into ``<BACKUP_ROOT>/<server time>/`` and zip it. Returns the zip path.
    """
    root = app.config.get("BACKUP_ROOT", "backups")
    backup_dir = os.path.join(root, server_now().strftime("%Y%m%d_%H%M%S_%f"))
    with app.app_context():
        dump_tables_to_csv(backup_dir, include_quarantine=False)
    shutil.make_archive(backup_dir, "zip", backup_dir)
    shutil.rmtree(backup_dir)
    return f"{backup_dir}.zip"


def check_fields(data) -> tuple[bool, str]:
    """
    Validate the /update request (Monitoring-Spec §6.5; API-Spec §2.4:
    ``timestamp`` missing or malformed -> 400).
    """
    if not isinstance(data, dict) or "timestamp" not in data:
        return False, "timestamp is required."
    problem = timestamp_problem(data["timestamp"])
    if problem is not None:
        return False, problem
    return True, ""


def _legacy_update(app, rid: str, rl_algorithm: RLAlgorithm) -> dict:
    """The pre-``inf_lsvi_eb`` update path (contract D1): fit over all ``study_data`` with a reward."""
    current_params = (
        ModelParameters.query.filter(ModelParameters.snapshot_type.is_(None))
        .order_by(ModelParameters.timestamp.desc(), ModelParameters.id.desc())
        .first()
    )
    rows = (
        StudyData.query.filter(StudyData.reward.isnot(None))
        .order_by(StudyData.decision_type.asc(), StudyData.group_id.asc(), StudyData.decision_idx.asc())
        .all()
    )
    records = []
    current_index: dict[str, int] = {}
    for row in rows:
        ctx = row.raw_context or {}
        agent_idx = int(ctx.get("agent_decision_index", row.decision_idx + 1))
        records.append(
            {
                "group_id": row.group_id,
                "decision_idx": row.decision_idx,
                "decision_type": row.decision_type,
                "agent_decision_index": agent_idx,
                "state": row.state,
                "action": row.action,
                "reward": row.reward,
                "raw_context": ctx,
                "outcome": row.outcome,
            }
        )
        current_index[row.decision_type] = max(current_index.get(row.decision_type, 0), agent_idx)

    old = {"probability_of_action": current_params.probability_of_action if current_params else 0.5}
    ok, new_parameters = rl_algorithm.update(old, {"records": records, "current_index": current_index})
    if not ok:
        raise RuntimeError("Model update failed.")
    db.session.add(ModelParameters(new_parameters["probability_of_action"]))
    db.session.commit()
    return {"status": STATUS_COMPLETED, "agents": {}, "n_records": len(records)}


def _finish(rid: str, status: str, error: str | None, published_at_study) -> None:
    row = ModelUpdateRequests.query.filter_by(rid=rid).first()
    if row is None:
        return
    row.status = status
    row.completed_at = server_now()
    row.error_message = None if error is None else str(error)[:1024]
    if published_at_study is not None:
        row.published_at_study = published_at_study
    db.session.commit()


def process_update_request(app, rid: str, rl_algorithm: RLAlgorithm | None = None) -> dict:
    """
    Run one accepted update (API-Spec §2.4 "Behavior once accepted"). Never
    raises: the outcome is written to ``model_update_requests`` (§7.1) and
    handed to ``monitoring.on_update_finished``. Returns the summary.
    """
    summary: dict = {"status": STATUS_FAILED, "agents": {}}
    with app.app_context():
        rl_algorithm = rl_algorithm if rl_algorithm is not None else app.rl_algorithm
        try:
            row = ModelUpdateRequests.query.filter_by(rid=rid).first()
            if row is None:
                raise RuntimeError(f"model_update_requests row {rid} not found")
            request_timestamp = row.request_timestamp
            started_wall = server_now()
            row.started_at = started_wall
            db.session.commit()
            summary["study_week_monday"] = row.study_week_monday

            from app import fault_setting

            hang = float(fault_setting("FAULT_UPDATE_HANG_SECONDS", app) or 0.0)
            if hang > 0:
                logger.warning("[Update] FAULT_UPDATE_HANG_SECONDS: sleeping %.3fs (rid=%s)", hang, rid)
                time.sleep(hang)

            if app.config.get("BACKUP_DATABASE"):
                logger.info("[Update] Database backed up to: %s", backup_tables(app))

            current_policy = (
                ModelParameters.query.filter(ModelParameters.snapshot_type.is_(None))
                .order_by(ModelParameters.timestamp.desc(), ModelParameters.id.desc())
                .first()
            )
            snap_dir = save_pre_update_repro_snapshot(app, rid, current_policy.id if current_policy else None)
            if snap_dir:
                logger.info("[Update] Pre-update reproducibility snapshot: %s", snap_dir)

            # Rewards from uploads received before the fit started (API-Spec §2.4).
            n_derived = derive_study_data(app, cutoff=started_wall)
            logger.info("[Update] rid=%s derived %d study_data rows", rid, n_derived)

            if isinstance(rl_algorithm, InfLsviEbAlgorithm):
                result = learner.run_weekly_update(rid, request_timestamp, started_wall)
                summary = dict(result or {})
            else:
                summary = _legacy_update(app, rid, rl_algorithm)
            summary.setdefault("study_week_monday", row.study_week_monday)
            summary["n_rewards_derived"] = n_derived

            status = str(summary.get("status") or STATUS_COMPLETED).lower()
            if status in _FAILED_SUMMARY_STATUSES:
                error = summary.get("error") or summary.get("error_message") or f"update {status}"
                _finish(rid, STATUS_FAILED, error, None)
                logger.info("[Update] rid=%s failed: %s", rid, error)
            else:
                published = summary.get("published_at_study")
                if not isinstance(published, datetime.datetime):
                    published = request_timestamp + (server_now() - started_wall)
                _finish(rid, STATUS_COMPLETED, None, published)
                logger.info("[Update] rid=%s completed.", rid)

        except Exception as exc:  # noqa: BLE001 - the outcome goes to the table, never up
            db.session.rollback()
            logger.error("[Update] rid=%s failed: %s", rid, exc, exc_info=True)
            summary = {
                **summary,
                "status": STATUS_FAILED,
                "error": f"{type(exc).__name__}: {exc}",
            }
            try:
                _finish(rid, STATUS_FAILED, summary["error"], None)
            except Exception:  # noqa: BLE001
                db.session.rollback()
                logger.error("[Update] rid=%s could not record the failure.", rid, exc_info=True)

        call_monitoring("on_update_finished", rid, summary)
    return summary


@update_blueprint.route("/update", methods=["POST"])
def update_model():
    """
    Start the weekly update (API-Spec §2.4). Asynchronous with no callback:
    monitoring observes completion through ``model_update_requests``, keyed by
    this call's ``rid``.
    """
    rid = new_rid()
    try:
        data = read_json_body()

        ok, error_message = check_fields(data)
        if not ok:
            return envelope(400, "Invalid Parameter", error_message, rid)

        request_timestamp = parse_timestamp(data["timestamp"])
        rl_algorithm = current_app.rl_algorithm
        logger.info("[Update] rid=%s request_timestamp=%s", rid, request_timestamp.isoformat())

        # Record the request. On failure nothing is recorded: 503, retry.
        try:
            db.session.add(
                ModelUpdateRequests(
                    rid,
                    request_timestamp,
                    status=STATUS_PROCESSING,
                    study_week_monday=week_monday(request_timestamp),
                )
            )
            db.session.commit()
        except Exception as exc:  # noqa: BLE001
            db.session.rollback()
            logger.error("[Update] Could not record rid=%s: %s", rid, type(exc).__name__)
            return envelope(503, "Service Unavailable", MSG_ENQUEUE, rid)

        call_monitoring("on_update_received", rid)

        app = current_app._get_current_object()
        if app.config.get("UPDATE_RUN_INLINE"):
            # Tests: a background thread cannot share the in-memory SQLite
            # connection with the request thread.
            process_update_request(app, rid, rl_algorithm)
        else:
            try:
                Thread(
                    target=process_update_request,
                    args=(app, rid, rl_algorithm),
                    name=f"update-{rid}",
                ).start()
            except Exception as exc:  # noqa: BLE001
                logger.error("[Update] Could not start the update thread: %s", type(exc).__name__)
                _finish(rid, STATUS_FAILED, "could not start the update thread", None)
                return envelope(503, "Service Unavailable", MSG_ENQUEUE, rid)

        return envelope(202, "Update Accepted", MSG_ACCEPTED, rid)

    except Exception:  # noqa: BLE001 - last resort before enqueue
        db.session.rollback()
        logger.error("[Update] Unexpected error rid=%s", rid, exc_info=True)
        return envelope(500, "Internal Error", "Internal server error.", rid)
