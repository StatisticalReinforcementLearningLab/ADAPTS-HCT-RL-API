"""
Full dataset + decision-state snapshots before each model update.

Writes JSON files under REPRO_SNAPSHOT_ROOT/<rid>/ (API-Spec §2.4, §5.9):
  - data_uploads.json (the upstream timeline study_data is derived from)
  - actions.json (includes `state`, `raw_context` with the design state, the
    policy columns and the sampler cursors used at decision time)
  - groups.json
  - group_history.json (every /register_group call, Monitoring-Spec §9.6, so
    consent dates as of any time can be replayed)
  - model_update_requests.json
  - metadata.json
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Any

from app.extensions import db
from app.clock import server_now
from app.models import (
    Action,
    DataUpload,
    Group,
    GroupHistory,
    ModelUpdateRequests,
    UpdateReproducibilitySnapshot,
)


def _json_default(obj: Any):
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _write_json(path: str, data: Any) -> int:
    text = json.dumps(data, indent=2, default=_json_default, sort_keys=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return len(text.encode("utf-8"))


def save_pre_update_repro_snapshot(app, rid: str, model_parameters_id: int | None) -> str | None:
    """
    Persist a full copy of data_uploads, actions (with state), groups and
    group_history before the learner runs. Returns snapshot directory or None
    if disabled. ``rid`` is the /update call's request id (API-Spec §5.9).
    """
    if not app.config.get("SAVE_UPDATE_REPRO_SNAPSHOTS", True):
        return None

    root = app.config.get("REPRO_SNAPSHOT_ROOT", "repro_snapshots")
    out_dir = os.path.abspath(os.path.join(root, rid))
    os.makedirs(out_dir, exist_ok=True)

    upload_rows = DataUpload.query.order_by(
        DataUpload.group_id.asc(),
        DataUpload.request_timestamp.asc(),
        DataUpload.id.asc(),
    ).all()
    action_rows = Action.query.order_by(
        Action.group_id.asc(),
        Action.decision_idx.asc(),
    ).all()
    group_rows = Group.query.order_by(Group.group_id.asc()).all()
    history_rows = GroupHistory.query.order_by(GroupHistory.id.asc()).all()
    update_rows = ModelUpdateRequests.query.order_by(
        ModelUpdateRequests.request_timestamp.asc()
    ).all()

    upload_payload = []
    for r in upload_rows:
        upload_payload.append(
            {
                "id": r.id,
                "rid": r.rid,
                "group_id": r.group_id,
                "data": r.data,
                "request_timestamp": r.request_timestamp,
                "created_at": r.created_at,
            }
        )

    action_payload = []
    for r in action_rows:
        action_payload.append(
            {
                "id": r.id,
                "group_id": r.group_id,
                "rid": r.rid,
                "decision_idx": r.decision_idx,
                "decision_type": r.decision_type,
                "action": r.action,
                "action_prob": r.action_prob,
                "is_warmup": bool(r.is_warmup),
                "warmup_reason": r.warmup_reason,
                "state": r.state,
                "raw_context": r.raw_context,
                "random_state": r.random_state,
                "model_parameters_id": r.model_parameters_id,
                "model_param": r.model_param,
                "policy_week": r.policy_week,
                "policy_version_id": r.policy_version_id,
                "hyperparams_hash": r.hyperparams_hash,
                "mu": r.mu,
                "v": r.v,
                "upload_id": r.upload_id,
                "request_timestamp": r.request_timestamp,
                "timestamp": r.timestamp,
            }
        )

    group_payload = []
    for r in group_rows:
        group_payload.append(
            {
                "id": r.id,
                "group_id": r.group_id,
                "rid": r.rid,
                "group_info": r.group_info,
                "created_at": r.created_at,
            }
        )

    history_payload = [
        {
            "id": r.id,
            "group_id": r.group_id,
            "member_list": r.member_list,
            "consent_start_date": r.consent_start_date,
            "consent_end_date": r.consent_end_date,
            "rid": r.rid,
            "recorded_at": r.recorded_at,
        }
        for r in history_rows
    ]

    updates_payload = []
    for r in update_rows:
        updates_payload.append(
            {
                "id": r.id,
                "rid": r.rid,
                "status": r.status,
                "request_timestamp": r.request_timestamp,
                "created_at": r.created_at,
                "completed_at": r.completed_at,
                "error_message": r.error_message,
                "study_week_monday": r.study_week_monday,
                "started_at": r.started_at,
                "published_at_study": r.published_at_study,
            }
        )

    meta = {
        "rid": rid,
        "model_parameters_id": model_parameters_id,
        "saved_at": server_now().isoformat(),
        "data_uploads_count": len(upload_payload),
        "actions_count": len(action_payload),
        "groups_count": len(group_payload),
        "group_history_count": len(history_payload),
        "updates_count": len(updates_payload),
    }

    total = 0
    total += _write_json(os.path.join(out_dir, "data_uploads.json"), upload_payload)
    total += _write_json(os.path.join(out_dir, "actions.json"), action_payload)
    total += _write_json(os.path.join(out_dir, "groups.json"), group_payload)
    total += _write_json(os.path.join(out_dir, "group_history.json"), history_payload)
    total += _write_json(
        os.path.join(out_dir, "model_update_requests.json"), updates_payload
    )
    total += _write_json(os.path.join(out_dir, "metadata.json"), meta)

    row = UpdateReproducibilitySnapshot(
        rid=rid,
        model_parameters_id=model_parameters_id,
        snapshot_dir=out_dir,
        data_uploads_count=len(upload_payload),
        actions_count=len(action_payload),
        groups_count=len(group_payload),
        total_bytes=total,
    )
    db.session.add(row)
    db.session.commit()

    return out_dir
