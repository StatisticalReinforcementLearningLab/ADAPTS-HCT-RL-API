"""
Full dataset + decision-state snapshots before each model update.

Writes JSON files under REPRO_SNAPSHOT_ROOT/<update_id>/:
  - study_data.json
  - actions.json (includes `state` used at decision time)
  - groups.json
  - metadata.json
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Any

from app.extensions import db
from app.models import (
    Action,
    Group,
    ModelUpdateRequests,
    StudyData,
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


def save_pre_update_repro_snapshot(app, update_id: str, model_parameters_id: int | None) -> str | None:
    """
    Persist a full copy of study_data, actions (with state), and groups
    before the learner runs. Returns snapshot directory or None if disabled.
    """
    if not app.config.get("SAVE_UPDATE_REPRO_SNAPSHOTS", True):
        return None

    root = app.config.get("REPRO_SNAPSHOT_ROOT", "repro_snapshots")
    out_dir = os.path.abspath(os.path.join(root, update_id))
    os.makedirs(out_dir, exist_ok=True)

    study_rows = StudyData.query.order_by(
        StudyData.group_id.asc(),
        StudyData.decision_idx.asc(),
    ).all()
    action_rows = Action.query.order_by(
        Action.group_id.asc(),
        Action.decision_idx.asc(),
    ).all()
    group_rows = Group.query.order_by(Group.group_id.asc()).all()
    update_rows = ModelUpdateRequests.query.order_by(
        ModelUpdateRequests.request_timestamp.asc()
    ).all()

    study_payload = []
    for r in study_rows:
        study_payload.append(
            {
                "id": r.id,
                "group_id": r.group_id,
                "decision_idx": r.decision_idx,
                "decision_type": r.decision_type,
                "action": r.action,
                "action_prob": r.action_prob,
                "state": r.state,
                "raw_context": r.raw_context,
                "outcome": r.outcome,
                "reward": r.reward,
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
                "state": r.state,
                "raw_context": r.raw_context,
                "random_state": r.random_state,
                "model_parameters_id": r.model_parameters_id,
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
                "group_info": r.group_info,
                "warmup": bool(r.warmup),
                "created_at": r.created_at,
            }
        )

    updates_payload = []
    for r in update_rows:
        updates_payload.append(
            {
                "id": r.id,
                "update_id": r.update_id,
                "status": r.status,
                "callback_url": r.callback_url,
                "request_timestamp": r.request_timestamp,
                "created_at": r.created_at,
                "completed_at": r.completed_at,
                "error_message": r.error_message,
            }
        )

    meta = {
        "update_id": update_id,
        "model_parameters_id": model_parameters_id,
        "saved_at": datetime.datetime.now().isoformat(),
        "study_data_count": len(study_payload),
        "actions_count": len(action_payload),
        "groups_count": len(group_payload),
        "updates_count": len(updates_payload),
    }

    total = 0
    total += _write_json(os.path.join(out_dir, "study_data.json"), study_payload)
    total += _write_json(os.path.join(out_dir, "actions.json"), action_payload)
    total += _write_json(os.path.join(out_dir, "groups.json"), group_payload)
    total += _write_json(
        os.path.join(out_dir, "model_update_requests.json"), updates_payload
    )
    total += _write_json(os.path.join(out_dir, "metadata.json"), meta)

    row = UpdateReproducibilitySnapshot(
        update_id=update_id,
        model_parameters_id=model_parameters_id,
        snapshot_dir=out_dir,
        study_data_count=len(study_payload),
        actions_count=len(action_payload),
        groups_count=len(group_payload),
        total_bytes=total,
    )
    db.session.add(row)
    db.session.commit()

    return out_dir
