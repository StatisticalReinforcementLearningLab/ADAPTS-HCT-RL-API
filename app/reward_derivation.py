"""
Server-side reward derivation (API-Spec §5.3).

Each action's outcome is computed at /update time by walking forward from the
action's timestamp on the data_uploads timeline and reading the relevant
outcome value from the next scheduled upload of the matching kind:

  - aya_message: next AYA upload (AM decision -> next PM upload; PM decision ->
    next AM upload), ~12 h. 4-tier ordinal reward from previous_med_adherence
    and prompted_by_message.
  - cp_message: next CP-decision upload (next AM, ~24 h). daily_diary_score if
    daily_diary_completed else 0.
  - dyad_game: next dyad_game upload (next week's Monday AM, ~7 days).
    weekly_relationship_score if weekly_survey_completed else 0.

The pairing produces (or updates) one study_data row per action. It is
idempotent across /update re-runs: an action whose outcome window has not yet
filled is left unpaired and picked up by a later /update.
"""

from __future__ import annotations

import datetime

from app.extensions import db
from app.models import Action, DataUpload, Group, StudyData
from app.protocol import compute_reward, outcome_from_snapshot


def _find_outcome_upload(decision_type: str, action: Action, uploads_after: list):
    """The first upload after the action that closes its outcome window."""
    ctx = action.raw_context or {}
    if decision_type == "aya_message":
        target_slot = "pm" if ctx.get("slot") == "am" else "am"
        for up in uploads_after:
            if up.data.get("slot") == target_slot:
                return up
        return None

    if decision_type == "cp_message":
        # CP decides in the morning; the outcome (yesterday's diary completion)
        # is read at the next morning upload.
        for up in uploads_after:
            if up.data.get("slot") == "am":
                return up
        return None

    if decision_type == "dyad_game":
        action_week = int(ctx.get("week_in_study", 0))
        for up in uploads_after:
            if up.data.get("slot") == "am" and int(
                up.data.get("week_in_study", 0)
            ) > action_week:
                return up
        return None

    return None


def derive_study_data(app) -> int:
    """
    Pair every unpaired action with its outcome upload and write/update the
    corresponding study_data row. Returns the number of rows finalized this
    pass.
    """
    now = datetime.datetime.now()
    finalized = 0

    for group in Group.query.order_by(Group.group_id.asc()).all():
        gid = group.group_id
        actions = (
            Action.query.filter_by(group_id=gid)
            .order_by(Action.request_timestamp.asc(), Action.id.asc())
            .all()
        )
        uploads = (
            DataUpload.query.filter_by(group_id=gid)
            .order_by(DataUpload.request_timestamp.asc(), DataUpload.id.asc())
            .all()
        )

        for action in actions:
            existing = StudyData.query.filter_by(
                group_id=gid,
                decision_type=action.decision_type,
                decision_idx=action.decision_idx,
            ).first()
            if existing is not None and existing.reward is not None:
                continue  # already finalized; the chosen upload is stable

            uploads_after = [
                u for u in uploads if u.request_timestamp > action.request_timestamp
            ]
            outcome_upload = _find_outcome_upload(
                action.decision_type, action, uploads_after
            )
            if outcome_upload is None:
                continue  # outcome window not filled yet

            outcome = outcome_from_snapshot(action.decision_type, outcome_upload.data)
            reward = compute_reward(action.decision_type, int(action.action), outcome)
            state = action.state if action.state is not None else []

            if existing is None:
                db.session.add(
                    StudyData(
                        group_id=gid,
                        decision_idx=action.decision_idx,
                        decision_type=action.decision_type,
                        action=int(action.action),
                        action_prob=float(action.action_prob),
                        state=state,
                        raw_context=action.raw_context,
                        outcome=outcome,
                        reward=reward,
                        request_timestamp=action.request_timestamp,
                        derived_at=now,
                    )
                )
            else:
                existing.action = int(action.action)
                existing.action_prob = float(action.action_prob)
                existing.state = state
                existing.raw_context = action.raw_context
                existing.outcome = outcome
                existing.reward = reward
                existing.request_timestamp = action.request_timestamp
                existing.derived_at = now
            finalized += 1

        db.session.commit()

    return finalized
