"""
Server-side reward derivation at ``/update`` time (API-Spec §2.4, §5.4, §7.1;
main_new.tex §5.3; implementation contract D6).

For every action not yet paired, the outcome upload is found on the dyad's
``data_uploads`` timeline and the reward is computed. The pairing rules and the
reward formulas belong to the learner and are not repeated here:

- :func:`app.learner.find_outcome_upload` (with :func:`app.learner.action_slot`)
  picks the outcome upload of an action (AYA: the first later upload with the
  opposite slot; CP: the first ``am`` upload on a later date; REL: the first
  upload of a later study week, normally the next Monday's week-closing
  upload);
- :func:`app.learner.compute_reward` turns its ``data`` into the normalized
  reward (stored in ``study_data.reward``) and the native reward (stored in
  ``study_data.outcome["reward_native"]``);
- :func:`app.learner.aya_week_mean_level` gives the mean AYA level of the REL
  decision's week, which the REL reward adds with weight ``w_adh`` when
  ``w_adh > 0`` (main_new.tex eq. (reward-rel-adh)).

Only uploads received before the update started are used (``cutoff`` on
``data_uploads.created_at``; API-Spec §2.4). Actions without a stored context
(the ``no_state`` fallback, ``raw_context`` NULL) cannot enter learning and are
skipped. Re-runs are idempotent: a row that already has a reward is left as
is, an action whose outcome upload has not arrived stays unpaired, and an
action whose reward could not be computed gets ``reward = NULL`` and is retried
by the next run (API-Spec §7.1).
"""

from __future__ import annotations

import datetime
import logging
from collections.abc import Sequence

from app import learner
from app.clock import server_now
from app.extensions import db
from app.models import Action, DataUpload, Group, StudyData
from app.study_calendar import study_week, week_monday

logger = logging.getLogger(__name__)

#: Upload fields recorded in ``study_data.outcome["fields"]`` per agent: the
#: values the D6 reward reads (plus the slot / week that located the upload).
OUTCOME_FIELDS: dict[str, tuple[str, ...]] = {
    "aya_message": ("slot", "previous_med_adherence"),
    "cp_message": ("slot", "daily_diary_completed", "cp_diary_mood"),
    "dyad_game": ("week_in_study", "relationship_quality_aya", "relationship_quality_cp"),
}


def find_outcome_upload(consent_start, action: Action, uploads: Sequence[DataUpload]):
    """
    The outcome upload of ``action`` among the dyad's ``uploads`` (ordered by
    ``(request_timestamp, id)``), or ``None`` if it has not arrived. Delegates
    to the learner's pairing rule (contract D6); the REL decision's study week
    comes from the study calendar (API-Spec §4.1).
    """
    week = None
    if action.decision_type == "dyad_game":
        week = study_week(consent_start, action.request_timestamp.date())
    return learner.find_outcome_upload(
        action.decision_type,
        learner.action_slot(action),
        action.request_timestamp,
        week,
        uploads,
    )


def _outcome_record(decision_type: str, upload: DataUpload, native: float, extra: dict) -> dict:
    data = upload.data or {}
    return {
        "decision_type": decision_type,
        "reward_native": float(native),
        "outcome_upload_id": upload.id,
        "outcome_upload_rid": upload.rid,
        "outcome_upload_timestamp": upload.request_timestamp.isoformat(),
        "fields": {name: data.get(name) for name in OUTCOME_FIELDS.get(decision_type, ())},
        **extra,
    }


def _write_row(existing: StudyData | None, action: Action, outcome: dict, reward: float | None, now) -> None:
    if existing is None:
        db.session.add(
            StudyData(
                group_id=action.group_id,
                decision_idx=action.decision_idx,
                decision_type=action.decision_type,
                action=int(action.action),
                action_prob=float(action.action_prob),
                state=action.state,
                raw_context=action.raw_context,
                outcome=outcome,
                reward=reward,
                request_timestamp=action.request_timestamp,
                derived_at=now,
            )
        )
        return
    existing.action = int(action.action)
    existing.action_prob = float(action.action_prob)
    existing.state = action.state
    existing.raw_context = action.raw_context
    existing.outcome = outcome
    existing.reward = reward
    existing.request_timestamp = action.request_timestamp
    existing.derived_at = now


def derive_study_data(app=None, cutoff: datetime.datetime | None = None) -> int:
    """
    Pair every unpaired action with its outcome upload and write or update its
    ``study_data`` row (contract D6). ``cutoff``: only uploads with
    ``created_at < cutoff`` are used (the update's start time); ``None`` uses
    every upload. Returns the number of rows that received a reward in this
    pass. Requires an application context; ``app`` is accepted for
    compatibility and not used.
    """
    hp = learner.get_hyperparams()
    w_adh = float(hp.w_adh)
    now = server_now()
    finalized = 0

    for group in Group.query.order_by(Group.group_id.asc()).all():
        gid = group.group_id
        consent_start = group.group_info["consent_start_date"]
        actions = (
            Action.query.filter_by(group_id=gid)
            .order_by(Action.request_timestamp.asc(), Action.id.asc())
            .all()
        )
        if not actions:
            continue
        upload_query = DataUpload.query.filter_by(group_id=gid)
        if cutoff is not None:
            upload_query = upload_query.filter(DataUpload.created_at < cutoff)
        uploads = upload_query.order_by(DataUpload.request_timestamp.asc(), DataUpload.id.asc()).all()
        existing_rows = {
            (row.decision_type, row.decision_idx): row
            for row in StudyData.query.filter_by(group_id=gid).all()
        }

        for action in actions:
            if action.raw_context is None:
                continue  # no_state fallback: no decision-time context, cannot enter learning
            existing = existing_rows.get((action.decision_type, action.decision_idx))
            if existing is not None and existing.reward is not None:
                continue  # already finalized; the chosen upload is stable

            try:
                outcome_upload = find_outcome_upload(consent_start, action, uploads)
                if outcome_upload is None:
                    continue  # outcome window not filled yet
                extra: dict = {}
                week_mean = None
                if action.decision_type == "dyad_game" and w_adh > 0:
                    # The REL decision's week: AYA outcomes up to (and
                    # including) the REL outcome upload, received before the
                    # update started.
                    week_mean, n_used = learner.aya_week_mean_level(
                        gid,
                        week_monday(action.request_timestamp),
                        outcome_upload.request_timestamp,
                        received_by=cutoff,
                    )
                    extra["aya_week_mean_level"] = float(week_mean)
                    extra["aya_week_mean_level_n"] = int(n_used)
                reward, native = learner.compute_reward(
                    action.decision_type, outcome_upload.data, aya_week_mean_level=week_mean
                )
                outcome = _outcome_record(action.decision_type, outcome_upload, native, extra)
                _write_row(existing, action, outcome, float(reward), now)
                finalized += 1
            except Exception as exc:  # noqa: BLE001 - one action's failure skips that action only (§7.1)
                logger.error(
                    "[Rewards] Derivation failed for action id=%s (%s): %s",
                    action.id,
                    action.decision_type,
                    type(exc).__name__,
                    exc_info=True,
                )
                _write_row(
                    existing,
                    action,
                    {"decision_type": action.decision_type, "error": f"{type(exc).__name__}: {exc}"},
                    None,
                    now,
                )

        db.session.commit()

    return finalized
