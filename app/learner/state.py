"""
Raw design state of a decision from the dyad's upload snapshot
(implementation contract D7; main_new.tex §5.4 Table 3 and its definitions;
API-Spec §4.1).

Snapshot mapping (contract D7):

- ``eng_*`` = ``*_app_engagement``; ``"miss"`` is system-side missingness and
  is imputed by last observation carried forward over the dyad's earlier
  uploads (main_new.tex §5.4 "Missingness encoding"), else level 1
  (API-Spec §4.1: "the learner substitutes level 1").
- ``burden_*`` = ``*_app_burden``; ``missrate_cp`` = ``cp_missing_rate_7d``
  (never missing by contract; LOCF, else 0, if a host sends ``"miss"``).
- ``game`` = ``current_game_on``; ``prior_game`` = ``prior_game_action``;
  ``"miss"`` -> 0.
- ``prior_aya_level`` = level of ``previous_med_adherence`` (2/1/0).
- ``aya_mood``, ``aya_phys``, ``cp_mood``, ``relq_aya``, ``relq_cp``: the raw
  value, ``"miss"`` -> 0 (participant-side missingness is level 0 of the
  variable, eq. (miss-encoding)).
- ``relq_dyad``: the completion-weighted dyad summary (eq.
  (reward-rel-quality)), 0 if neither role responded.
- ``aya_week_mean_level``: the mean AYA level over the dyad's AYA decisions of
  the previous calendar week (Mon-Sun before the decision's week) whose
  outcome upload exists by decision time; 0 if none.
- ``eng_avg``, ``burden_avg``, ``dsum_avg``: means over the AYA and CP values
  (``*_diary_summary`` ``"miss"`` -> 0).

The returned dict is stored as ``actions.raw_context["design_state"]`` so the
weekly fit never re-reads uploads for past decisions.
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any

from app.learner import features as F
from app.learner.rewards import (
    _number,
    action_slot,
    aya_level,
    find_outcome_upload,
    is_missing,
    rel_quality,
    upload_time,
)

# How many earlier uploads are fetched per query when looking back for LOCF.
_LOCF_BATCH = 64


def _locf(group_id: str, upload: Any, field: str) -> tuple[float | None, int | None]:
    """
    Last non-missing value of ``field`` in the dyad's uploads strictly before
    ``upload`` on the ``(request_timestamp, id)`` timeline. Returns
    ``(value, data_uploads.id)`` or ``(None, None)``.
    """
    from sqlalchemy import and_, or_

    from app.models import DataUpload

    ts = upload_time(upload)
    up_id = getattr(upload, "id", None)
    cond = DataUpload.request_timestamp < ts
    if up_id is not None:
        cond = or_(cond, and_(DataUpload.request_timestamp == ts, DataUpload.id < up_id))
    query = (
        DataUpload.query.filter(DataUpload.group_id == group_id, cond)
        .order_by(DataUpload.request_timestamp.desc(), DataUpload.id.desc())
    )
    offset = 0
    while True:
        batch = query.offset(offset).limit(_LOCF_BATCH).all()
        if not batch:
            return None, None
        for row in batch:
            value = _number((row.data or {}).get(field))
            if value is not None:
                return value, row.id
        offset += len(batch)


def _system_value(
    group_id: str, upload: Any, data: Mapping, field: str, name: str, default: float, imputed: dict
) -> float:
    """A system-side coordinate: the value, else LOCF, else ``default`` (recorded in ``imputed``)."""
    value = _number(data.get(field))
    if value is not None:
        return value
    prev, prev_id = _locf(group_id, upload, field)
    if prev is not None:
        imputed[name] = f"locf:{field}@upload_id={prev_id}"
        return prev
    imputed[name] = f"default:{field}={default:g}"
    return float(default)


def _participant_value(data: Mapping, field: str, name: str, imputed: dict) -> float:
    """A participant-side coordinate: the value, ``"miss"`` -> 0 (eq. (miss-encoding))."""
    value = _number(data.get(field))
    if value is None:
        imputed[name] = f"miss:{field}->0"
        return 0.0
    return value


def _binary_value(data: Mapping, field: str, name: str, imputed: dict) -> float:
    """A game indicator: 0/1, ``"miss"`` -> 0 (contract D7)."""
    value = _number(data.get(field))
    if value is None:
        imputed[name] = f"miss:{field}->0"
        return 0.0
    return value


def aya_week_mean_level(
    group_id: str,
    week_monday: datetime.date,
    as_of: datetime.datetime,
    *,
    received_by: datetime.datetime | None = None,
) -> tuple[float, int]:
    """
    Mean AYA level (contract D6; main_new.tex eq. (reward-rel-adh)) over the
    dyad's ``aya_message`` decisions stamped in the calendar week starting
    ``week_monday`` whose outcome upload (stamped at or before ``as_of`` and,
    if given, received by server time ``received_by``) exists. Returns
    ``(mean, n_decisions_used)``; the mean is 0 when no decision qualifies.

    Used for the REL state (previous week, ``as_of`` = decision time) and for
    the REL reward when ``rel_reward.w_adh > 0`` (the REL decision's week).
    """
    from app.models import Action, DataUpload

    start = datetime.datetime.combine(week_monday, datetime.time.min)
    end = start + datetime.timedelta(days=7)
    actions = (
        Action.query.filter(
            Action.group_id == group_id,
            Action.decision_type == F.AYA,
            Action.request_timestamp >= start,
            Action.request_timestamp < end,
            Action.request_timestamp < as_of,
        )
        .order_by(Action.request_timestamp.asc(), Action.id.asc())
        .all()
    )
    if not actions:
        return 0.0, 0
    q = DataUpload.query.filter(
        DataUpload.group_id == group_id,
        DataUpload.request_timestamp > start,
        DataUpload.request_timestamp <= as_of,
    )
    if received_by is not None:
        q = q.filter(DataUpload.created_at <= received_by)
    uploads = q.order_by(DataUpload.request_timestamp.asc(), DataUpload.id.asc()).all()
    levels: list[int] = []
    for action in actions:
        outcome = find_outcome_upload(F.AYA, action_slot(action), action.request_timestamp, None, uploads)
        if outcome is not None:
            levels.append(aya_level((outcome.data or {}).get("previous_med_adherence")))
    if not levels:
        return 0.0, 0
    return float(sum(levels)) / float(len(levels)), len(levels)


def build_design_state(
    group_id: str,
    decision_type: str,
    upload: Any,
    decision_time: datetime.datetime,
    *,
    hp=None,
) -> dict:
    """
    Raw design state of contract D7 from the snapshot ``upload`` (a
    ``DataUpload`` row) for a decision at ``decision_time``. Returns

    ``{"decision_type", "names", "raw", "upload_id", "upload_rid",
    "upload_timestamp", "imputed": {name: note}, "clipped": [names],
    "slot", "week_in_study", "day_in_study"}``

    plus, for REL, ``"aya_week_mean_level_n"`` (decisions averaged). Requires
    an application context (LOCF and the REL mean read the database).
    """
    F.check_decision_type(decision_type)
    if hp is None:
        from app.hyperparams import get_hyperparams

        hp = get_hyperparams()
    data = getattr(upload, "data", None)
    if not isinstance(data, Mapping):
        raise ValueError("upload has no data snapshot.")
    imputed: dict[str, str] = {}
    extra: dict[str, Any] = {}

    if decision_type == F.AYA:
        raw = [
            1.0,
            _system_value(group_id, upload, data, "aya_app_engagement", "eng_aya", 1.0, imputed),
            _system_value(group_id, upload, data, "aya_app_burden", "burden_aya", 0.0, imputed),
            _binary_value(data, "current_game_on", "game", imputed),
            float(aya_level(data.get("previous_med_adherence"))),
            _participant_value(data, "aya_diary_mood", "aya_mood", imputed),
            _participant_value(data, "aya_diary_physical", "aya_phys", imputed),
            _participant_value(data, "relationship_quality_aya", "relq_aya", imputed),
        ]
        if is_missing(data.get("previous_med_adherence")):
            imputed["prior_aya_level"] = "miss:previous_med_adherence->0"
    elif decision_type == F.CP:
        raw = [
            1.0,
            _system_value(group_id, upload, data, "cp_app_engagement", "eng_cp", 1.0, imputed),
            _system_value(group_id, upload, data, "cp_app_burden", "burden_cp", 0.0, imputed),
            _binary_value(data, "current_game_on", "game", imputed),
            _system_value(group_id, upload, data, "cp_missing_rate_7d", "missrate_cp", 0.0, imputed),
            _participant_value(data, "cp_diary_mood", "cp_mood", imputed),
            _participant_value(data, "relationship_quality_cp", "relq_cp", imputed),
        ]
    else:
        relq_dyad = rel_quality(
            data.get("relationship_quality_aya"), data.get("relationship_quality_cp"), hp.respondent_weights
        )
        if is_missing(data.get("relationship_quality_aya")) and is_missing(data.get("relationship_quality_cp")):
            imputed["relq_dyad"] = "miss:relationship_quality_aya,relationship_quality_cp->0"
        from app.study_calendar import week_monday

        prev_monday = week_monday(decision_time) - datetime.timedelta(days=7)
        mean_level, n_used = aya_week_mean_level(group_id, prev_monday, decision_time)
        extra["aya_week_mean_level_n"] = n_used
        eng_aya = _system_value(group_id, upload, data, "aya_app_engagement", "eng_aya", 1.0, imputed)
        eng_cp = _system_value(group_id, upload, data, "cp_app_engagement", "eng_cp", 1.0, imputed)
        b_aya = _system_value(group_id, upload, data, "aya_app_burden", "burden_aya", 0.0, imputed)
        b_cp = _system_value(group_id, upload, data, "cp_app_burden", "burden_cp", 0.0, imputed)
        d_aya = _participant_value(data, "aya_diary_summary", "dsum_aya", imputed)
        d_cp = _participant_value(data, "cp_diary_summary", "dsum_cp", imputed)
        raw = [
            1.0,
            relq_dyad,
            mean_level,
            0.5 * (eng_aya + eng_cp),
            0.5 * (b_aya + b_cp),
            _binary_value(data, "prior_game_action", "prior_game", imputed),
            0.5 * (d_aya + d_cp),
        ]

    raw = [float(v) for v in raw]
    _, clipped = F.standardize(decision_type, raw)
    ts = upload_time(upload)
    state = {
        "decision_type": decision_type,
        "names": list(F.NAMES[decision_type]),
        "raw": raw,
        "upload_id": getattr(upload, "id", None),
        "upload_rid": getattr(upload, "rid", None),
        "upload_timestamp": ts.isoformat(),
        "imputed": imputed,
        "clipped": clipped,
        "slot": data.get("slot"),
        "week_in_study": data.get("week_in_study"),
        "day_in_study": data.get("day_in_study"),
    }
    state.update(extra)
    return state


__all__ = ["aya_week_mean_level", "build_design_state"]
