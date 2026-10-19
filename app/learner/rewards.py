"""
Rewards of the ``inf_lsvi_eb`` learner and the pairing of a decision with the
upload that closes its outcome (implementation contract D6; main_new.tex §5.3,
eqs. (reward-aya), (reward-cp), (reward-rel-quality), (reward-rel-adh),
(reward-rel); API-Spec §2.3, §4.1).

Every reward is bounded with non-response at its worst level, and the learner
uses it divided by the agent's native range maximum ``r_max`` (AYA 2, CP 5,
REL ``5 + 2 w_adh``) when ``rewards.normalize`` is on (the default), so every
reward enters the working model on ``[0, 1]``.

Pairing (contract D6), on the dyad's upload timeline, among uploads stamped
strictly after the decision:

- AYA: the first upload whose ``slot`` is opposite to the decision's slot
  (AM decision -> that evening's ``pm`` upload; PM decision -> the next ``am``
  upload, which for Sunday PM is the Monday week-closing upload).
- CP: the first ``am`` upload on a calendar date after the decision's date
  (its ``cp_diary_mood`` is the diary of the decision's evening).
- REL: the first upload whose ``week_in_study`` exceeds the decision's study
  week (normally the next Monday's week-closing upload).

:func:`find_outcome_upload` is the single implementation of this rule; the
``/update`` reward derivation (``app.reward_derivation``) and the learner's
state builder both call it.
"""

from __future__ import annotations

import datetime
import math
from collections.abc import Mapping, Sequence
from typing import Any

from app.learner.features import AYA, CP, REL, check_decision_type

MISSING_TOKEN = "miss"
SLOT_AM = "am"
SLOT_PM = "pm"
DEFAULT_NOON_HOUR = 12


def is_missing(value: Any) -> bool:
    """``True`` for the literal ``"miss"`` and JSON ``null`` (API-Spec §4)."""
    return value is None or (isinstance(value, str) and value.strip().lower() == MISSING_TOKEN)


def _number(value: Any) -> float | None:
    """A finite number from an upload field, else ``None`` (``"miss"``, ``null``, garbage)."""
    if is_missing(value) or isinstance(value, (dict, list)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def aya_level(previous_med_adherence) -> int:
    """
    AYA reward level (main_new.tex eq. (reward-aya); contract D6): ``2`` if the
    dose was reported taken (``1``), ``1`` if reported missed (``0``), ``0`` if
    there is no usable report (``"miss"``, ``null`` or anything else).
    """
    value = _number(previous_med_adherence)
    if value == 1.0:
        return 2
    if value == 0.0:
        return 1
    return 0


def rel_quality(relq_aya: Any, relq_cp: Any, weights: Sequence[float]) -> float:
    """
    Completion-weighted relationship quality (main_new.tex eq.
    (reward-rel-quality)): ``(w_A 1_A q_A + w_C 1_C q_C) / (w_A 1_A + w_C 1_C)``
    if at least one respondent completed the survey, else ``0``.
    """
    w_a, w_c = float(weights[0]), float(weights[1])
    q_a, q_c = _number(relq_aya), _number(relq_cp)
    num = 0.0
    den = 0.0
    if q_a is not None:
        num += w_a * q_a
        den += w_a
    if q_c is not None:
        num += w_c * q_c
        den += w_c
    return num / den if den > 0.0 else 0.0


def cp_native_reward(outcome_upload_data: Mapping) -> float:
    """CP reward (eq. (reward-cp)): ``cp_diary_mood`` if the diary was completed and the mood is reported, else 0."""
    completed = outcome_upload_data.get("daily_diary_completed") is True
    mood = _number(outcome_upload_data.get("cp_diary_mood"))
    return float(mood) if completed and mood is not None else 0.0


def compute_reward(
    decision_type: str,
    outcome_upload_data: Mapping,
    *,
    aya_week_mean_level: float | None = None,
    hp=None,
) -> tuple[float, float]:
    """
    Reward of contract D6 from the outcome upload's ``data``; returns
    ``(normalized, native)``.

    - AYA: native = level in {0, 1, 2} of ``previous_med_adherence``.
    - CP: native = ``cp_diary_mood`` if ``daily_diary_completed`` and the mood
      is not missing, else 0.
    - REL: native = relationship quality (eq. (reward-rel-quality)) plus
      ``w_adh`` times ``aya_week_mean_level``, the mean AYA level of the REL
      decision's study week (required when ``w_adh > 0``).

    ``normalized = native / r_max`` when ``rewards.normalize`` is on (else the
    native value). ``hp`` defaults to the running app's hyperparameters.
    """
    check_decision_type(decision_type)
    if not isinstance(outcome_upload_data, Mapping):
        raise ValueError("outcome_upload_data must be the upload's data mapping.")
    if hp is None:
        from app.hyperparams import get_hyperparams

        hp = get_hyperparams()
    divisor = hp.agent(decision_type).reward_divisor
    if decision_type == AYA:
        native = float(aya_level(outcome_upload_data.get("previous_med_adherence")))
    elif decision_type == CP:
        native = cp_native_reward(outcome_upload_data)
    else:
        native = rel_quality(
            outcome_upload_data.get("relationship_quality_aya"),
            outcome_upload_data.get("relationship_quality_cp"),
            hp.respondent_weights,
        )
        if hp.w_adh != 0.0:
            if aya_week_mean_level is None:
                raise ValueError("the REL reward needs aya_week_mean_level when rel_reward.w_adh > 0.")
            native += hp.w_adh * float(aya_week_mean_level)
    return native / divisor, native


# ------------------------------------------------------------------ pairing


def _field(upload: Any, name: str) -> Any:
    if isinstance(upload, Mapping):
        return upload.get(name)
    return getattr(upload, name, None)


def upload_data(upload: Any) -> Mapping:
    """The ``data`` mapping of a ``DataUpload`` row or of a ``{"data", "request_timestamp"}`` mapping."""
    data = _field(upload, "data")
    return data if isinstance(data, Mapping) else {}


def upload_time(upload: Any) -> datetime.datetime:
    """The host ``request_timestamp`` of an upload (row or mapping)."""
    ts = _field(upload, "request_timestamp")
    if isinstance(ts, str):
        from app.clock import to_study_local

        ts = to_study_local(ts)
    if not isinstance(ts, datetime.datetime):
        raise ValueError("upload has no request_timestamp.")
    return ts


def slot_from_time(ts: datetime.datetime, noon_hour: int = DEFAULT_NOON_HOUR) -> str:
    """``"am"`` before ``noon_hour`` local time, else ``"pm"`` (API-Spec §4.1 ``slot``)."""
    return SLOT_AM if ts.hour < int(noon_hour) else SLOT_PM


def _noon_hour() -> int:
    try:
        from flask import current_app, has_app_context

        if has_app_context():
            return int(current_app.config.get("UPLOAD_SLOT_NOON_HOUR", DEFAULT_NOON_HOUR))
    except Exception:  # noqa: BLE001 - outside an app the default applies
        pass
    return DEFAULT_NOON_HOUR


def action_slot(action: Any) -> str:
    """
    Slot (``"am"``/``"pm"``) of a decision: the ``slot`` of the snapshot it was
    built from (``raw_context["slot"]`` or ``raw_context["design_state"]["slot"]``),
    else derived from its ``request_timestamp`` (no-state fallbacks have no
    snapshot).
    """
    ctx = _field(action, "raw_context") or {}
    if isinstance(ctx, Mapping):
        slot = ctx.get("slot")
        if slot in (SLOT_AM, SLOT_PM):
            return slot
        ds = ctx.get("design_state")
        if isinstance(ds, Mapping) and ds.get("slot") in (SLOT_AM, SLOT_PM):
            return ds["slot"]
    return slot_from_time(upload_time(action), _noon_hour())


def find_outcome_upload(
    decision_type: str,
    action_slot: str | None,
    action_time: datetime.datetime,
    action_week: int | None,
    uploads_after: Sequence[Any],
):
    """
    The upload that closes a decision's outcome (contract D6), or ``None`` if
    it has not arrived.

    ``uploads_after`` are the dyad's uploads ordered by
    ``(request_timestamp, id)``; uploads stamped at or before ``action_time``
    are skipped. Each item is a ``DataUpload`` row or a mapping with ``data``
    and ``request_timestamp``.

    - AYA (``action_slot`` "am"/"pm"; derived from ``action_time`` if not
      given): first upload with the opposite ``slot``.
    - CP: first ``am`` upload on a calendar date after ``action_time``'s date.
    - REL (``action_week`` required): first upload with ``week_in_study`` >
      ``action_week``.
    """
    check_decision_type(decision_type)
    if decision_type == AYA:
        slot = action_slot if action_slot in (SLOT_AM, SLOT_PM) else slot_from_time(action_time, _noon_hour())
        target = SLOT_PM if slot == SLOT_AM else SLOT_AM
    elif decision_type == REL and action_week is None:
        raise ValueError("REL pairing needs the decision's study week (action_week).")
    action_date = action_time.date()
    for up in uploads_after:
        ts = upload_time(up)
        if ts <= action_time:
            continue
        data = upload_data(up)
        if decision_type == AYA:
            if data.get("slot") == target:
                return up
        elif decision_type == CP:
            if data.get("slot") == SLOT_AM and ts.date() > action_date:
                return up
        else:
            week = _number(data.get("week_in_study"))
            if week is not None and week > int(action_week):
                return up
    return None


__all__ = [
    "MISSING_TOKEN",
    "is_missing",
    "aya_level",
    "rel_quality",
    "cp_native_reward",
    "compute_reward",
    "upload_data",
    "upload_time",
    "slot_from_time",
    "action_slot",
    "find_outcome_upload",
]
