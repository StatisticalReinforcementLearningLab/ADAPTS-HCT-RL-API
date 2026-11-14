"""
Rewards of the ``inf_lsvi_eb`` learner and the pairing of a decision with the
upload that closes its outcome (implementation contract D6; main_new.tex §5.3,
eqs. (reward-aya), (reward-cp), (reward-rel-quality), (reward-rel-adh),
(reward-rel); API-Spec §2.3, §4.1).

Every reward is bounded with non-response at its worst level, and the learner
uses it divided by the agent's native range maximum ``r_max`` (AYA 2, CP 5,
REL ``5 + 2 w_adh``) when ``rewards.normalize`` is on (the default), so every
reward enters the working model on ``[0, 1]``.

Pairing (contract D6) uses the host-sent ``slot``, ``day_in_study`` and
``week_in_study`` values of the decision's snapshot and later snapshots:

- AYA AM: the first ``pm`` upload on the same or a later study day.
- AYA PM: the first ``am`` upload on a later study day.
- CP: the first ``am`` upload on a later study day.
- REL: the first upload from a later study week.

No study index is derived from a timestamp. Thus a report filed after noon or
a previous-day PM decision sent shortly after midnight keeps its host-defined
slot, day and week.

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


def _snapshot_values(action: Any, name: str) -> list:
    """Values of ``name`` carried in a decision's stored snapshot context."""
    ctx = _field(action, "raw_context") or {}
    out = []
    if isinstance(ctx, Mapping):
        if name in ctx:
            out.append(ctx[name])
        design_state = ctx.get("design_state")
        if isinstance(design_state, Mapping) and name in design_state:
            out.append(design_state[name])
    return out


def _snapshot_int(action: Any, name: str) -> int | None:
    for value in _snapshot_values(action, name):
        number = _number(value)
        if number is not None and number == int(number):
            return int(number)
    return None


def action_slot(action: Any) -> str | None:
    """
    Slot (``"am"``/``"pm"``) of a decision: the ``slot`` of the snapshot it was
    built from (``raw_context["slot"]`` or ``raw_context["design_state"]["slot"]``),
    else ``None`` (no-state fallbacks have no snapshot).
    """
    for slot in _snapshot_values(action, "slot"):
        if slot in (SLOT_AM, SLOT_PM):
            return slot
    return None


def action_day(action: Any) -> int | None:
    """Host-sent ``day_in_study`` of a decision's snapshot, else ``None``."""
    return _snapshot_int(action, "day_in_study")


def action_week(action: Any) -> int | None:
    """Host-sent ``week_in_study`` of a decision's snapshot, else ``None``."""
    return _snapshot_int(action, "week_in_study")


def find_outcome_upload(
    decision_type: str,
    action_slot: str | None,
    action_day: int | None,
    action_week: int | None,
    uploads: Sequence[Any],
):
    """
    The upload that closes a decision's outcome (contract D6), or ``None`` if
    it has not arrived.

    ``action_slot`` / ``action_day`` / ``action_week`` come from the snapshot
    used for the decision. ``uploads`` are ordered by ``(request_timestamp,
    id)``; the first matching host-sent bookkeeping values wins.

    - AYA: opposite slot on the applicable host study day.
    - CP: first ``am`` upload on a later host study day.
    - REL (``action_week`` required): first upload with ``week_in_study`` >
      ``action_week``.
    """
    check_decision_type(decision_type)
    if decision_type == AYA:
        if action_slot not in (SLOT_AM, SLOT_PM) or action_day is None:
            raise ValueError("AYA pairing needs the decision's slot and study day (action_slot, action_day).")
    elif decision_type == CP and action_day is None:
        raise ValueError("CP pairing needs the decision's study day (action_day).")
    elif decision_type == REL and action_week is None:
        raise ValueError("REL pairing needs the decision's study week (action_week).")
    for up in uploads:
        data = upload_data(up)
        if decision_type == REL:
            week = _number(data.get("week_in_study"))
            if week is not None and week > int(action_week):
                return up
            continue
        day = _number(data.get("day_in_study"))
        if day is None:
            continue
        if decision_type == AYA and action_slot == SLOT_AM:
            if data.get("slot") == SLOT_PM and day >= int(action_day):
                return up
        elif data.get("slot") == SLOT_AM and day > int(action_day):
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
    "action_slot",
    "action_day",
    "action_week",
    "find_outcome_upload",
]
