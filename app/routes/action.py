"""
``POST /api/v1/action`` — mint one randomized decision (API-Spec §2.2;
Monitoring-Spec §6.3, §7 F-A2/F-A3, A13; implementation contract D2, D3, D7,
D11-D13, D21).

Flow for the default learner (``RL_ALGORITHM = "inf_lsvi_eb"``):

1. Validate the request (Monitoring-Spec §6.3 messages) -> ``400``; unknown
   dyad -> ``404``.
2. Under one process-wide lock: the idempotency check (a committed
   ``(group_id, decision_type, decision_idx)`` replays as ``200 Duplicate
   Decision`` with the original outputs and ``rid``), the decision, the single
   uniform draw and the commit (contract D11).
3. Decision: the dyad's latest upload (by ``request_timestamp``, ``id``); none
   -> ``no_state`` fallback. Otherwise the design state
   (``app.learner.build_design_state``), then the scheduled warm-up rule
   (cohort rule D2, then per-dyad rule D3; ``state = null``), else the learned
   path: ``standardized_g`` -> ``learned_probability`` (``NoPolicy`` ->
   ``no_policy``; any other exception -> ``learner_error``) -> A13
   (``invalid_probability``).
4. ``action = int(u < pi)`` with exactly one uniform ``u`` from the
   deterministic buffer for every minted decision.
5. Persist every column (``policy_week``, ``policy_version_id``,
   ``hyperparams_hash``, ``mu``, ``v``, ``upload_id``, ``model_parameters_id``,
   ``raw_context`` = legacy projection + ``design_state``), then
   ``monitoring.after_action_committed``.

Every fallback is an ordinary ``201`` with ``action_prob = 0.5``,
``warmup = true``, ``model_param = null`` (contract D12). Errors: ``503`` when
the decision store cannot be read before minting (``OperationalError``),
``500`` when the decision cannot be persisted.

Legacy learners (contract D1) keep their ``make_state`` / ``get_action`` path
with the same request handling, warm-up rule and fallbacks; ``action_prob`` is
always Pr(A = 1).
"""

from __future__ import annotations

import logging
import math
import random
import threading
from dataclasses import dataclass, field
from typing import Any

from flask import Blueprint, current_app
from sqlalchemy.exc import IntegrityError

from app import learner
from app.algorithms.inf_lsvi_eb import InfLsviEbAlgorithm
from app.extensions import db
from app.models import Action, DataUpload, Group, ModelParameters
from app.protocol import project_snapshot, validate_decision_type
from app.routes.envelope import (
    call_monitoring,
    envelope,
    is_store_error,
    new_rid,
    parse_timestamp,
    read_json_body,
    timestamp_problem,
)
from app.study_calendar import warmup_reason as scheduled_warmup_reason

logger = logging.getLogger(__name__)

action_blueprint = Blueprint("action", __name__)

#: Serializes [idempotency check, decision, uniform draw, commit] across the
#: request threads of this process (contract D11), so the buffer order equals
#: the commit order and a retried triple can never mint twice.
ACTION_LOCK = threading.Lock()

MSG_SUCCESS = "Action requested successfully."
MSG_NO_STATE = "Action requested successfully using the no-state fallback."
MSG_FALLBACK = "Action requested successfully using a fallback."
MSG_PERSIST = "Failed to persist the action."
MSG_STORE = "The decision store is unavailable; use the host fallback."

REASON_NO_STATE = "no_state"
REASON_NO_POLICY = "no_policy"
REASON_LEARNER_ERROR = "learner_error"
REASON_INVALID_PROBABILITY = "invalid_probability"
FALLBACK_REASONS = (
    REASON_NO_STATE,
    REASON_NO_POLICY,
    REASON_LEARNER_ERROR,
    REASON_INVALID_PROBABILITY,
)

MODE_WARMUP = "warmup"
MODE_FALLBACK = "fallback"
MODE_LEARNED = "learned"
MODE_LEGACY = "legacy"

#: A13 tolerance on the allocation bounds (contract D12).
PI_BOUND_TOL = 1e-12

#: Legacy learners whose ``get_action`` already returns Pr(A = 1); every other
#: legacy learner returns the probability of the chosen action.
LEGACY_RETURNS_PR1 = frozenset({"flat_prob"})


class StoreUnavailable(Exception):
    """The decision store could not be read before minting (-> 503 AC-503-STORE)."""


def check_fields(data) -> tuple[bool, str, str | None]:
    """
    Validate the context-free /action request (API-Spec §2.2;
    Monitoring-Spec §6.3).

    Returns ``(ok, message, offending_field)``; the offending field is echoed
    as ``null``. A ``timestamp`` string that is not ISO-8601 is rejected with
    ``timestamp must be an ISO-8601 timestamp; got '<value>'.``.
    """
    if not isinstance(data, dict):
        return False, "Request body is required.", None

    if "group_id" not in data:
        return False, "group_id is required.", "group_id"
    if not isinstance(data["group_id"], str):
        return False, "group_id must be a string.", "group_id"

    if "timestamp" not in data:
        return False, "timestamp is required.", "timestamp"
    problem = timestamp_problem(data["timestamp"])
    if problem is not None:
        return False, problem, "timestamp"

    if "decision_idx" not in data:
        return False, "decision_idx is required.", "decision_idx"
    if not isinstance(data["decision_idx"], int) or isinstance(data["decision_idx"], bool):
        return False, "decision_idx must be an integer.", "decision_idx"

    if "decision_type" not in data:
        return False, "decision_type is required.", "decision_type"
    if not isinstance(data["decision_type"], str):
        return False, "decision_type must be a string.", "decision_type"

    ok, message = validate_decision_type(data["decision_type"])
    if not ok:
        return False, message, "decision_type"
    return True, "", None


def _echo(data, null_field: str | None = None) -> dict:
    """Echo the request identifiers, nulling the failure's cause."""
    data = data if isinstance(data, dict) else {}
    echo = {
        "group_id": data.get("group_id"),
        "decision_type": data.get("decision_type"),
        "decision_idx": data.get("decision_idx"),
    }
    if null_field in echo:
        echo[null_field] = None
    return echo


def _null_outputs() -> dict:
    """The un-producible decision outputs, all null (used on every failure)."""
    return {
        "action": None,
        "action_prob": None,
        "warmup": None,
        "state": None,
        "model_param": None,
    }


def _replay(action_row: Action):
    """
    Idempotent replay of an already-committed decision (API-Spec §2.2;
    Monitoring-Spec §6.3 ``AC-200-REPLAY``): the originally minted outputs and
    the original ``rid``, verbatim.
    """
    return envelope(
        200,
        "Duplicate Decision",
        f"Returning the existing action for ("
        f"{action_row.group_id}, {action_row.decision_type}, "
        f"{action_row.decision_idx}).",
        action_row.rid,
        group_id=action_row.group_id,
        decision_type=action_row.decision_type,
        decision_idx=action_row.decision_idx,
        action=action_row.action,
        action_prob=action_row.action_prob,
        warmup=bool(action_row.is_warmup),
        state=action_row.state,
        model_param=action_row.model_param,
    )


@dataclass
class Decision:
    """Everything the route persists for one decision, before the draw."""

    pi: float
    is_warmup: bool
    reason: str | None
    mode: str
    hyperparams_hash: str | None
    state: list | None = None
    raw_context: dict | None = None
    upload_id: int | None = None
    model_param: list | None = None
    model_parameters_id: int | None = None
    policy_week: Any = None
    policy_version_id: int | None = None
    mu: float | None = None
    v: float | None = None
    # Legacy learned path only: the learner drew the action itself.
    legacy_action: int | None = None
    legacy_random_state: Any = None
    notes: dict = field(default_factory=dict)

    @property
    def message(self) -> str:
        if self.reason == REASON_NO_STATE:
            return MSG_NO_STATE
        if self.reason in FALLBACK_REASONS:
            return MSG_FALLBACK
        return MSG_SUCCESS


# --------------------------------------------------------------- decisions


def _uses_inf_lsvi_eb() -> bool:
    return isinstance(getattr(current_app, "rl_algorithm", None), InfLsviEbAlgorithm)


def _store_guard(exc: BaseException) -> None:
    """Re-raise a database outage as :class:`StoreUnavailable` (D12: 503 before minting)."""
    if is_store_error(exc):
        raise StoreUnavailable(str(exc)) from exc


def _latest_upload(group_id: str) -> DataUpload | None:
    """The dyad's most recent snapshot by (``request_timestamp``, ``id``) (API-Spec §2.2)."""
    return (
        DataUpload.query.filter_by(group_id=group_id)
        .order_by(DataUpload.request_timestamp.desc(), DataUpload.id.desc())
        .first()
    )


def _legacy_context(decision_type: str, upload: DataUpload, decision_idx: int) -> dict:
    """Legacy projection of the snapshot (contract D7: kept in ``raw_context`` for legacy learners)."""
    try:
        return project_snapshot(decision_type, upload.data, decision_idx)
    except Exception as exc:  # noqa: BLE001 - a projection problem must not block the decision
        logger.warning("[Action] Legacy projection failed: %s", type(exc).__name__)
        return {"legacy_projection_error": type(exc).__name__}


def _scheduled_warmup(group: Group, decision_time) -> str | None:
    """``cohort`` (D2, checked first), ``week1`` (D3) or ``None``, with the running warm-up hyperparameters."""
    hp = current_app.hyperparams
    return scheduled_warmup_reason(
        group.group_info["consent_start_date"],
        decision_time.date(),
        n_warm=hp.n_warm,
        t_warm=hp.t_warm,
    )


def _pi_in_bounds(pi: float) -> bool:
    """F-A2 guard: ``pi`` finite and within ``[l_min - 1e-12, l_max + 1e-12]`` (contract D12)."""
    hp = current_app.hyperparams
    return math.isfinite(pi) and hp.l_min - PI_BOUND_TOL <= pi <= hp.l_max + PI_BOUND_TOL


def _json_number(value):
    """A float for JSON columns; non-finite or unreadable values as their repr (Postgres JSON rejects NaN)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return repr(value)
    return number if math.isfinite(number) else repr(number)


def _fault_setting(name: str):
    from app import fault_setting

    return fault_setting(name)


def _fallback(reason: str, base: dict, **extra) -> Decision:
    return Decision(
        pi=0.5,
        is_warmup=True,
        reason=reason,
        mode=MODE_FALLBACK,
        **{**base, **extra},
    )


def _decide_inf_lsvi_eb(
    group: Group,
    decision_type: str,
    decision_idx: int,
    decision_time,
    rid: str,
) -> Decision:
    """The ``inf_lsvi_eb`` decision (contract D7, D12, D13), before the draw."""
    hp = current_app.hyperparams
    group_id = group.group_id

    upload = _latest_upload(group_id)
    if upload is None:
        return _fallback(REASON_NO_STATE, {"hyperparams_hash": hp.sha256})

    raw_context = _legacy_context(decision_type, upload, decision_idx)
    raw_context["upload_id"] = upload.id
    raw_context["upload_rid"] = upload.rid
    base = {"hyperparams_hash": hp.sha256, "raw_context": raw_context, "upload_id": upload.id}

    design_state = None
    try:
        design_state = learner.build_design_state(group_id, decision_type, upload, decision_time)
        raw_context["design_state"] = design_state
    except Exception as exc:  # noqa: BLE001
        _store_guard(exc)
        logger.warning("[Action] build_design_state failed: %s", type(exc).__name__, exc_info=True)
        raw_context["design_state_error"] = f"{type(exc).__name__}: {exc}"

    reason = _scheduled_warmup(group, decision_time)
    if reason is not None:
        # Scheduled warm-up: state = null (API-Spec §2.2); the design state
        # stays in raw_context so the decision enters learning (D8).
        return Decision(pi=0.5, is_warmup=True, reason=reason, mode=MODE_WARMUP, **base)

    if design_state is None:
        return _fallback(REASON_LEARNER_ERROR, base)

    try:
        state = [float(x) for x in learner.standardized_g(decision_type, design_state)]
    except Exception as exc:  # noqa: BLE001
        _store_guard(exc)
        logger.warning("[Action] standardized_g failed: %s", type(exc).__name__, exc_info=True)
        raw_context["learner_error"] = f"{type(exc).__name__}: {exc}"
        return _fallback(REASON_LEARNER_ERROR, base)

    try:
        if decision_type in _fault_setting("FAULT_ACTION_CRASH"):
            raise RuntimeError(f"fault injection: FAULT_ACTION_CRASH for {decision_type}")
        policy = learner.learned_probability(group_id, decision_type, design_state, decision_time)
        pi = float(policy.action_prob)
    except learner.NoPolicy as exc:
        raw_context["no_policy"] = str(exc) or "no in-force policy"
        return _fallback(REASON_NO_POLICY, base, state=state)
    except Exception as exc:  # noqa: BLE001
        _store_guard(exc)
        logger.warning("[Action] learned_probability failed: %s", type(exc).__name__, exc_info=True)
        raw_context["learner_error"] = f"{type(exc).__name__}: {exc}"
        return _fallback(REASON_LEARNER_ERROR, base, state=state)

    forced = _fault_setting("FAULT_FORCE_PI")
    if forced is not None:
        pi = float(forced)

    context = {
        "group_id": group_id,
        "decision_type": decision_type,
        "decision_idx": decision_idx,
        "rid": rid,
        "request_timestamp": decision_time.isoformat(),
        "policy_version_id": policy.policy_version_id,
        "policy_week": policy.policy_week.isoformat() if policy.policy_week is not None else None,
        "model_parameters_id": policy.model_parameters_id,
        "mu": policy.mu,
        "v": policy.v,
    }
    hook_ok = call_monitoring("check_action_probability", pi, learned=True, context=context, default=None)
    if hook_ok is False or not _pi_in_bounds(pi):
        raw_context["rejected_policy"] = {
            "pi": _json_number(pi),
            "mu": _json_number(policy.mu),
            "v": _json_number(policy.v),
            "policy_version_id": policy.policy_version_id,
            "model_parameters_id": policy.model_parameters_id,
        }
        return _fallback(REASON_INVALID_PROBABILITY, base, state=state)

    return Decision(
        pi=pi,
        is_warmup=False,
        reason=None,
        mode=MODE_LEARNED,
        hyperparams_hash=policy.hyperparams_hash,
        state=state,
        raw_context=raw_context,
        upload_id=upload.id,
        model_param=[float(x) for x in policy.model_param],
        model_parameters_id=policy.model_parameters_id,
        policy_week=policy.policy_week,
        policy_version_id=policy.policy_version_id,
        mu=float(policy.mu),
        v=float(policy.v),
    )


def _decide_legacy(
    group: Group,
    decision_type: str,
    decision_idx: int,
    decision_time,
) -> Decision:
    """A legacy learner's decision (contract D1): same gates and fallbacks, the learner's own draw."""
    hp = current_app.hyperparams
    group_id = group.group_id

    upload = _latest_upload(group_id)
    if upload is None:
        return _fallback(REASON_NO_STATE, {"hyperparams_hash": hp.sha256})

    raw_context = _legacy_context(decision_type, upload, decision_idx)
    base = {"hyperparams_hash": hp.sha256, "raw_context": raw_context, "upload_id": upload.id}

    reason = _scheduled_warmup(group, decision_time)
    if reason is not None:
        return Decision(pi=0.5, is_warmup=True, reason=reason, mode=MODE_WARMUP, **base)

    policy_row = (
        ModelParameters.query.filter(ModelParameters.snapshot_type.is_(None))
        .order_by(ModelParameters.timestamp.desc(), ModelParameters.id.desc())
        .first()
    )
    if policy_row is None:
        return _fallback(REASON_NO_POLICY, base)

    algo = current_app.rl_algorithm
    try:
        if decision_type in _fault_setting("FAULT_ACTION_CRASH"):
            raise RuntimeError(f"fault injection: FAULT_ACTION_CRASH for {decision_type}")
        ok, state = algo.make_state({**raw_context, "decision_type": decision_type, "group_id": group_id})
        if not ok:
            raw_context["learner_error"] = f"make_state: {state}"
            return _fallback(REASON_LEARNER_ERROR, base)
        action, prob, random_state = algo.get_action(
            group_id, state, {"probability": policy_row.probability_of_action}, decision_type, decision_idx
        )
        action, prob = int(action), float(prob)
    except Exception as exc:  # noqa: BLE001
        _store_guard(exc)
        logger.warning("[Action] legacy learner failed: %s", type(exc).__name__, exc_info=True)
        raw_context["learner_error"] = f"{type(exc).__name__}: {exc}"
        return _fallback(REASON_LEARNER_ERROR, base)

    algo_name = current_app.config.get("RL_ALGORITHM")
    pr1 = prob if algo_name in LEGACY_RETURNS_PR1 or action == 1 else 1.0 - prob
    model_param = None
    if isinstance(random_state, dict):
        model_param = random_state.pop("model_param", None)
    return Decision(
        pi=pr1,
        is_warmup=False,
        reason=None,
        mode=MODE_LEGACY,
        state=state,
        model_param=model_param,
        model_parameters_id=policy_row.id,
        legacy_action=action,
        legacy_random_state=random_state,
        **base,
    )


# -------------------------------------------------------------------- draw


def _draw_one_uniform(pi: float) -> tuple[int, dict]:
    """
    Exactly one uniform per minted decision, ``action = int(u < pi)``
    (contract D11). ``random_state`` records ``u`` and the buffer cursors.
    Under ``FAULT_SAMPLER_SHIFT = s`` (testing only, D21) ``u`` is read at
    ``cursor + s`` while the unshifted cursors are recorded, so the
    reproducibility check A4 must catch it.
    """
    sampler = getattr(current_app, "sampler", None)
    if sampler is None:
        # Legacy learners without a buffer (flat_prob, thompson_sampling, ...).
        u = random.random()
        return int(u < pi), {"u": u}

    shift = int(_fault_setting("FAULT_SAMPLER_SHIFT") or 0)
    start = sampler.cursor()
    if shift:
        sampler.restore({"normal": start["normal"], "uniform": max(0, start["uniform"] + shift)})
        u = sampler.draw_uniform()
        sampler.restore({"normal": start["normal"], "uniform": start["uniform"] + 1})
    else:
        u = sampler.draw_uniform()
    end = sampler.cursor()
    return int(u < pi), {"u": u, "sampler_cursor_start": start, "sampler_cursor_end": end}


def _mint(decision: Decision) -> tuple[int, dict]:
    """The action and ``actions.random_state`` for ``decision``."""
    if decision.mode == MODE_LEGACY:
        rs = decision.legacy_random_state
        return decision.legacy_action, rs if isinstance(rs, dict) else {"mode": MODE_LEGACY, "learner_state": repr(rs)}
    action, random_state = _draw_one_uniform(decision.pi)
    return action, {"mode": decision.mode, **random_state}


# ------------------------------------------------------------------- route


@action_blueprint.route("/action", methods=["POST"])
def request_action():
    """
    Request an action for a dyad (API-Spec §2.2). Context is pulled from the
    dyad's latest uploaded snapshot, not the request body.
    """
    rid = new_rid()
    data = None
    try:
        data = read_json_body()

        ok, message, field_name = check_fields(data)
        if not ok:
            return envelope(
                400, "Invalid Parameter", message, rid,
                **_echo(data, field_name), **_null_outputs(),
            )

        group_id = data["group_id"]
        decision_idx = data["decision_idx"]
        decision_type = data["decision_type"]
        decision_time = parse_timestamp(data["timestamp"])

        def store_unavailable():
            db.session.rollback()
            return envelope(503, "Service Unavailable", MSG_STORE, rid, **_echo(data), **_null_outputs())

        try:
            group = Group.query.filter_by(group_id=group_id).first()
        except Exception as exc:  # noqa: BLE001
            if is_store_error(exc):
                logger.error("[Action] Decision store unavailable: %s", type(exc).__name__)
                return store_unavailable()
            raise
        if group is None:
            return envelope(
                404, "Unknown Group",
                f"group_id {group_id} is not registered.", rid,
                **_echo(data, "group_id"), **_null_outputs(),
            )

        with ACTION_LOCK:
            # --- before minting: a database outage is a 503 (D12).
            try:
                existing = Action.query.filter_by(
                    group_id=group_id, decision_type=decision_type, decision_idx=decision_idx
                ).first()
                if existing is not None:
                    return _replay(existing)
                if _uses_inf_lsvi_eb():
                    decision = _decide_inf_lsvi_eb(group, decision_type, decision_idx, decision_time, rid)
                else:
                    decision = _decide_legacy(group, decision_type, decision_idx, decision_time)
            except StoreUnavailable:
                logger.error("[Action] Decision store unavailable while deciding.")
                return store_unavailable()
            except Exception as exc:  # noqa: BLE001
                if is_store_error(exc):
                    logger.error("[Action] Decision store unavailable: %s", type(exc).__name__)
                    return store_unavailable()
                raise

            # --- minting: from here a failure is a 500 (AC-500-PERSIST).
            try:
                action, random_state = _mint(decision)
                row = Action(
                    group_id=group_id,
                    action=int(action),
                    rid=rid,
                    state=decision.state,
                    decision_idx=decision_idx,
                    decision_type=decision_type,
                    raw_context=decision.raw_context,
                    action_prob=float(decision.pi),
                    random_state=random_state,
                    model_parameters_id=decision.model_parameters_id,
                    request_timestamp=decision_time,
                    is_warmup=decision.is_warmup,
                    warmup_reason=decision.reason,
                    model_param=decision.model_param,
                    policy_week=decision.policy_week,
                    policy_version_id=decision.policy_version_id,
                    hyperparams_hash=decision.hyperparams_hash,
                    mu=decision.mu,
                    v=decision.v,
                    upload_id=decision.upload_id,
                )
                db.session.add(row)
                db.session.flush()
                action_id = row.id
                db.session.commit()
            except IntegrityError:
                # Another process committed the same triple first: replay it.
                db.session.rollback()
                existing = Action.query.filter_by(
                    group_id=group_id, decision_type=decision_type, decision_idx=decision_idx
                ).first()
                if existing is not None:
                    return _replay(existing)
                logger.error("[Action] Integrity error persisting rid=%s", rid)
                return envelope(500, "Internal Error", MSG_PERSIST, rid, **_echo(data), **_null_outputs())
            except Exception as exc:  # noqa: BLE001
                db.session.rollback()
                logger.error("[Action] Could not persist rid=%s: %s", rid, type(exc).__name__)
                return envelope(500, "Internal Error", MSG_PERSIST, rid, **_echo(data), **_null_outputs())

        call_monitoring("after_action_committed", action_id)

        return envelope(
            201, "Success", decision.message, rid,
            group_id=group_id,
            decision_type=decision_type,
            decision_idx=decision_idx,
            action=int(action),
            action_prob=float(decision.pi),
            warmup=bool(decision.is_warmup),
            state=decision.state,
            model_param=decision.model_param,
        )

    except Exception:  # noqa: BLE001 - last resort: no decision was persisted
        db.session.rollback()
        logger.error("[Action] Unexpected error rid=%s", rid, exc_info=True)
        return envelope(500, "Internal Error", MSG_PERSIST, rid, **_echo(data), **_null_outputs())
