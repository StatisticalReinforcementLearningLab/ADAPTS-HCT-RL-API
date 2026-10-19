"""
Decision checks (Monitoring-Spec §5.A): A13, A4, A10, A6, A7, A16, A17, A2,
A8, A15, A5.

Per-decision checks (A13, A8, A15, A17) have a ``*_for_action`` form used by
the inline hooks and a window form used by the runner; both build the same
findings with the same dedup keys, so a rerun finds exactly what the live
hooks found.
"""

from __future__ import annotations

import collections
import datetime
import math

from app.extensions import db
from app.monitoring import catalog
from app.monitoring.checks.common import (
    AGENT_ORDER,
    OPERATIONAL_FALLBACKS,
    CheckContext,
    ConsentTimeline,
    actions_between,
    actions_on_day,
    consent_of,
    consents_at_logical,
    day_bounds,
    design_state_of,
    expected_calls,
    is_learned,
    number,
    start_dates,
    trial_weeks,
    week_closing_close,
    weekly_opportunities,
)
from app.monitoring.findings import Finding, capped, dedup_key, finite_or_none
from app.study_calendar import study_week, warmup_reason, week_monday

AYA, CP, REL = "aya_message", "cp_message", "dyad_game"


def _who(action) -> str:
    return f"{action.decision_type} decision {action.decision_idx} of {action.group_id}"


# ------------------------------------------------------------------- A13


def pi_valid(pi, *, learned: bool, hp) -> bool:
    """A13 (contract D12): a learned pi in ``[l_min, l_max]`` (1e-12 tolerance) and finite; a warm-up pi of 0.5."""
    try:
        x = float(pi)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(x):
        return False
    if not learned:
        return x == 0.5
    return hp.l_min - catalog.A13_PI_TOL <= x <= hp.l_max + catalog.A13_PI_TOL


def a13_inline_finding(pi, *, learned: bool, context: dict, hp, at=None) -> Finding:
    """The A13 finding raised inside /action when ``pi`` is impossible (the route then uses F-A2)."""
    ctx = context or {}
    rid = ctx.get("rid")
    kind = "learned_pi_out_of_range" if learned else "warmup_pi_not_half"
    return Finding(
        "A13",
        f"Impossible pi {pi!r} for {ctx.get('decision_type')} decision {ctx.get('decision_idx')} of "
        f"{ctx.get('group_id')} ({kind}); the 50/50 fallback is used (F-A2).",
        dedup_key("A13", rid) if rid else dedup_key("A13", "inline", ctx.get("group_id"), ctx.get("decision_type"),
                                                     ctx.get("decision_idx")),
        at=at,
        detail={
            "kind": kind,
            "pi": finite_or_none(pi) if finite_or_none(pi) is not None else repr(pi),
            "l_min": hp.l_min,
            "l_max": hp.l_max,
            "policy_version_id": ctx.get("policy_version_id"),
            "policy_week": ctx.get("policy_week"),
            "model_parameters_id": ctx.get("model_parameters_id"),
            "mu": ctx.get("mu"),
            "v": ctx.get("v"),
        },
        group_id=ctx.get("group_id"),
        decision_type=ctx.get("decision_type"),
        decision_idx=ctx.get("decision_idx"),
        rid=rid,
        value=finite_or_none(pi),
        threshold=hp.l_max if (finite_or_none(pi) or 0.0) > hp.l_max else hp.l_min,
        fallback_used="F-A2",
    )


def a13_for_action(action, hp) -> list[Finding]:
    """
    A13 from a stored decision: an F-A2 ``invalid_probability`` fallback (the
    rejected pi is in ``raw_context["rejected_policy"]``), a warm-up decision
    with pi != 0.5, or a learned decision with pi outside the limits.
    """
    pi = action.action_prob
    kind = None
    value = pi
    fallback = None
    if action.warmup_reason == "invalid_probability":
        rejected = ((action.raw_context or {}).get("rejected_policy") or {}) if isinstance(action.raw_context, dict) else {}
        value = rejected.get("pi")
        kind, fallback = "learned_pi_out_of_range", "F-A2"
    elif action.is_warmup and not pi_valid(pi, learned=False, hp=hp):
        kind = "warmup_pi_not_half"
    elif not action.is_warmup and not pi_valid(pi, learned=True, hp=hp):
        kind = "learned_pi_out_of_range"
    if kind is None:
        return []
    x = finite_or_none(value)
    suffix = "; the 50/50 fallback was used (F-A2)." if fallback else " was recorded."
    return [
        Finding(
            "A13",
            f"Impossible pi {value!r} for {_who(action)} ({kind})" + suffix,
            dedup_key("A13", action.rid),
            at=action.request_timestamp,
            detail={
                "kind": kind,
                "pi": x if x is not None else repr(value),
                "recorded_action_prob": action.action_prob,
                "is_warmup": bool(action.is_warmup),
                "warmup_reason": action.warmup_reason,
                "l_min": hp.l_min,
                "l_max": hp.l_max,
            },
            group_id=action.group_id,
            decision_type=action.decision_type,
            decision_idx=action.decision_idx,
            rid=action.rid,
            value=x,
            threshold=(0.5 if kind == "warmup_pi_not_half" else (hp.l_max if (x or 0.0) > hp.l_max else hp.l_min)),
            fallback_used=fallback,
        )
    ]


def check_a13(ctx: CheckContext) -> list[Finding]:
    hp = ctx.hp
    out: list[Finding] = []
    for action in actions_between(ctx.window_start, ctx.window_end):
        out.extend(a13_for_action(action, hp))
    return out


# -------------------------------------------------------------------- A4


def _uniform_at(sampler, index: int):
    """The buffer's uniform at ``index`` (``None`` when unavailable)."""
    if sampler is None:
        return None
    getter = getattr(sampler, "uniform_at", None)
    try:
        if callable(getter):
            return float(getter(int(index)))
        uniforms = getattr(sampler, "_uniforms", None)
        if uniforms is not None and 0 <= int(index) < len(uniforms):
            return float(uniforms[int(index)])
    except Exception:  # noqa: BLE001 - treated as unavailable
        return None
    return None


class _HpResolver:
    """The hyperparameter set a decision recorded (by fingerprint), else the running one."""

    def __init__(self, running):
        self.running = running
        self.cache = {running.sha256: running}

    def get(self, sha: str | None):
        if sha is None or sha in self.cache:
            return self.cache.get(sha, self.running), sha in self.cache or sha is None
        from app.hyperparams import build_hyperparams, flatten
        from app.models import HyperparameterSet

        row = HyperparameterSet.query.filter_by(sha256=sha).order_by(HyperparameterSet.version.desc()).first()
        hp = None
        if row is not None:
            try:
                hp = build_hyperparams(flatten((row.settings or {}).get("settings") or {}))
            except Exception:  # noqa: BLE001 - fall back to the running set
                hp = None
        self.cache[sha] = hp if hp is not None else self.running
        return self.cache[sha], hp is not None


def recompute_pi(action, hp, bank=None) -> tuple[float, float, float]:
    """
    A4 (1): pi of a learned decision from its saved posterior
    (``actions.model_parameters_id``), its saved design state and the fixed
    Monte Carlo bank (contract D10). Returns ``(pi, mu, v)``; raises
    ``LookupError`` / ``ValueError`` when an input is missing or malformed.
    """
    import numpy as np

    from app.learner import features as F
    from app.learner.allocation import allocation_probability, contrast_moments, mc_bank
    from app.models import ModelParameters

    if action.model_parameters_id is None:
        raise LookupError("no posterior recorded (model_parameters_id is NULL)")
    row = db.session.get(ModelParameters, action.model_parameters_id)
    if row is None:
        raise LookupError(f"posterior row {action.model_parameters_id} is missing")
    ds = design_state_of(action)
    if ds is None:
        raise LookupError("no design_state recorded")
    _, f = F.hf_from_design_state(action.decision_type, ds)
    alpha = np.asarray(row.theta, dtype=float)
    cov = np.asarray(row.covariance, dtype=float)
    if alpha.shape != f.shape or cov.shape != (f.size, f.size):
        raise ValueError(f"posterior shapes {alpha.shape}/{cov.shape} do not match f {f.shape}")
    mu, v = contrast_moments(f, alpha, cov)
    if bank is None:
        bank = mc_bank(hp)
    pi = allocation_probability(
        mu, v, bank=bank, l_min=hp.l_min, l_max=hp.l_max, c=hp.c_rho, b=hp.agent(action.decision_type).b_rho,
        kappa=hp.kappa_rho,
    )
    return pi, mu, v


def _uniform_indices(until: datetime.datetime) -> dict[int, list[str]]:
    """Uniform buffer index -> rids of the decisions (stamped by ``until``) that recorded it."""
    from app.models import Action

    out: dict[int, list[str]] = collections.defaultdict(list)
    rows = Action.query.with_entities(Action.rid, Action.random_state).filter(Action.request_timestamp < until)
    for rid, rs in rows:
        start = (rs or {}).get("sampler_cursor_start") if isinstance(rs, dict) else None
        if isinstance(start, dict) and "uniform" in start:
            out[int(start["uniform"])].append(rid)
    return out


def a4_problems(action, *, hp_resolver: _HpResolver, sampler, banks: dict, indices: dict) -> list[dict]:
    """Every A4 mismatch of one decision (empty when it reproduces)."""
    from app.learner.allocation import mc_bank

    problems: list[dict] = []
    rs = action.random_state if isinstance(action.random_state, dict) else {}
    mode = rs.get("mode")
    base = {"rid": action.rid, "group_id": action.group_id, "decision_idx": action.decision_idx}

    # (1) pi of a learned decision from the saved policy, state and bank.
    if is_learned(action) and mode != "legacy":
        hp, known = hp_resolver.get(action.hyperparams_hash)
        key = (hp.mc_seed, hp.mc_samples)
        if key not in banks:
            banks[key] = mc_bank(hp)
        try:
            pi, mu, v = recompute_pi(action, hp, banks[key])
            if not (math.isfinite(pi) and abs(pi - float(action.action_prob)) <= catalog.A4_PI_TOL):
                problems.append({
                    **base, "kind": "pi_mismatch", "recorded": action.action_prob, "recomputed": pi,
                    "abs_diff": abs(pi - float(action.action_prob)) if math.isfinite(pi) else None,
                    "mu": mu, "v": v, "recorded_mu": action.mu, "recorded_v": action.v,
                    "hyperparams_known": known,
                })
        except (LookupError, ValueError, TypeError) as exc:
            problems.append({**base, "kind": "pi_not_recomputable", "reason": str(exc)[:200]})

    # (2) the action from its saved random number; exactly one uniform per decision.
    if mode == "legacy":
        return problems
    if "u" not in rs:
        problems.append({**base, "kind": "no_random_number"})
        return problems
    start, end = rs.get("sampler_cursor_start"), rs.get("sampler_cursor_end")
    u = finite_or_none(rs.get("u"))
    if isinstance(start, dict) and isinstance(end, dict):
        du = int(end.get("uniform", 0)) - int(start.get("uniform", 0))
        dn = int(end.get("normal", 0)) - int(start.get("normal", 0))
        if du != 1 or dn != 0:
            problems.append({**base, "kind": "uniform_count", "uniforms_used": du, "normals_used": dn})
        index = int(start.get("uniform", 0))
        buffered = _uniform_at(sampler, index)
        if buffered is not None:
            if u is None or buffered != u:
                problems.append({**base, "kind": "uniform_mismatch", "cursor": index, "recorded_u": u,
                                 "buffer_u": buffered})
            u = buffered
        reused = [r for r in indices.get(index, []) if r != action.rid]
        if reused:
            problems.append({**base, "kind": "uniform_reused", "cursor": index, "also_used_by": capped(reused, 10)})
    if u is not None and int(u < float(action.action_prob)) != int(action.action):
        problems.append({**base, "kind": "action_mismatch", "u": u, "pi": action.action_prob,
                         "recorded_action": action.action, "recomputed_action": int(u < float(action.action_prob))})
    return problems


def check_a4(ctx: CheckContext) -> list[Finding]:
    """
    A4 (nightly): every decision of yesterday is redone from saved data —
    (1) pi of each learned decision to within 1e-12, (2) each action from the
    buffer uniform at its saved cursor, with exactly one uniform per decision.
    One red alarm per agent per night lists the mismatches.
    """
    from flask import current_app

    night = (ctx.now - datetime.timedelta(days=1)).date()
    start, end = day_bounds(night)
    resolver = _HpResolver(ctx.hp)
    sampler = getattr(current_app, "sampler", None)
    banks: dict = {}
    indices = _uniform_indices(end)
    by_type: dict[str, list[dict]] = collections.defaultdict(list)
    counts: dict[str, int] = collections.Counter()
    for action in actions_between(start, end, include_start=True):
        if action.request_timestamp >= end:
            continue
        counts[action.decision_type] += 1
        by_type[action.decision_type].extend(
            a4_problems(action, hp_resolver=resolver, sampler=sampler, banks=banks, indices=indices)
        )
    out = []
    for dt in AGENT_ORDER:
        problems = by_type.get(dt) or []
        if not problems:
            continue
        kinds = collections.Counter(p["kind"] for p in problems)
        out.append(
            Finding(
                "A4",
                f"{len({p['rid'] for p in problems})} of {counts[dt]} {dt} decisions on {night.isoformat()} "
                f"could not be reproduced ({', '.join(f'{k}: {n}' for k, n in sorted(kinds.items()))}).",
                dedup_key("A4", night, dt),
                at=ctx.now,
                detail={
                    "night": night,
                    "decision_type": dt,
                    "n_decisions": counts[dt],
                    "kinds": dict(kinds),
                    "tolerance": catalog.A4_PI_TOL,
                    "sampler_available": sampler is not None,
                    "mismatches": capped(problems, 50),
                },
                decision_type=dt,
                value=float(len(problems)),
            )
        )
    return out


# ------------------------------------------------------------------- A10


def check_a10(ctx: CheckContext) -> list[Finding]:
    """
    A10 (nightly): whether each of yesterday's decisions should have been a
    coin flip (cohort rule D2 with consent dates from ``group_history`` as of
    the decision, then the dyad's partial week / week 1, D3) against what was
    recorded. Operational fallbacks are reported by ``AC-201-*``, not here.
    """
    hp = ctx.hp
    night = (ctx.now - datetime.timedelta(days=1)).date()
    start, end = day_bounds(night)
    timeline = ConsentTimeline.load()
    by_type: dict[str, list[dict]] = collections.defaultdict(list)
    for action in actions_between(start, end, include_start=True):
        if action.request_timestamp >= end or action.warmup_reason in OPERATIONAL_FALLBACKS:
            continue
        consents = timeline.as_of(action.timestamp)
        consent = consents.get(action.group_id)
        if consent is None:
            continue
        expected = warmup_reason(
            consent.start,
            action.request_timestamp.date(),
            start_dates(consents),
            n_warm=hp.n_warm,
            t_warm=hp.t_warm,
        )
        recorded = action.warmup_reason if action.is_warmup else None
        kind = None
        if expected is not None and not action.is_warmup:
            kind = "learned_in_warmup"  # the red case of Monitoring-Spec A10
        elif expected is not None and recorded != expected:
            kind = "reason_differs"
        elif expected is None and action.is_warmup:
            kind = "warmup_not_due"
        if kind is not None:
            by_type[action.decision_type].append({
                "rid": action.rid, "group_id": action.group_id, "decision_idx": action.decision_idx,
                "kind": kind, "expected": expected, "recorded_is_warmup": bool(action.is_warmup),
                "recorded_reason": action.warmup_reason, "study_week": study_week(consent.start, action.request_timestamp),
            })
    out = []
    for dt in AGENT_ORDER:
        rows = by_type.get(dt) or []
        if not rows:
            continue
        n_learned = sum(1 for r in rows if r["kind"] == "learned_in_warmup")
        out.append(
            Finding(
                "A10",
                f"{len(rows)} {dt} decisions on {night.isoformat()} did not follow the warm-up rules "
                f"({n_learned} used a learned pi during a coin-flip week).",
                dedup_key("A10", night, dt),
                at=ctx.now,
                detail={"night": night, "decision_type": dt, "n_warm": hp.n_warm, "t_warm": hp.t_warm,
                        "mismatches": capped(rows, 50), "n_mismatches": len(rows)},
                decision_type=dt,
                value=float(len(rows)),
            )
        )
    return out


# ---------------------------------------------------------------- A6, A7


_DELIVERY_FIELDS = (("aya_message", "aya_messages_delivered_week"), ("cp_message", "cp_messages_delivered_week"))


def weekly_delivery(ctx: CheckContext) -> list[dict]:
    """
    Delivered share of scheduled AYA and CP opportunities per dyad in the
    completed full study week before ``ctx.now``'s week (weeks 1..T; the
    partial week excluded), from the week-closing upload's
    ``*_messages_delivered_week`` (API-Spec §4.1).
    """
    from app.models import DataUpload

    W = week_monday(ctx.now)
    prev = W - datetime.timedelta(days=7)
    T = trial_weeks()
    close = week_closing_close(W)
    start = datetime.datetime.combine(W, datetime.time.min)
    rows = []
    for gid, consent in sorted(consents_at_logical(ctx.now).items()):
        if not (1 <= study_week(consent.start, prev) <= T):
            continue
        if consent.start > prev or consent.end < prev + datetime.timedelta(days=6):
            continue
        upload = (
            DataUpload.query.filter(
                DataUpload.group_id == gid,
                DataUpload.request_timestamp >= start,
                DataUpload.request_timestamp <= close,
            )
            .order_by(DataUpload.request_timestamp.asc(), DataUpload.id.asc())
            .first()
        )
        if upload is None:
            continue
        for dt, field in _DELIVERY_FIELDS:
            delivered = number((upload.data or {}).get(field))
            scheduled = weekly_opportunities(dt)
            if delivered is None or scheduled <= 0:
                continue
            rows.append({
                "group_id": gid, "decision_type": dt, "week": prev, "delivered": delivered,
                "scheduled": scheduled, "proportion": delivered / scheduled, "upload_rid": upload.rid,
                "study_week": study_week(consent.start, prev),
            })
    return rows


def _delivery_finding(item: str, row: dict, bound: float, at) -> Finding:
    word = "More than" if item == "A6" else "Fewer than"
    return Finding(
        item,
        f"{word} {bound:.0%} of scheduled {row['decision_type']} messages were delivered to {row['group_id']} "
        f"in the week of {row['week'].isoformat()} ({row['delivered']:g} of {row['scheduled']}).",
        dedup_key(item, row["week"], row["group_id"], row["decision_type"]),
        at=at,
        detail={**row, "bound": bound},
        group_id=row["group_id"],
        decision_type=row["decision_type"],
        value=row["proportion"],
        threshold=bound,
    )


def check_a6(ctx: CheckContext) -> list[Finding]:
    """A6 (weekly, Monday): delivered share strictly above ``MONITOR_MESSAGE_CAP`` (0.8 allowed)."""
    cap = float(ctx.config.get("MONITOR_MESSAGE_CAP", 0.80))
    return [_delivery_finding("A6", r, cap, ctx.now) for r in weekly_delivery(ctx) if r["proportion"] > cap]


def check_a7(ctx: CheckContext) -> list[Finding]:
    """A7 (weekly, Monday): delivered share strictly below ``MONITOR_MESSAGE_FLOOR`` (0.2 allowed)."""
    floor = float(ctx.config.get("MONITOR_MESSAGE_FLOOR", 0.20))
    return [_delivery_finding("A7", r, floor, ctx.now) for r in weekly_delivery(ctx) if r["proportion"] < floor]


# ------------------------------------------------------------------- A16


def _slot_of(action) -> str | None:
    ds = design_state_of(action) or {}
    ctx = action.raw_context if isinstance(action.raw_context, dict) else {}
    slot = ds.get("slot") or ctx.get("slot")
    if slot in ("am", "pm"):
        return slot
    return None


def _first_upload_after(group_id: str, ts, *, slot: str | None = None):
    from app.models import DataUpload

    q = DataUpload.query.filter(DataUpload.group_id == group_id, DataUpload.request_timestamp > ts)
    for upload in q.order_by(DataUpload.request_timestamp.asc(), DataUpload.id.asc()).limit(50):
        if slot is None or (upload.data or {}).get("slot") == slot:
            return upload
    return None


def check_a16(ctx: CheckContext) -> list[Finding]:
    """
    A16 (daily): host deliveries against saved actions (API-Spec §4.1).
    Each AYA and CP decision of two days ago (so that every outcome upload has
    arrived by the 07:00 run) against the delivery flag of the upload that
    reports it (``prompted_by_message`` of the AYA outcome upload;
    ``cp_prompted_by_message`` of the next upload), and the week-closing
    uploads of yesterday's Monday against the week's sends
    (``*_messages_delivered_week``): an action 1 not delivered, an action 0
    followed by a delivery, more deliveries than sends (a delivery without a
    matching action, or duplicates).
    """
    from app.models import Action, DataUpload

    run_day = ctx.now.date()
    decision_day = run_day - datetime.timedelta(days=2)
    rows: list[dict] = []
    for action in actions_on_day(decision_day):
        if action.decision_type == AYA:
            slot = _slot_of(action) or ("am" if action.request_timestamp.hour < 12 else "pm")
            upload = _first_upload_after(action.group_id, action.request_timestamp,
                                         slot="pm" if slot == "am" else "am")
            flag_field = "prompted_by_message"
        elif action.decision_type == CP:
            upload = _first_upload_after(action.group_id, action.request_timestamp)
            flag_field = "cp_prompted_by_message"
        else:
            continue
        if upload is None:
            continue
        flag = (upload.data or {}).get(flag_field)
        if not isinstance(flag, bool):
            continue
        kind = None
        if int(action.action) == 1 and not flag:
            kind = "action_1_not_delivered"
        elif int(action.action) == 0 and flag:
            kind = "action_0_delivered"
        if kind:
            rows.append({"kind": kind, "rid": action.rid, "group_id": action.group_id,
                         "decision_type": action.decision_type, "decision_idx": action.decision_idx,
                         "upload_rid": upload.rid, "field": flag_field})

    closing_day = run_day - datetime.timedelta(days=1)
    if closing_day.weekday() == 0:
        W = closing_day
        prev_start = datetime.datetime.combine(W - datetime.timedelta(days=7), datetime.time.min)
        start = datetime.datetime.combine(W, datetime.time.min)
        close = week_closing_close(W)
        uploads = (
            DataUpload.query.filter(DataUpload.request_timestamp >= start, DataUpload.request_timestamp <= close)
            .order_by(DataUpload.request_timestamp.asc(), DataUpload.id.asc())
            .all()
        )
        seen = set()
        for upload in uploads:
            if upload.group_id in seen:
                continue
            seen.add(upload.group_id)
            for dt, field in _DELIVERY_FIELDS:
                delivered = number((upload.data or {}).get(field))
                if delivered is None:
                    continue
                sends = (
                    Action.query.filter(
                        Action.group_id == upload.group_id,
                        Action.decision_type == dt,
                        Action.action == 1,
                        Action.request_timestamp >= prev_start,
                        Action.request_timestamp < start,
                    ).count()
                )
                if delivered > sends:
                    rows.append({"kind": "deliveries_exceed_sends", "group_id": upload.group_id, "decision_type": dt,
                                 "week": (W - datetime.timedelta(days=7)), "delivered": delivered, "sends": sends,
                                 "upload_rid": upload.rid})
                elif delivered < sends:
                    rows.append({"kind": "sends_not_delivered", "group_id": upload.group_id, "decision_type": dt,
                                 "week": (W - datetime.timedelta(days=7)), "delivered": delivered, "sends": sends,
                                 "upload_rid": upload.rid})
    if not rows:
        return []
    kinds = collections.Counter(r["kind"] for r in rows)
    return [
        Finding(
            "A16",
            f"{len(rows)} delivery mismatches found on {run_day.isoformat()} "
            f"({', '.join(f'{k}: {n}' for k, n in sorted(kinds.items()))}).",
            dedup_key("A16", run_day),
            at=ctx.now,
            detail={"decision_day": decision_day, "closing_day": closing_day, "kinds": dict(kinds),
                    "group_ids": sorted({r["group_id"] for r in rows}), "mismatches": capped(rows, 50)},
            value=float(len(rows)),
        )
    ]


# ------------------------------------------------------------------- A17


def a17_for_action(action, *, streak: int = 3) -> list[Finding]:
    """A17: the ``streak`` most recent weekly ``dyad_game`` actions of the dyad, up to this one, all equal."""
    from app.models import Action

    if action.decision_type != REL:
        return []
    rows = (
        Action.query.filter(
            Action.group_id == action.group_id,
            Action.decision_type == REL,
            Action.request_timestamp <= action.request_timestamp,
        )
        .order_by(Action.request_timestamp.desc(), Action.id.desc())
        .all()
    )
    weekly: list = []
    seen_weeks = set()
    for row in rows:
        if row.request_timestamp == action.request_timestamp and row.id > action.id:
            continue
        week = week_monday(row.request_timestamp)
        if week in seen_weeks:
            continue
        seen_weeks.add(week)
        weekly.append(row)
        if len(weekly) == streak:
            break
    if len(weekly) < streak or len({int(r.action) for r in weekly}) != 1:
        return []
    state = "on" if int(weekly[0].action) == 1 else "off"
    return [
        Finding(
            "A17",
            f"The game of {action.group_id} has been {state} for {streak} weeks in a row.",
            dedup_key("A17", action.rid),
            at=action.request_timestamp,
            detail={"streak_weeks": streak, "game": int(weekly[0].action),
                    "weeks": [week_monday(r.request_timestamp) for r in reversed(weekly)],
                    "rids": [r.rid for r in reversed(weekly)],
                    "warmup": [bool(r.is_warmup) for r in reversed(weekly)]},
            group_id=action.group_id,
            decision_type=REL,
            decision_idx=action.decision_idx,
            rid=action.rid,
            value=float(streak),
            threshold=float(streak),
        )
    ]


def check_a17(ctx: CheckContext) -> list[Finding]:
    streak = int(ctx.config.get("MONITOR_GAME_STREAK_WEEKS", 3))
    out: list[Finding] = []
    for action in actions_between(ctx.window_start, ctx.window_end, decision_type=REL):
        out.extend(a17_for_action(action, streak=streak))
    return out


# -------------------------------------------------------------------- A2


def a2_rows(day: datetime.date) -> dict[str, list[dict]]:
    """Per agent: dyads whose ``/action`` count on ``day`` differs from the decision calendar."""
    end = day_bounds(day)[1]
    consents = consents_at_logical(end)
    actual = collections.Counter((a.group_id, a.decision_type) for a in actions_on_day(day))
    out: dict[str, list[dict]] = {}
    for dt in AGENT_ORDER:
        rows = []
        gids = set(consents) | {g for (g, t) in actual if t == dt}
        for gid in sorted(gids):
            consent = consents.get(gid)
            expected = expected_calls(dt, consent, day) if consent is not None else 0
            got = actual.get((gid, dt), 0)
            if expected != got:
                rows.append({"group_id": gid, "expected": expected, "actual": got,
                             "study_week": study_week(consent.start, day) if consent else None})
        out[dt] = rows
    return out


def check_a2(ctx: CheckContext) -> list[Finding]:
    """
    A2 (daily 07:00, previous day): per active dyad, agent and day, the
    ``/action`` calls against ``MONITOR_DECISION_CALENDAR`` (AYA 2, CP 1 a day;
    REL 1 on Mondays from week 1), and calls on days without decisions. A
    missing call may be a host coin flip (F-A1).
    """
    day = (ctx.now - datetime.timedelta(days=1)).date()
    out = []
    for dt, rows in a2_rows(day).items():
        if not rows:
            continue
        missing = sum(max(r["expected"] - r["actual"], 0) for r in rows)
        extra = sum(max(r["actual"] - r["expected"], 0) for r in rows)
        out.append(
            Finding(
                "A2",
                f"{dt} calls on {day.isoformat()} differ from the decision calendar for {len(rows)} dyads "
                f"({missing} missing, {extra} extra).",
                dedup_key("A2", day, dt),
                at=ctx.now,
                detail={"day": day, "decision_type": dt, "missing_calls": missing, "extra_calls": extra,
                        "group_ids": [r["group_id"] for r in rows], "dyads": capped(rows, 50)},
                decision_type=dt,
                value=float(missing + extra),
                fallback_used="F-A1" if missing else None,
            )
        )
    return out


# -------------------------------------------------------------------- A8


def a8_for_action(action, consent=None) -> list[Finding]:
    """A8: a decision dated before the dyad's ``consent_start_date`` or after its ``consent_end_date`` (as of the call)."""
    if consent is None:
        consent = consent_of(action.group_id, action.timestamp)
    if consent is None:
        return []
    day = action.request_timestamp.date()
    if consent.start <= day <= consent.end:
        return []
    side = "before consent_start_date" if day < consent.start else "after consent_end_date"
    return [
        Finding(
            "A8",
            f"/action for {_who(action)} on {day.isoformat()} is {side} "
            f"({consent.start.isoformat()} to {consent.end.isoformat()}).",
            dedup_key("A8", action.rid),
            at=action.request_timestamp,
            detail={"decision_date": day, "consent_start_date": consent.start, "consent_end_date": consent.end,
                    "side": side},
            group_id=action.group_id,
            decision_type=action.decision_type,
            decision_idx=action.decision_idx,
            rid=action.rid,
        )
    ]


def check_a8(ctx: CheckContext) -> list[Finding]:
    timeline = ConsentTimeline.load()
    out: list[Finding] = []
    for action in actions_between(ctx.window_start, ctx.window_end):
        out.extend(a8_for_action(action, timeline.as_of(action.timestamp).get(action.group_id)))
    return out


# ------------------------------------------------------------------- A15


def a15_for_action(action) -> list[Finding]:
    """
    A15: a learned AYA or CP decision of week w whose ``current_game_on``
    (the ``game`` design coordinate) differs from the REL action drawn for
    that week before it. Imputed values (``"miss"``) are not compared.
    """
    from app.models import Action

    if action.decision_type not in (AYA, CP) or not is_learned(action):
        return []
    ds = design_state_of(action)
    if ds is None:
        return []
    names, raw = list(ds.get("names") or []), list(ds.get("raw") or [])
    if "game" not in names or len(raw) != len(names) or "game" in (ds.get("imputed") or {}):
        return []
    game = finite_or_none(raw[names.index("game")])
    if game is None:
        return []
    week_start = datetime.datetime.combine(week_monday(action.request_timestamp), datetime.time.min)
    rel = (
        Action.query.filter(
            Action.group_id == action.group_id,
            Action.decision_type == REL,
            Action.request_timestamp >= week_start,
            Action.request_timestamp <= action.request_timestamp,
        )
        .order_by(Action.request_timestamp.desc(), Action.id.desc())
        .first()
    )
    if rel is None or int(round(game)) == int(rel.action):
        return []
    return [
        Finding(
            "A15",
            f"{_who(action)} used current_game_on = {int(round(game))} but the week's dyad_game action was "
            f"{int(rel.action)}.",
            dedup_key("A15", action.rid),
            at=action.request_timestamp,
            detail={"game_used": int(round(game)), "rel_action": int(rel.action), "rel_rid": rel.rid,
                    "week": week_start.date(), "upload_rid": ds.get("upload_rid")},
            group_id=action.group_id,
            decision_type=action.decision_type,
            decision_idx=action.decision_idx,
            rid=action.rid,
        )
    ]


def check_a15(ctx: CheckContext) -> list[Finding]:
    out: list[Finding] = []
    for action in actions_between(ctx.window_start, ctx.window_end):
        out.extend(a15_for_action(action))
    return out


# -------------------------------------------------------------------- A5


def _a5_day(day: datetime.date, decision_type: str, hp) -> dict:
    latest: dict[str, float] = {}
    for action in actions_on_day(day, decision_type):
        if is_learned(action):
            latest[action.group_id] = float(action.action_prob)
    at_limit = [g for g, pi in latest.items()
                if abs(pi - hp.l_min) <= catalog.A5_LIMIT_DISTANCE or abs(pi - hp.l_max) <= catalog.A5_LIMIT_DISTANCE]
    n = len(latest)
    share = len(at_limit) / n if n else 0.0
    return {"day": day, "n_dyads": n, "n_at_limit": len(at_limit), "share": share,
            "qualifies": n >= catalog.A5_MIN_DYADS and share >= catalog.A5_SHARE}


def check_a5(ctx: CheckContext) -> list[Finding]:
    """
    A5 (daily): on each of the last 7 days, at least 80% of the agent's dyads
    with learned decisions (and at least 5 such dyads) had their latest pi of
    the day within 0.01 of L_min or L_max.
    """
    hp = ctx.hp
    last = (ctx.now - datetime.timedelta(days=1)).date()
    out = []
    for dt in AGENT_ORDER:
        days = [_a5_day(last - datetime.timedelta(days=k), dt, hp) for k in range(catalog.A5_DAYS)]
        if not all(d["qualifies"] for d in days):
            continue
        out.append(
            Finding(
                "A5",
                f"For {catalog.A5_DAYS} days in a row at least {catalog.A5_SHARE:.0%} of {dt} dyads with learned "
                f"decisions had pi at a limit.",
                dedup_key("A5", dt, last),
                at=ctx.now,
                detail={"decision_type": dt, "days": list(reversed(days)), "l_min": hp.l_min, "l_max": hp.l_max},
                decision_type=dt,
                value=min(d["share"] for d in days),
                threshold=catalog.A5_SHARE,
            )
        )
    return out


__all__ = [
    "pi_valid",
    "a13_inline_finding",
    "a13_for_action",
    "check_a13",
    "recompute_pi",
    "a4_problems",
    "check_a4",
    "check_a10",
    "weekly_delivery",
    "check_a6",
    "check_a7",
    "check_a16",
    "a17_for_action",
    "check_a17",
    "a2_rows",
    "check_a2",
    "a8_for_action",
    "check_a8",
    "a15_for_action",
    "check_a15",
    "check_a5",
]
