"""
Model diagnostics for the record (Monitoring-Spec §5.L): L1-L3 after each
update from the learner's ``model_parameters`` rows (contract D15: ``hyper``
metadata ``tau2`` / ``floor`` / ``at_floor``; ``posterior`` metadata
``own_data_weight``; ``local_fit`` metadata ``spectral_norm_GXtXplus``,
``theta_change_inf``, ``n_new_rows``), and L4 weekly from the decisions.
"""

from __future__ import annotations

import collections
import datetime
import math

from app.monitoring import catalog
from app.monitoring.checks.common import (
    AGENT_ORDER,
    CheckContext,
    actions_between,
    finished_updates,
    is_learned,
    lookback_start,
    update_event_time,
    update_week,
)
from app.monitoring.findings import Finding, capped, dedup_key, finite_or_none
from app.study_calendar import week_monday


def _rows(rid: str, snapshot_type: str) -> dict[str, list]:
    from app.models import ModelParameters

    out: dict[str, list] = collections.defaultdict(list)
    for row in ModelParameters.query.filter_by(update_rid=rid, snapshot_type=snapshot_type).order_by(ModelParameters.id):
        out[row.decision_type].append(row)
    return out


def l1_for_update(req) -> list[Finding]:
    """L1 (green): a between-dyad variance tau^2_b within 1% of its floor, per agent and block."""
    out = []
    for dt in AGENT_ORDER:
        for row in _rows(req.rid, "hyper").get(dt, []):
            meta = row.metadata_json or {}
            tau, floor = list(meta.get("tau2") or []), list(meta.get("floor") or [])
            if len(tau) != 2 or len(floor) != 2:
                continue
            blocks = [b for b, name in enumerate(("act", "mod"))
                      if finite_or_none(tau[b]) is not None and finite_or_none(floor[b]) is not None
                      and tau[b] <= (1.0 + catalog.L1_FLOOR_REL) * floor[b]]
            if not blocks:
                continue
            names = [("act", "mod")[b] for b in blocks]
            out.append(Finding(
                "L1",
                f"Between-dyad variance at its floor for {dt} ({', '.join(names)}) in the week of "
                f"{update_week(req).isoformat()}; expected while few dyads are enrolled.",
                dedup_key("L1", req.rid, dt),
                at=update_event_time(req),
                detail={"agent": dt, "blocks": names, "tau2": tau, "floor": floor, "N_w": meta.get("N_w"),
                        "update_rid": req.rid, "study_week_monday": update_week(req)},
                decision_type=dt,
                rid=req.rid,
                value=min(tau[b] / floor[b] for b in blocks if floor[b]),
                threshold=1.0 + catalog.L1_FLOOR_REL,
            ))
    return out


def l2_for_update(req) -> list[Finding]:
    """L2 (green, report only): mean weight on each dyad's own data, tr(Sigma_EB (Sigma_hat^aa)^-1) / p_f."""
    out = []
    for dt, rows in _rows(req.rid, "posterior").items():
        weights = {r.group_id: finite_or_none((r.metadata_json or {}).get("own_data_weight")) for r in rows}
        values = [w for w in weights.values() if w is not None]
        if not values:
            continue
        mean = sum(values) / len(values)
        out.append(Finding(
            "L2",
            f"Mean weight on own data for {dt}: {mean:.3f} over {len(values)} dyads (week of "
            f"{update_week(req).isoformat()}).",
            dedup_key("L2", req.rid, dt),
            at=update_event_time(req),
            detail={"agent": dt, "mean_own_data_weight": mean, "n_dyads": len(values), "by_dyad": weights,
                    "update_rid": req.rid, "study_week_monday": update_week(req)},
            decision_type=dt,
            rid=req.rid,
            value=mean,
        ))
    return out


def l3_for_update(req) -> list[Finding]:
    """
    L3 (AYA and CP; REL has gamma = 0): yellow when a dyad's ``G X' X_+`` has
    spectral norm above 1; green record of the week-to-week change of theta for
    dyads whose data no longer change (no new rows).
    """
    out = []
    for dt in ("aya_message", "cp_message"):
        rows = _rows(req.rid, "local_fit").get(dt, [])
        if not rows:
            continue
        above, settled = [], {}
        for row in rows:
            meta = row.metadata_json or {}
            norm = finite_or_none(meta.get("spectral_norm_GXtXplus"))
            if norm is not None and norm > catalog.L3_NORM_THRESHOLD:
                above.append({"group_id": row.group_id, "spectral_norm": norm})
            if meta.get("n_new_rows") == 0 and finite_or_none(meta.get("theta_change_inf")) is not None:
                settled[row.group_id] = meta.get("theta_change_inf")
        at = update_event_time(req)
        if above:
            out.append(Finding(
                "L3",
                f"The weekly learning step for {dt} may not settle for {len(above)} dyads "
                f"(spectral norm of G X'X+ above {catalog.L3_NORM_THRESHOLD:g}).",
                dedup_key("L3", req.rid, dt),
                at=at,
                detail={"agent": dt, "dyads": capped(above, 50), "group_ids": [a["group_id"] for a in above],
                        "update_rid": req.rid, "study_week_monday": update_week(req)},
                decision_type=dt,
                rid=req.rid,
                value=max(a["spectral_norm"] for a in above),
                threshold=catalog.L3_NORM_THRESHOLD,
            ))
        if settled:
            out.append(Finding(
                "L3",
                f"Week-to-week change of theta for {len(settled)} {dt} dyads whose data no longer change: "
                f"max {max(settled.values()):.3g}.",
                dedup_key("L3", req.rid, dt, "theta_change"),
                at=at,
                detail={"agent": dt, "theta_change_inf": settled, "update_rid": req.rid,
                        "study_week_monday": update_week(req)},
                decision_type=dt,
                rid=req.rid,
                value=max(settled.values()),
                severity=catalog.GREEN,
            ))
    return out


def _check_updates(ctx: CheckContext, fn) -> list[Finding]:
    out: list[Finding] = []
    for req in finished_updates(lookback_start(ctx), ctx.window_end):
        out.extend(fn(req))
    return out


def check_l1(ctx: CheckContext) -> list[Finding]:
    return _check_updates(ctx, l1_for_update)


def check_l2(ctx: CheckContext) -> list[Finding]:
    return _check_updates(ctx, l2_for_update)


def check_l3(ctx: CheckContext) -> list[Finding]:
    return _check_updates(ctx, l3_for_update)


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default) of an already sorted list."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def check_l4(ctx: CheckContext) -> list[Finding]:
    """L4 (weekly, green): percentiles of pi over the learned decisions of the week that just ended, per agent."""
    hp = ctx.hp
    monday = week_monday(ctx.now) - datetime.timedelta(days=7)
    start = datetime.datetime.combine(monday, datetime.time.min)
    end = start + datetime.timedelta(days=7)
    by_type: dict[str, list[float]] = collections.defaultdict(list)
    for action in actions_between(start, end, include_start=True):
        if action.request_timestamp < end and is_learned(action):
            by_type[action.decision_type].append(float(action.action_prob))
    out = []
    for dt in AGENT_ORDER:
        values = sorted(by_type.get(dt) or [])
        if not values:
            continue
        pct = {f"p{q}": _percentile(values, q) for q in catalog.L4_PERCENTILES}
        near = sum(1 for v in values if abs(v - hp.l_min) <= catalog.L4_LIMIT_DISTANCE
                   or abs(v - hp.l_max) <= catalog.L4_LIMIT_DISTANCE) / len(values)
        out.append(Finding(
            "L4",
            f"Spread of pi for {dt} in the week of {monday.isoformat()}: median {pct['p50']:.3f}, "
            f"{near:.0%} within {catalog.L4_LIMIT_DISTANCE} of a limit ({len(values)} learned decisions).",
            dedup_key("L4", monday, dt),
            at=ctx.now,
            detail={"agent": dt, "week": monday, "n_learned": len(values), "percentiles": pct,
                    "share_near_limit": near, "l_min": hp.l_min, "l_max": hp.l_max},
            decision_type=dt,
            value=near,
        ))
    return out


__all__ = [
    "l1_for_update",
    "l2_for_update",
    "l3_for_update",
    "check_l1",
    "check_l2",
    "check_l3",
    "check_l4",
]
