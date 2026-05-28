"""
Per-dyad week-1 standardization (main.tex §3, "Variable Standardization").

For each dyad and each agent stream, baselines (mu, sigma) for continuous
variables are estimated once from the dyad's first calendar week alone and
stored in the StandardizationBaseline table. Subsequent action and update
calls look up these baselines and apply them inside the feature builder.

If a dyad has no recorded baselines yet (week 1 in progress, or warmup
override), the feature builder falls through to raw scaled values.
"""

from __future__ import annotations

import datetime
from typing import Iterable

import numpy as np

from app.extensions import db
from app.feature_builder import (
    CONTINUOUS_VARIABLES,
    MIN_BASELINE_SIGMA,
    ProtocolRLFeatureBuilder,
)
from app.models import StandardizationBaseline


def fetch_baselines(group_id: str, decision_type: str) -> dict[str, dict[str, float]]:
    """Return {variable_name: {'mu': μ, 'sigma': σ}} for one (dyad, agent)."""
    rows = StandardizationBaseline.query.filter_by(
        group_id=group_id, decision_type=decision_type
    ).all()
    return {row.variable_name: {"mu": row.mu, "sigma": row.sigma} for row in rows}


def compute_week1_baselines_for_dyad(
    group_id: str,
    decision_type: str,
    week1_records: Iterable[dict],
) -> dict[str, dict[str, float]]:
    """
    Compute and persist (μ, σ) for every continuous variable of one
    (dyad, decision_type) using only the records the caller has already
    filtered to week 1. No-op if baselines already exist for this dyad/agent
    or if the supplied records are empty.

    Each record is a dict carrying at least `raw_context`. The values are
    extracted using the same scaling (the spec.value() callable inside
    `ProtocolRLFeatureBuilder`) that the learner sees.
    """
    existing = StandardizationBaseline.query.filter_by(
        group_id=group_id, decision_type=decision_type
    ).first()
    if existing is not None:
        return fetch_baselines(group_id, decision_type)

    fb = ProtocolRLFeatureBuilder(decision_type)
    continuous_specs = [s for s in fb._specs if s.name in CONTINUOUS_VARIABLES]
    if not continuous_specs:
        return {}

    records = list(week1_records)
    if not records:
        return {}

    values_by_var: dict[str, list[float]] = {s.name: [] for s in continuous_specs}
    for rec in records:
        ctx = rec["raw_context"]
        for spec in continuous_specs:
            if not spec.observed(ctx):
                continue
            values_by_var[spec.name].append(float(spec.value(ctx)))

    baselines: dict[str, dict[str, float]] = {}
    now = datetime.datetime.now()
    for spec in continuous_specs:
        values = values_by_var[spec.name]
        if not values:
            # No observations of this variable in week 1: skip (fall-through to raw).
            continue
        arr = np.asarray(values, dtype=np.float64)
        mu = float(arr.mean())
        sigma = float(arr.std(ddof=0)) if len(arr) > 1 else 0.0
        sigma = max(sigma, MIN_BASELINE_SIGMA)
        baselines[spec.name] = {"mu": mu, "sigma": sigma}
        db.session.add(
            StandardizationBaseline(
                group_id=group_id,
                decision_type=decision_type,
                variable_name=spec.name,
                mu=mu,
                sigma=sigma,
                sample_size=len(values),
                created_at=now,
            )
        )
    db.session.commit()
    return baselines


def filter_week1_records(records: Iterable[dict]) -> list[dict]:
    """Subset of records whose `raw_context["week_in_study"]` is 1."""
    out = []
    for r in records:
        ctx = r.get("raw_context") or {}
        if int(ctx.get("week_in_study", 0)) == 1:
            out.append(r)
    return out
