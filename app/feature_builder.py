"""
Feature construction for RL.

Single shared missing indicator I across ALL variables: I = 1 iff every
raw variable is observed for this context, else I = 0. When I = 0 the
values v_j are forced to 0 (no partial-information features).

The action-value features φ(s,a) are:
  [1, a, I, v_1*I, ..., v_J*I, a*I, a*v_1*I, ..., a*v_J*I]

The pre-action base u(s) stored on API rows:
  [1, I, v_1*I, ..., v_J*I]

so dim(u) = 2 + J and dim(φ) = 4 + 2*J.

Per-dyad week-1 standardization (main.tex §3): continuous variables are
centered/scaled by per-dyad baselines (μ, σ) from week 1 when available.
Ordinal and binary variables are left untouched.

Callers pass a `baselines` dict at feature-build time:

    {variable_name: {"mu": float, "sigma": float}}

Only variable names listed in `CONTINUOUS_VARIABLES` below are candidates for
standardization; all others fall through unchanged. Missing baselines fall back
to the raw (already-scaled) value — this is the steady-state behavior during
week 1 and the warmup cohort.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from app.protocol import DEFAULT_DIARY_ITEMS, is_missing


# Variables that are standardized per-dyad (continuous scale on the raw
# protocol units). Diary/relationship-quality items are nominally ordinal
# 1–5 and left unstandardized per main.tex §3. Burden / missing-rate /
# diary-summary are continuous summary statistics and are standardized.
CONTINUOUS_VARIABLES = frozenset(
    {
        "aya_app_burden",
        "cp_app_burden",
        "aya_missing_rate_7d",
        "cp_missing_rate_7d",
        "aya_diary_summary",
        "cp_diary_summary",
    }
)

# Minimum per-dyad standard deviation to avoid division blow-up.
MIN_BASELINE_SIGMA = 1e-3


def _standardize(
    variable_name: str,
    scaled_value: float,
    baselines: dict[str, dict[str, float]] | None,
) -> float:
    """Apply (v - μ) / max(σ, ε) when a baseline for `variable_name` exists."""
    if baselines is None:
        return scaled_value
    base = baselines.get(variable_name)
    if not base:
        return scaled_value
    mu = float(base.get("mu", 0.0))
    sigma = float(base.get("sigma", 1.0))
    return (scaled_value - mu) / max(sigma, MIN_BASELINE_SIGMA)


@dataclass(frozen=True)
class RawVariableSpec:
    """One logical input variable after flattening nested blocks."""

    name: str
    observed: Callable[[dict], bool]
    value: Callable[[dict], float]


class ProtocolRLFeatureBuilder:
    """
    Builds u(s) and φ(s,a) from the API's raw context dict. Callers may pass
    `baselines` to activate per-dyad standardization of continuous variables
    (main.tex §3); warmup / week-1 states leave it None.
    """

    def __init__(self, decision_type: str):
        self.decision_type = decision_type
        self._specs = self._specs_for(decision_type)
        self.n_vars = len(self._specs)

    @property
    def base_dim(self) -> int:
        # [1, I, v_1*I, ..., v_J*I] — single shared missing indicator.
        return 2 + self.n_vars

    @property
    def phi_dim(self) -> int:
        # [1, a, I, v_1*I, ..., v_J*I, a*I, a*v_1*I, ..., a*v_J*I]
        return 4 + 2 * self.n_vars

    @property
    def variable_names(self) -> list[str]:
        return [spec.name for spec in self._specs]

    def base_vector(
        self,
        context: dict[str, Any],
        baselines: dict[str, dict[str, float]] | None = None,
    ) -> np.ndarray:
        """u(s): [1, I, v_1*I, ..., v_J*I] with shared I = AND_j observed."""
        all_observed = all(spec.observed(context) for spec in self._specs)
        I = 1.0 if all_observed else 0.0
        values: list[float] = []
        for spec in self._specs:
            if I == 0.0:
                values.append(0.0)
                continue
            raw_val = float(spec.value(context))
            if spec.name in CONTINUOUS_VARIABLES:
                raw_val = _standardize(spec.name, raw_val, baselines)
            values.append(raw_val)
        return np.asarray([1.0, I, *values], dtype=np.float64)

    def phi(
        self,
        context: dict[str, Any],
        action: int,
        baselines: dict[str, dict[str, float]] | None = None,
    ) -> np.ndarray:
        """φ(s,a) for linear Q / value regression."""
        u = self.base_vector(context, baselines=baselines)
        return self.expand_base_to_phi(u, action)

    def expand_base_to_phi(self, base: np.ndarray, action: int) -> np.ndarray:
        """Map stored u(s) = [1, I, v_1*I, ..., v_J*I] to φ(s,a)."""
        a = float(action)
        if base.shape[0] != self.base_dim:
            raise ValueError(
                f"base length {base.shape[0]} != expected {self.base_dim} for {self.decision_type}"
            )
        I = float(base[1])
        vIs = base[2 : 2 + self.n_vars]
        main = np.concatenate(([1.0, a, I], vIs))
        inter = np.concatenate(([a * I], a * vIs))
        return np.concatenate([main, inter])

    @staticmethod
    def for_decision_type(decision_type: str) -> ProtocolRLFeatureBuilder:
        return ProtocolRLFeatureBuilder(decision_type)

    def _specs_for(self, dt: str) -> list[RawVariableSpec]:
        if dt == "aya_message":
            return [
                RawVariableSpec(
                    "slot_pm",
                    lambda c: True,
                    lambda c: 1.0 if c.get("slot") == "pm" else 0.0,
                ),
                RawVariableSpec(
                    "day_in_study",
                    lambda c: True,
                    lambda c: float(c["day_in_study"]) / 100.0,
                ),
                RawVariableSpec(
                    "week_in_study",
                    lambda c: True,
                    lambda c: float(c["week_in_study"]) / 14.0,
                ),
                RawVariableSpec(
                    "prior_med_adherence",
                    lambda c: not is_missing(c.get("prior_med_adherence")),
                    lambda c: float(c["prior_med_adherence"])
                    if not is_missing(c.get("prior_med_adherence"))
                    else 0.0,
                ),
                *[
                    RawVariableSpec(
                        f"aya_diary_{item}",
                        lambda c, it=item: not is_missing(c.get("aya_diary", {}).get(it)),
                        lambda c, it=item: float(c["aya_diary"][it]) / 5.0
                        if not is_missing(c.get("aya_diary", {}).get(it))
                        else 0.0,
                    )
                    for item in DEFAULT_DIARY_ITEMS
                ],
                RawVariableSpec(
                    "relationship_quality_cp",
                    lambda c: not is_missing(c.get("relationship_quality_cp")),
                    lambda c: float(c["relationship_quality_cp"]) / 5.0
                    if not is_missing(c.get("relationship_quality_cp"))
                    else 0.0,
                ),
                RawVariableSpec(
                    "relationship_quality_aya",
                    lambda c: not is_missing(c.get("relationship_quality_aya")),
                    lambda c: float(c["relationship_quality_aya"]) / 5.0
                    if not is_missing(c.get("relationship_quality_aya"))
                    else 0.0,
                ),
                RawVariableSpec(
                    "aya_app_engagement",
                    lambda c: True,
                    lambda c: float(c["aya_app_engagement"]),
                ),
                RawVariableSpec(
                    "aya_app_burden",
                    lambda c: True,
                    lambda c: float(c["aya_app_burden"]) / 10.0,
                ),
                RawVariableSpec(
                    "aya_missing_rate_7d",
                    lambda c: True,
                    lambda c: float(c["aya_missing_rate_7d"]),
                ),
                RawVariableSpec(
                    "current_game_on",
                    lambda c: True,
                    lambda c: float(c["current_game_on"]),
                ),
            ]
        if dt == "cp_message":
            return [
                RawVariableSpec(
                    "day_in_study",
                    lambda c: True,
                    lambda c: float(c["day_in_study"]) / 100.0,
                ),
                RawVariableSpec(
                    "week_in_study",
                    lambda c: True,
                    lambda c: float(c["week_in_study"]) / 14.0,
                ),
                *[
                    RawVariableSpec(
                        f"cp_diary_{item}",
                        lambda c, it=item: not is_missing(c.get("cp_diary", {}).get(it)),
                        lambda c, it=item: float(c["cp_diary"][it]) / 5.0
                        if not is_missing(c.get("cp_diary", {}).get(it))
                        else 0.0,
                    )
                    for item in DEFAULT_DIARY_ITEMS
                ],
                RawVariableSpec(
                    "cp_app_engagement",
                    lambda c: True,
                    lambda c: float(c["cp_app_engagement"]),
                ),
                RawVariableSpec(
                    "cp_app_burden",
                    lambda c: True,
                    lambda c: float(c["cp_app_burden"]) / 10.0,
                ),
                RawVariableSpec(
                    "cp_missing_rate_7d",
                    lambda c: True,
                    lambda c: float(c["cp_missing_rate_7d"]),
                ),
                RawVariableSpec(
                    "relationship_quality_cp",
                    lambda c: not is_missing(c.get("relationship_quality_cp")),
                    lambda c: float(c["relationship_quality_cp"]) / 5.0
                    if not is_missing(c.get("relationship_quality_cp"))
                    else 0.0,
                ),
                RawVariableSpec(
                    "relationship_quality_aya",
                    lambda c: not is_missing(c.get("relationship_quality_aya")),
                    lambda c: float(c["relationship_quality_aya"]) / 5.0
                    if not is_missing(c.get("relationship_quality_aya"))
                    else 0.0,
                ),
                RawVariableSpec(
                    "current_game_on",
                    lambda c: not is_missing(c.get("current_game_on")),
                    lambda c: float(c["current_game_on"])
                    if not is_missing(c.get("current_game_on"))
                    else 0.0,
                ),
            ]
        if dt == "dyad_game":
            # Pruned to the 5 tailoring vars (engagement/burden/prior action).
            # Dropped week_in_study (γ=0 → no time-trend needed), the two
            # relationship_quality_* fields (they are the outcome, not state),
            # and the two diary_summary fields (weak/noisy). J=5 → D=2+4J=22.
            return [
                RawVariableSpec(
                    "aya_app_engagement",
                    lambda c: True,
                    lambda c: float(c["aya_app_engagement"]),
                ),
                RawVariableSpec(
                    "cp_app_engagement",
                    lambda c: True,
                    lambda c: float(c["cp_app_engagement"]),
                ),
                RawVariableSpec(
                    "aya_app_burden",
                    lambda c: True,
                    lambda c: float(c["aya_app_burden"]) / 10.0,
                ),
                RawVariableSpec(
                    "cp_app_burden",
                    lambda c: True,
                    lambda c: float(c["cp_app_burden"]) / 10.0,
                ),
                RawVariableSpec(
                    "prior_game_action",
                    lambda c: not is_missing(c.get("prior_game_action")),
                    lambda c: float(c["prior_game_action"])
                    if not is_missing(c.get("prior_game_action"))
                    else 0.0,
                ),
            ]
        raise ValueError(f"unknown decision_type: {dt}")


def phi_dims_by_decision_type() -> dict[str, int]:
    return {dt: ProtocolRLFeatureBuilder(dt).phi_dim for dt in ("aya_message", "cp_message", "dyad_game")}


# Tailoring vs prognostic classification per agent (main.tex Table 2 and §3.1).
# Tailoring features are those that plausibly modify the incremental effect of
# the agent's action. Prognostic features index dyad health/context with weaker
# expected treatment interaction; the prior down-weights their action-interaction
# coefficients by τ_x² / 2 (main.tex Appendix B, two-block prior).
TAILORING_VARS_BY_AGENT: dict[str, frozenset[str]] = {
    "aya_message": frozenset({
        "prior_med_adherence",
        "aya_app_engagement",
        "aya_app_burden",
        "current_game_on",
    }),
    "cp_message": frozenset({
        "cp_app_engagement",
        "cp_app_burden",
        "current_game_on",
    }),
    "dyad_game": frozenset({
        "aya_app_engagement",
        "cp_app_engagement",
        "aya_app_burden",
        "cp_app_burden",
        "prior_game_action",
    }),
}


def tailoring_mask(decision_type: str) -> np.ndarray:
    """
    Return a length-n_vars boolean array: True for tailoring features, False for
    prognostic. Indexed in the same order as ProtocolRLFeatureBuilder(decision_type)._specs.
    """
    fb = ProtocolRLFeatureBuilder(decision_type)
    tailoring = TAILORING_VARS_BY_AGENT.get(decision_type, frozenset())
    return np.asarray([name in tailoring for name in fb.variable_names], dtype=bool)
