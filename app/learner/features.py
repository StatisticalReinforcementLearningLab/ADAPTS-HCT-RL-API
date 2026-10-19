"""
Design-state coordinates, design ranges, standardization and the h/f split of
the ``inf_lsvi_eb`` learner (implementation contract D7; main_new.tex §5.4
Table 3, eq. (range-std), §5.5 Table 4).

Every agent has a raw design vector ``g`` whose first coordinate is the
constant. A coordinate ``x`` with design range ``[l, u]`` enters as

    x~ = (x - (u + l)/2) / ((u - l)/2),

clipped to ``[-1, 1]`` (eq. (range-std)); nothing is estimated from data. The
baseline map ``h`` is ``g`` without the agent's own app burden (AYA, CP), and
``f`` holds the tailoring coordinates (main_new.tex §5.5):

- AYA ``g = [1, eng_aya, burden_aya, game, prior_aya_level, aya_mood, aya_phys, relq_aya]``,
  ``h = g \\ burden_aya`` (p_h = 7), ``f = [1, eng_aya, burden_aya, game]`` (p_f = 4);
- CP  ``g = [1, eng_cp, burden_cp, game, missrate_cp, cp_mood, relq_cp]``,
  ``h = g \\ burden_cp`` (p_h = 6), ``f = [1, eng_cp, burden_cp, game]`` (p_f = 4);
- REL ``g = h = [1, relq_dyad, aya_week_mean_level, eng_avg, burden_avg, prior_game, dsum_avg]``
  (p_h = 7), ``f = [1, relq_dyad]`` (p_f = 2).

The app-burden ranges are the bounds of the host's recursion
``b_{k+1} = gamma b_k + d_k`` with the API-Spec §4.1 discounts (13/14 for AYA,
6/7 for CP): ``[0, 14]``, ``[0, 7]`` and their average ``[0, 10.5]`` for REL.
They describe the host's counter, so they do not follow the learner's
``discount`` setting.

This module is pure (no database access) so it can be used by the fit, the
allocation and offline replays alike.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

import numpy as np

AYA = "aya_message"
CP = "cp_message"
REL = "dyad_game"
DECISION_TYPES = (AYA, CP, REL)

CONST = "const"

# (name, design range [l, u]); the constant has no range. Order is the order
# of g (contract D7).
_COORDINATES: dict[str, tuple[tuple[str, tuple[float, float] | None], ...]] = {
    AYA: (
        (CONST, None),
        ("eng_aya", (1.0, 4.0)),
        ("burden_aya", (0.0, 14.0)),
        ("game", (0.0, 1.0)),
        ("prior_aya_level", (0.0, 2.0)),
        ("aya_mood", (0.0, 5.0)),
        ("aya_phys", (0.0, 5.0)),
        ("relq_aya", (0.0, 5.0)),
    ),
    CP: (
        (CONST, None),
        ("eng_cp", (1.0, 4.0)),
        ("burden_cp", (0.0, 7.0)),
        ("game", (0.0, 1.0)),
        ("missrate_cp", (0.0, 1.0)),
        ("cp_mood", (0.0, 5.0)),
        ("relq_cp", (0.0, 5.0)),
    ),
    REL: (
        (CONST, None),
        ("relq_dyad", (0.0, 5.0)),
        ("aya_week_mean_level", (0.0, 2.0)),
        ("eng_avg", (1.0, 4.0)),
        ("burden_avg", (0.0, 10.5)),
        ("prior_game", (0.0, 1.0)),
        ("dsum_avg", (0.0, 1.0)),
    ),
}

# The agent's own app burden enters f only (main_new.tex §5.4 "App burden").
_H_EXCLUDE = MappingProxyType({AYA: ("burden_aya",), CP: ("burden_cp",), REL: ()})
_F_NAMES = MappingProxyType(
    {
        AYA: (CONST, "eng_aya", "burden_aya", "game"),
        CP: (CONST, "eng_cp", "burden_cp", "game"),
        REL: (CONST, "relq_dyad"),
    }
)

NAMES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {dt: tuple(name for name, _ in coords) for dt, coords in _COORDINATES.items()}
)
RANGES: Mapping[str, tuple[tuple[float, float] | None, ...]] = MappingProxyType(
    {dt: tuple(rng for _, rng in coords) for dt, coords in _COORDINATES.items()}
)
H_INDEX: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {dt: tuple(i for i, n in enumerate(NAMES[dt]) if n not in _H_EXCLUDE[dt]) for dt in DECISION_TYPES}
)
F_INDEX: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {dt: tuple(NAMES[dt].index(n) for n in _F_NAMES[dt]) for dt in DECISION_TYPES}
)
H_NAMES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {dt: tuple(NAMES[dt][i] for i in H_INDEX[dt]) for dt in DECISION_TYPES}
)
F_NAMES: Mapping[str, tuple[str, ...]] = _F_NAMES
# Index of the current-week game coordinate in g (AYA and CP only): the
# coordinate set to a in {0, 1} for the two boundary successors of contract D8.
GAME_INDEX: Mapping[str, int] = MappingProxyType({AYA: NAMES[AYA].index("game"), CP: NAMES[CP].index("game")})


def check_decision_type(decision_type: str) -> str:
    """Return ``decision_type`` if it names one of the three agents, else raise ``ValueError``."""
    if decision_type not in _COORDINATES:
        raise ValueError(f"unknown decision_type {decision_type!r}; expected one of {DECISION_TYPES}.")
    return decision_type


def dims(decision_type: str) -> tuple[int, int]:
    """``(p_h, p_f)`` of the agent (main_new.tex Table 4): AYA (7, 4), CP (6, 4), REL (7, 2)."""
    check_decision_type(decision_type)
    return len(H_INDEX[decision_type]), len(F_INDEX[decision_type])


def standardize(decision_type: str, raw: Sequence[float]) -> tuple[list[float], list[str]]:
    """
    Range-standardize a raw ``g`` (eq. (range-std)) and clip to ``[-1, 1]``.
    Returns ``(g_std, clipped_names)``; the constant stays 1.
    """
    check_decision_type(decision_type)
    names, ranges = NAMES[decision_type], RANGES[decision_type]
    if len(raw) != len(names):
        raise ValueError(f"{decision_type}: raw g has {len(raw)} coordinates, expected {len(names)}.")
    out: list[float] = []
    clipped: list[str] = []
    for name, rng, value in zip(names, ranges, raw):
        x = float(value)
        if not np.isfinite(x):
            raise ValueError(f"{decision_type}: coordinate {name} is not finite ({value!r}).")
        if rng is None:
            out.append(1.0)
            continue
        lo, hi = rng
        z = (x - (hi + lo) / 2.0) / ((hi - lo) / 2.0)
        if z > 1.0 or z < -1.0:
            clipped.append(name)
            z = 1.0 if z > 1.0 else -1.0
        out.append(float(z))
    return out, clipped


def check_design_state(decision_type: str, design_state: Mapping) -> list[float]:
    """Validate a stored design state (contract D7) and return its raw ``g``."""
    check_decision_type(decision_type)
    if not isinstance(design_state, Mapping):
        raise ValueError("design_state must be a mapping.")
    names = list(design_state.get("names") or [])
    if names != list(NAMES[decision_type]):
        raise ValueError(
            f"design_state names {names} do not match the {decision_type} layout {list(NAMES[decision_type])}."
        )
    raw = design_state.get("raw")
    if not isinstance(raw, Sequence) or len(raw) != len(names):
        raise ValueError(f"design_state raw must be a list of {len(names)} numbers.")
    return [float(v) for v in raw]


def standardized_g(decision_type: str, design_state: Mapping) -> list[float]:
    """Standardized full ``g`` of a design state (contract D7)."""
    g, _ = standardize(decision_type, check_design_state(decision_type, design_state))
    return g


def split_hf(decision_type: str, g_std: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """``(h, f)`` from a standardized ``g`` (main_new.tex §5.5)."""
    g = np.asarray(g_std, dtype=float)
    return g[list(H_INDEX[decision_type])], g[list(F_INDEX[decision_type])]


def hf_from_design_state(decision_type: str, design_state: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """``(h, f)`` of a stored design state."""
    return split_hf(decision_type, standardized_g(decision_type, design_state))


def with_game(decision_type: str, design_state: Mapping, game: int) -> dict:
    """
    Copy of an AYA or CP design state with the current-week game coordinate set
    to ``game`` (the boundary successors ``S^{(a)}`` of main_new.tex eq.
    (boundary-game-average); contract D8).
    """
    if decision_type not in GAME_INDEX:
        raise ValueError(f"{decision_type} has no game coordinate.")
    raw = check_design_state(decision_type, design_state)
    raw[GAME_INDEX[decision_type]] = float(game)
    out = dict(design_state)
    out["raw"] = raw
    out["clipped"] = standardize(decision_type, raw)[1]
    return out


__all__ = [
    "AYA",
    "CP",
    "REL",
    "DECISION_TYPES",
    "NAMES",
    "RANGES",
    "H_INDEX",
    "F_INDEX",
    "H_NAMES",
    "F_NAMES",
    "GAME_INDEX",
    "check_decision_type",
    "dims",
    "standardize",
    "check_design_state",
    "standardized_g",
    "split_hf",
    "hf_from_design_state",
    "with_game",
]
