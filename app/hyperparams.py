"""
Hyperparameters of the ``inf_lsvi_eb`` learner (Hyperparameters.md §2-6;
implementation contract D16; main_new.tex Table 1, §5.6-§5.9).

The resolved set is built from three layers, later layers winning:

1. the design defaults of main_new.tex Table 1 (``DEFAULTS`` below);
2. the YAML file (``ADAPTS_HYPERPARAMS`` env var, else
   ``ADAPTS-HCT-RL-API/hyperparameters.yaml``; a missing default file means
   "no overrides");
3. ``app.config["HYPERPARAMS_OVERRIDES"]`` (tests and simulations only).

Unknown keys and out-of-range values stop the app with a message naming the
key (Hyperparameters.md §2.1). Fractions may be written as strings such as
``"13/14"``; numeric strings such as ``"1e-4"`` (which YAML 1.1 reads as a
string) are accepted too.

Derived values (Hyperparameters.md §3.2, §5; contract D16) are computed from
the resolved settings and are part of the fingerprint:
``p_h, p_f, p_main = p_h + 1, p_mod = p_f - 1, lambda_b = m p_b``
(eq. (runin-choice)), ``tau0_b = sigma^2 / lambda_b`` (eq. (hyperconstants)),
``r_max`` (native reward range: AYA 2, CP 5, REL 5 + 2 w_adh; contract D6),
the per-agent allocation slope ``b_rho * r_max`` (eq. (rho-params)), and the
noise bound ``sigma_bar^2 = range^2 / (4 (1 - gamma^2))`` (eq. (sigma-plugin)).

The fingerprint is ``sha256`` of the canonical JSON of
``{"settings": ..., "derived": ...}``. The set is frozen into
``hyperparameter_sets`` at the first learning update; after freezing the app
refuses to start with a different resolved set (Hyperparameters.md §2.3;
Monitoring-Spec C9).
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import logging
import math
import os
import sys
from collections.abc import Mapping
from fractions import Fraction
from types import MappingProxyType
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_HYPERPARAMS_FILE = os.path.join(REPO_ROOT, "hyperparameters.yaml")
ENV_VAR = "ADAPTS_HYPERPARAMS"

SOURCE_DEFAULT = "default"
SOURCE_FILE = "file"
SOURCE_OVERRIDE = "override"
SOURCE_DERIVED = "derived"

AGENTS = ("aya", "cp", "rel")
DECISION_TYPE_TO_AGENT = {"aya_message": "aya", "cp_message": "cp", "dyad_game": "rel"}
AGENT_TO_DECISION_TYPE = {agent: dt for dt, agent in DECISION_TYPE_TO_AGENT.items()}

# Structural, not configurable (Hyperparameters.md §3.1; main_new.tex Table 3/4;
# contract D7): dimensions of the baseline map h and the advantage map f.
P_H = MappingProxyType({"aya": 7, "cp": 6, "rel": 7})
P_F = MappingProxyType({"aya": 4, "cp": 4, "rel": 2})
# Upper bound on the information per decision per coordinate (c_0, Table 1),
# used only to report run-ins.
RUNIN_C0 = MappingProxyType({"main": 1.0, "mod": 0.25})
# Native reward range upper end (lower end 0; contract D6): AYA level/2 has
# native range [0, 2], CP mood [0, 5], REL [0, 5 + 2 w_adh].
R_MAX_BASE = MappingProxyType({"aya": 2.0, "cp": 5.0, "rel": 5.0})
REL_R_MAX_PER_W_ADH = 2.0

# Flat key -> (kind, default). Kinds: "frac" (finite float; fraction strings
# allowed), "int", "bool", "pair" (two fracs). Defaults are main_new.tex Table 1.
_LEAVES: dict[str, tuple[str, Any]] = {
    "discount.aya": ("frac", Fraction(13, 14)),
    "discount.cp": ("frac", Fraction(6, 7)),
    "discount.rel": ("frac", 0),
    "trial_weeks": ("int", 14),
    "rel_reward.respondent_weights": ("pair", (Fraction(1, 3), Fraction(2, 3))),
    "rel_reward.w_adh": ("frac", 0),
    "noise.sigma_sq.aya": ("frac", 0.158),
    "noise.sigma_sq.cp": ("frac", 0.122),
    "noise.sigma_sq.rel": ("frac", 0.078),
    "rewards.normalize": ("bool", True),
    "penalty.m": ("frac", 2),
    "eb.nu0": ("frac", 4),
    "eb.lambda0": ("frac", 0.1),
    "allocation.l_min": ("frac", 0.2),
    "allocation.l_max": ("frac", 0.8),
    "allocation.c_rho": ("frac", 3.5),
    "allocation.b_rho": ("frac", 20),
    "allocation.kappa_rho": ("frac", 1),
    "allocation.mc_samples": ("int", 500),
    "allocation.mc_seed": ("int", 12345),
    "warmup.n_warm": ("int", 5),
    "warmup.t_warm": ("int", 1),
    "numerics.j_backups": ("int", 1),
    "numerics.k_max": ("int", 200),
    "numerics.eps_tol": ("frac", 1e-4),
    "numerics.eps_ridge": ("frac", 1e-6),
}
LEAF_KEYS = tuple(_LEAVES)
_BRANCHES = frozenset(
    ".".join(key.split(".")[:i]) for key in _LEAVES for i in range(1, key.count(".") + 1)
)


class HyperparameterError(ValueError):
    """A hyperparameter file or value is invalid; ``key`` names the offending key."""

    def __init__(self, key: str, message: str):
        self.key = key
        super().__init__(f"hyperparameter {key!r}: {message}")


# ------------------------------------------------------------------ parsing


def _parse_frac(key: str, value: Any) -> float:
    if isinstance(value, bool) or value is None:
        raise HyperparameterError(key, f"must be a number, got {value!r}.")
    if isinstance(value, (int, float)):
        out = float(value)
    elif isinstance(value, Fraction):
        out = float(value)
    elif isinstance(value, str):
        try:
            out = float(Fraction(value.strip()))
        except (ValueError, ZeroDivisionError) as exc:
            raise HyperparameterError(
                key, f"must be a number or a fraction string such as '13/14', got {value!r}."
            ) from exc
    else:
        raise HyperparameterError(key, f"must be a number, got {type(value).__name__}.")
    if not math.isfinite(out):
        raise HyperparameterError(key, f"must be finite, got {value!r}.")
    return out


def _parse_int(key: str, value: Any) -> int:
    if isinstance(value, bool) or value is None:
        raise HyperparameterError(key, f"must be an integer, got {value!r}.")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        raise HyperparameterError(key, f"must be an integer, got {value!r}.")
    if isinstance(value, str):
        try:
            frac = Fraction(value.strip())
        except (ValueError, ZeroDivisionError) as exc:
            raise HyperparameterError(key, f"must be an integer, got {value!r}.") from exc
        if frac.denominator == 1:
            return int(frac)
    raise HyperparameterError(key, f"must be an integer, got {value!r}.")


def _parse_bool(key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raise HyperparameterError(key, f"must be true or false, got {value!r}.")


def _parse_pair(key: str, value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise HyperparameterError(key, f"must be a list of two numbers, got {value!r}.")
    return (_parse_frac(f"{key}[0]", value[0]), _parse_frac(f"{key}[1]", value[1]))


_PARSERS = {"frac": _parse_frac, "int": _parse_int, "bool": _parse_bool, "pair": _parse_pair}


def _parse_leaf(key: str, value: Any) -> Any:
    kind, _ = _LEAVES[key]
    return _PARSERS[kind](key, value)


def _default_values() -> dict[str, Any]:
    return {key: _parse_leaf(key, default) for key, (_, default) in _LEAVES.items()}


def parse_settings(data: Any, origin: str = "hyperparameters") -> dict[str, Any]:
    """
    Parse a (possibly partial) settings tree into ``{flat_key: value}``.

    ``data`` is the YAML mapping (nested, as in Hyperparameters.md §6) or a
    mapping of flat dotted keys (``{"warmup.n_warm": 2}``); both may be mixed.
    ``None`` means no settings. Unknown keys raise :class:`HyperparameterError`.
    """
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise HyperparameterError(origin, "the top level must be a mapping of keys to values.")
    flat: dict[str, Any] = {}

    def walk(node: Mapping, prefix: str) -> None:
        for raw_key, value in node.items():
            if not isinstance(raw_key, str):
                raise HyperparameterError(f"{prefix}.{raw_key}" if prefix else str(raw_key), "keys must be strings.")
            key = f"{prefix}.{raw_key}" if prefix else raw_key
            if key in _LEAVES:
                flat[key] = _parse_leaf(key, value)
            elif key in _BRANCHES:
                if value is None:
                    continue
                if not isinstance(value, Mapping):
                    children = sorted(k[len(key) + 1:] for k in _LEAVES if k.startswith(key + "."))
                    raise HyperparameterError(key, f"must be a mapping with keys {children}, got {value!r}.")
                walk(value, key)
            else:
                raise HyperparameterError(
                    key, "unknown key (allowed keys are listed in Hyperparameters.md §3-§4)."
                )

    walk(data, "")
    return flat


def _sigma_bar_sq(gamma: float, value_range: float) -> float:
    """eq. (sigma-plugin): range^2 / (4 (1 - gamma^2))."""
    return value_range**2 / (4.0 * (1.0 - gamma**2))


def _r_max(agent: str, w_adh: float) -> float:
    if agent == "rel":
        return R_MAX_BASE["rel"] + REL_R_MAX_PER_W_ADH * w_adh
    return R_MAX_BASE[agent]


def validate_values(values: Mapping[str, Any]) -> None:
    """Range and cross-field validation of a complete flat value set (Hyperparameters.md §3-§4)."""

    def need(cond: bool, key: str, message: str) -> None:
        if not cond:
            raise HyperparameterError(key, f"{message}; got {values[key]!r}.")

    for agent in AGENTS:
        key = f"discount.{agent}"
        need(0.0 <= values[key] < 1.0, key, "must be in [0, 1)")
    need(values["trial_weeks"] >= 1, "trial_weeks", "must be an integer >= 1")
    weights = values["rel_reward.respondent_weights"]
    need(weights[0] > 0 and weights[1] > 0, "rel_reward.respondent_weights", "both weights must be positive")
    need(abs(weights[0] + weights[1] - 1.0) <= 1e-9, "rel_reward.respondent_weights", "weights must sum to 1")
    need(values["rel_reward.w_adh"] >= 0.0, "rel_reward.w_adh", "must be >= 0")
    need(values["penalty.m"] > 0.0, "penalty.m", "must be > 0")
    need(values["eb.nu0"] > 0.0, "eb.nu0", "must be > 0")
    need(values["eb.lambda0"] > 0.0, "eb.lambda0", "must be > 0")
    need(0.0 < values["allocation.l_min"] < 1.0, "allocation.l_min", "must be in (0, 1)")
    need(0.0 < values["allocation.l_max"] < 1.0, "allocation.l_max", "must be in (0, 1)")
    need(
        values["allocation.l_min"] < values["allocation.l_max"],
        "allocation.l_min",
        f"must be below allocation.l_max (= {values['allocation.l_max']!r})",
    )
    for key in ("allocation.c_rho", "allocation.b_rho", "allocation.kappa_rho"):
        need(values[key] > 0.0, key, "must be > 0")
    need(values["allocation.mc_samples"] >= 100, "allocation.mc_samples", "must be an integer >= 100")
    need(0 <= values["allocation.mc_seed"] < 2**63, "allocation.mc_seed", "must be an integer in [0, 2^63)")
    need(values["warmup.n_warm"] >= 2, "warmup.n_warm", "must be an integer >= 2")
    need(values["warmup.t_warm"] >= 1, "warmup.t_warm", "must be an integer >= 1")
    need(values["numerics.j_backups"] >= 1, "numerics.j_backups", "must be an integer >= 1")
    need(values["numerics.k_max"] >= 1, "numerics.k_max", "must be an integer >= 1")
    need(values["numerics.eps_tol"] > 0.0, "numerics.eps_tol", "must be > 0")
    need(values["numerics.eps_ridge"] > 0.0, "numerics.eps_ridge", "must be > 0")
    normalize = values["rewards.normalize"]
    for agent in AGENTS:
        key = f"noise.sigma_sq.{agent}"
        gamma = values[f"discount.{agent}"]
        value_range = 1.0 if normalize else _r_max(agent, values["rel_reward.w_adh"])
        bound = _sigma_bar_sq(gamma, value_range)
        need(values[key] > 0.0, key, "must be > 0")
        need(
            values[key] <= bound,
            key,
            f"must not exceed the Bellman-target variance bound sigma_bar^2 = {bound!r} "
            "(main_new.tex eq. (sigma-plugin))",
        )


# ------------------------------------------------------------- resolved set


@dataclasses.dataclass(frozen=True)
class AgentConstants:
    """Per-agent constants, configured and derived (Hyperparameters.md §3.2, §5; contract D16)."""

    agent: str
    decision_type: str
    gamma: float
    sigma_sq: float
    sigma_bar_sq: float
    p_h: int
    p_f: int
    p_main: int
    p_mod: int
    lambda_main: float
    lambda_mod: float
    tau0_act: float
    tau0_mod: float
    r_max: float
    reward_divisor: float
    b_rho: float
    run_in_main: float
    run_in_mod: float

    @property
    def dim(self) -> int:
        """Length of theta = (beta, alpha): p_h + p_f."""
        return self.p_h + self.p_f

    def local_penalty_diag(self) -> list[float]:
        """Diagonal of Lambda^loc = diag(lambda_main I_{p_h}, lambda_main, lambda_mod I_{p_f - 1}) (contract D8)."""
        return [self.lambda_main] * self.p_h + [self.lambda_main] + [self.lambda_mod] * (self.p_f - 1)

    def tau0_diag(self) -> list[float]:
        """Diagonal of the hyper-prior scale: (tau0_act, tau0_mod I_{p_f - 1}) (contract D9)."""
        return [self.tau0_act] + [self.tau0_mod] * (self.p_f - 1)


@dataclasses.dataclass(frozen=True, eq=False)
class Hyperparams:
    """
    The resolved hyperparameter set (Hyperparameters.md §3-§5). Attribute
    names follow main_new.tex; per-agent values are keyed ``"aya"``, ``"cp"``,
    ``"rel"`` and :meth:`agent` also accepts a decision type.
    """

    discount: Mapping[str, float]
    trial_weeks: int
    respondent_weights: tuple[float, float]
    w_adh: float
    sigma_sq: Mapping[str, float]
    normalize_rewards: bool
    m: float
    nu0: float
    lambda0: float
    l_min: float
    l_max: float
    c_rho: float
    b_rho: float
    kappa_rho: float
    mc_samples: int
    mc_seed: int
    n_warm: int
    t_warm: int
    j_backups: int
    k_max: int
    eps_tol: float
    eps_ridge: float
    agents: Mapping[str, AgentConstants]
    sources: Mapping[str, str]
    source_path: str | None = None

    # -- access

    def agent(self, agent_or_decision_type: str) -> AgentConstants:
        """Per-agent constants for ``"aya"|"cp"|"rel"`` or a decision type."""
        return self.agents[agent_key(agent_or_decision_type)]

    @property
    def rho_at_zero(self) -> float:
        """rho(0) = L_min + (L_max - L_min) (1 + c_rho)^(-kappa_rho) (eq. (rho))."""
        return self.l_min + (self.l_max - self.l_min) * (1.0 + self.c_rho) ** (-self.kappa_rho)

    def eb_floor(self, agent_or_decision_type: str, n_w: int) -> tuple[float, float]:
        """Hyper-update floors nu0 tau0_b / (N_w + nu0 + 2) for b = act, mod (eq. (floor); contract D9)."""
        a = self.agent(agent_or_decision_type)
        denom = float(n_w) + self.nu0 + 2.0
        return (self.nu0 * a.tau0_act / denom, self.nu0 * a.tau0_mod / denom)

    # -- serialization

    def flat_settings(self) -> dict[str, Any]:
        """``{flat_key: value}`` of the configured (non-derived) settings."""
        out: dict[str, Any] = {}
        for agent in AGENTS:
            out[f"discount.{agent}"] = self.discount[agent]
            out[f"noise.sigma_sq.{agent}"] = self.sigma_sq[agent]
        out.update(
            {
                "trial_weeks": self.trial_weeks,
                "rel_reward.respondent_weights": list(self.respondent_weights),
                "rel_reward.w_adh": self.w_adh,
                "rewards.normalize": self.normalize_rewards,
                "penalty.m": self.m,
                "eb.nu0": self.nu0,
                "eb.lambda0": self.lambda0,
                "allocation.l_min": self.l_min,
                "allocation.l_max": self.l_max,
                "allocation.c_rho": self.c_rho,
                "allocation.b_rho": self.b_rho,
                "allocation.kappa_rho": self.kappa_rho,
                "allocation.mc_samples": self.mc_samples,
                "allocation.mc_seed": self.mc_seed,
                "warmup.n_warm": self.n_warm,
                "warmup.t_warm": self.t_warm,
                "numerics.j_backups": self.j_backups,
                "numerics.k_max": self.k_max,
                "numerics.eps_tol": self.eps_tol,
                "numerics.eps_ridge": self.eps_ridge,
            }
        )
        return {key: out[key] for key in LEAF_KEYS}

    def settings_dict(self) -> dict[str, Any]:
        """The configured settings as a nested mapping in the YAML shape of Hyperparameters.md §6."""
        return unflatten(self.flat_settings())

    def derived_dict(self) -> dict[str, Any]:
        return {
            "agents": {name: dataclasses.asdict(self.agents[name]) for name in AGENTS},
            "rho_at_zero": self.rho_at_zero,
        }

    def to_dict(self) -> dict[str, Any]:
        """``{"settings": ..., "derived": ...}``: what is hashed and frozen."""
        return {"settings": self.settings_dict(), "derived": self.derived_dict()}

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def sha256(self) -> str:
        """SHA-256 of the canonical JSON of :meth:`to_dict` (contract D16)."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def rows(self) -> list[tuple[str, Any, str]]:
        """``(key, value, source)`` rows for ``flask hyperparams show``."""
        rows = [(key, value, self.sources.get(key, SOURCE_DEFAULT)) for key, value in self.flat_settings().items()]
        rows += [(key, value, SOURCE_DERIVED) for key, value in flatten({"derived": self.derived_dict()}).items()]
        return rows


def agent_key(agent_or_decision_type: str) -> str:
    """Map a decision type (``"aya_message"``) or agent key (``"aya"``) to the agent key."""
    if agent_or_decision_type in AGENTS:
        return agent_or_decision_type
    try:
        return DECISION_TYPE_TO_AGENT[agent_or_decision_type]
    except KeyError:
        raise KeyError(f"unknown agent or decision type {agent_or_decision_type!r}") from None


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def flatten(tree: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings to dotted keys (lists stay leaves)."""
    out: dict[str, Any] = {}
    for key, value in tree.items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten(value, full))
        else:
            out[full] = value
    return out


def unflatten(flat: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in flat.items():
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def build_hyperparams(
    values: Mapping[str, Any],
    sources: Mapping[str, str] | None = None,
    source_path: str | None = None,
) -> Hyperparams:
    """Validate a complete flat value set and compute the derived constants."""
    validate_values(values)
    m = values["penalty.m"]
    w_adh = values["rel_reward.w_adh"]
    normalize = values["rewards.normalize"]
    b_rho_native = values["allocation.b_rho"]
    agents: dict[str, AgentConstants] = {}
    for agent in AGENTS:
        gamma = values[f"discount.{agent}"]
        sigma_sq = values[f"noise.sigma_sq.{agent}"]
        p_h, p_f = P_H[agent], P_F[agent]
        p_main, p_mod = p_h + 1, p_f - 1
        lambda_main, lambda_mod = m * p_main, m * p_mod
        r_max = _r_max(agent, w_adh)
        divisor = r_max if normalize else 1.0
        agents[agent] = AgentConstants(
            agent=agent,
            decision_type=AGENT_TO_DECISION_TYPE[agent],
            gamma=gamma,
            sigma_sq=sigma_sq,
            sigma_bar_sq=_sigma_bar_sq(gamma, r_max / divisor),
            p_h=p_h,
            p_f=p_f,
            p_main=p_main,
            p_mod=p_mod,
            lambda_main=lambda_main,
            lambda_mod=lambda_mod,
            tau0_act=sigma_sq / lambda_main,
            tau0_mod=sigma_sq / lambda_mod,
            r_max=r_max,
            reward_divisor=divisor,
            b_rho=b_rho_native * divisor,
            run_in_main=lambda_main / RUNIN_C0["main"],
            run_in_mod=lambda_mod / RUNIN_C0["mod"],
        )
    src = {key: SOURCE_DEFAULT for key in LEAF_KEYS}
    src.update(sources or {})
    return Hyperparams(
        discount=MappingProxyType({a: values[f"discount.{a}"] for a in AGENTS}),
        trial_weeks=values["trial_weeks"],
        respondent_weights=tuple(values["rel_reward.respondent_weights"]),
        w_adh=w_adh,
        sigma_sq=MappingProxyType({a: values[f"noise.sigma_sq.{a}"] for a in AGENTS}),
        normalize_rewards=normalize,
        m=m,
        nu0=values["eb.nu0"],
        lambda0=values["eb.lambda0"],
        l_min=values["allocation.l_min"],
        l_max=values["allocation.l_max"],
        c_rho=values["allocation.c_rho"],
        b_rho=b_rho_native,
        kappa_rho=values["allocation.kappa_rho"],
        mc_samples=values["allocation.mc_samples"],
        mc_seed=values["allocation.mc_seed"],
        n_warm=values["warmup.n_warm"],
        t_warm=values["warmup.t_warm"],
        j_backups=values["numerics.j_backups"],
        k_max=values["numerics.k_max"],
        eps_tol=values["numerics.eps_tol"],
        eps_ridge=values["numerics.eps_ridge"],
        agents=MappingProxyType(agents),
        sources=MappingProxyType(src),
        source_path=source_path,
    )


def resolve_path(path: str | None = None) -> tuple[str, bool]:
    """``(path, explicit)``: the argument, else ``$ADAPTS_HYPERPARAMS``, else the repo default file."""
    if path:
        return str(path), True
    env = os.environ.get(ENV_VAR)
    if env:
        return env, True
    return DEFAULT_HYPERPARAMS_FILE, False


def read_settings_file(path: str) -> dict[str, Any]:
    """Parse one YAML file into flat values (comment-only or empty files give ``{}``)."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise HyperparameterError(path, "hyperparameter file not found.") from exc
    except yaml.YAMLError as exc:
        raise HyperparameterError(path, f"not valid YAML: {exc}") from exc
    return parse_settings(data, origin=path)


def load_hyperparams(
    path: str | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> Hyperparams:
    """
    Resolve the hyperparameter set: design defaults, then the YAML file
    (``path``, else ``$ADAPTS_HYPERPARAMS``, else the repo's
    ``hyperparameters.yaml`` if present), then ``overrides`` (nested or
    dotted keys). Raises :class:`HyperparameterError` naming the key on any
    unknown key or out-of-range value.
    """
    file_path, explicit = resolve_path(path)
    values = _default_values()
    sources: dict[str, str] = {key: SOURCE_DEFAULT for key in LEAF_KEYS}
    used_path: str | None = None
    if explicit or os.path.exists(file_path):
        for key, value in read_settings_file(file_path).items():
            values[key] = value
            sources[key] = SOURCE_FILE
        used_path = os.path.abspath(file_path)
    if overrides:
        for key, value in parse_settings(overrides, origin="HYPERPARAMS_OVERRIDES").items():
            values[key] = value
            sources[key] = SOURCE_OVERRIDE
    hp = build_hyperparams(values, sources, used_path)
    if hp.w_adh != 0.0 and sources["noise.sigma_sq.rel"] == SOURCE_DEFAULT:
        logger.warning(
            "rel_reward.w_adh = %s but noise.sigma_sq.rel keeps its w_adh = 0 design value %s "
            "(main_new.tex eq. (sigma-design)); set it explicitly.",
            hp.w_adh,
            hp.sigma_sq["rel"],
        )
    return hp


def default_hyperparams() -> Hyperparams:
    """The design set of main_new.tex Table 1 with no file and no overrides."""
    return build_hyperparams(_default_values())


def get_hyperparams() -> Hyperparams:
    """The running app's resolved set (``current_app.hyperparams``); outside an app, a fresh load."""
    from flask import current_app, has_app_context

    if has_app_context():
        hp = getattr(current_app, "hyperparams", None)
        if hp is not None:
            return hp
    return load_hyperparams()


def diff_resolved(frozen: Mapping[str, Any], running: Mapping[str, Any]) -> list[dict[str, Any]]:
    """
    Key-by-key differences between two ``to_dict()`` payloads (settings and
    derived values). Returns ``[{"key", "frozen", "running"}, ...]``, the
    ``settings.*`` keys first, each group sorted by key.
    """
    a, b = flatten(dict(frozen)), flatten(dict(running))
    out = []
    # Configured settings first (the cause), then the derived values.
    for key in sorted(set(a) | set(b), key=lambda k: (not k.startswith("settings."), k)):
        va, vb = a.get(key, "<absent>"), b.get(key, "<absent>")
        if isinstance(va, tuple):
            va = list(va)
        if isinstance(vb, tuple):
            vb = list(vb)
        if va != vb:
            out.append({"key": key, "frozen": va, "running": vb})
    return out


# ----------------------------------------------------------------- freezing


def latest_frozen():
    """The newest ``hyperparameter_sets`` row, or ``None`` before freezing. Requires an app context."""
    from app.models import HyperparameterSet

    return HyperparameterSet.query.order_by(HyperparameterSet.version.desc()).first()


def in_force_version(at: datetime.datetime | None = None):
    """
    The approved ``hyperparameter_sets`` row in force at study time ``at``
    (latest ``in_force_from <= at``); with ``at=None`` the newest version.
    ``None`` when nothing was frozen by then (Monitoring-Spec C9).
    """
    from app.models import HyperparameterSet

    query = HyperparameterSet.query
    if at is not None:
        query = query.filter(HyperparameterSet.in_force_from <= at)
    return query.order_by(HyperparameterSet.in_force_from.desc(), HyperparameterSet.version.desc()).first()


def in_force_hash(at: datetime.datetime | None = None) -> str | None:
    """SHA-256 of the approved version in force at ``at`` (see :func:`in_force_version`), else ``None``."""
    row = in_force_version(at)
    return row.sha256 if row is not None else None


def freeze_if_needed(
    reason: str = "first learning update",
    *,
    hp: Hyperparams | None = None,
    in_force_from: datetime.datetime | None = None,
    commit: bool = True,
):
    """
    Freeze the resolved set as version 1 unless a version already exists
    (Hyperparameters.md §2.3; contract D14/D16). Returns the version row in
    force (the existing newest one, or the new version 1).

    ``in_force_from`` is the study-logical time the set governs from
    (default ``clock.study_now()``; the learner passes the first learning
    update's ``request_timestamp``). With ``commit=False`` the row is only
    flushed into the caller's transaction.
    """
    from sqlalchemy.exc import IntegrityError

    from app import clock
    from app.extensions import db
    from app.models import HyperparameterSet

    existing = latest_frozen()
    if existing is not None:
        return existing
    hp = hp or get_hyperparams()
    row = HyperparameterSet(
        version=1,
        sha256=hp.sha256,
        settings=hp.to_dict(),
        in_force_from=in_force_from if in_force_from is not None else clock.study_now(),
        reason=reason,
    )
    db.session.add(row)
    try:
        if commit:
            db.session.commit()
        else:
            db.session.flush()
    except IntegrityError:
        # A concurrent run froze first; use its row.
        db.session.rollback()
        existing = latest_frozen()
        if existing is None:
            raise
        return existing
    logger.info("Froze hyperparameter set version 1 sha256=%s (%s)", hp.sha256, reason)
    return row


def amend(
    reason: str,
    *,
    hp: Hyperparams,
    in_force_from: datetime.datetime | None = None,
):
    """
    Record a protocol amendment (Hyperparameters.md §2.3): store ``hp`` as the
    next version. Raises ``RuntimeError`` when nothing is frozen yet or when
    ``hp`` equals the newest frozen version.
    """
    from app import clock
    from app.extensions import db
    from app.models import HyperparameterSet

    reason = (reason or "").strip()
    if not reason:
        raise ValueError("an amendment needs a non-empty --reason.")
    latest = latest_frozen()
    if latest is None:
        raise RuntimeError(
            "No hyperparameter set is frozen yet; edit the file freely and restart "
            "(Hyperparameters.md §2.3). Nothing amended."
        )
    if latest.sha256 == hp.sha256:
        raise RuntimeError(f"The resolved set equals frozen version {latest.version}; nothing to amend.")
    row = HyperparameterSet(
        version=latest.version + 1,
        sha256=hp.sha256,
        settings=hp.to_dict(),
        in_force_from=in_force_from if in_force_from is not None else clock.study_now(),
        reason=reason,
    )
    db.session.add(row)
    db.session.commit()
    logger.info("Recorded hyperparameter amendment version %d sha256=%s: %s", row.version, row.sha256, reason)
    return row


def check_startup_against_frozen(hp: Hyperparams | None = None):
    """
    Start-up check (Hyperparameters.md §2.3; Monitoring-Spec C9): if a set is
    frozen and the running resolved set differs from the newest version,
    report it to monitoring (C9 red alarm) and raise ``RuntimeError`` so the
    app refuses to start. Returns the matching frozen row (or ``None`` before
    freezing). Requires an app context.
    """
    hp = hp or get_hyperparams()
    frozen = latest_frozen()
    if frozen is None or frozen.sha256 == hp.sha256:
        return frozen
    differences = diff_resolved(frozen.settings or {}, hp.to_dict())
    detail = {
        "frozen_version": frozen.version,
        "frozen_sha256": frozen.sha256,
        "running_sha256": hp.sha256,
        "source_path": hp.source_path,
        "n_differences": len(differences),
        "differences": differences[:50],
    }
    try:
        from app import monitoring

        monitoring.on_startup_settings_mismatch(detail)
    except Exception:  # noqa: BLE001 - reporting must not mask the refusal
        logger.exception("Could not record the C9 start-up settings mismatch alarm.")
    keys = ", ".join(d["key"] for d in differences[:10]) or "(derived values only)"
    raise RuntimeError(
        f"Refusing to start: the resolved hyperparameters (sha256 {hp.sha256[:12]}) differ from "
        f"frozen version {frozen.version} (sha256 {frozen.sha256[:12]}) at: {keys}. "
        "Restore the frozen settings or record a protocol amendment with "
        "`flask hyperparams amend --reason \"...\"` (Hyperparameters.md §2.3)."
    )


# --------------------------------------------------------------------- CLI


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return repr(value)
    return json.dumps(value) if isinstance(value, (list, dict, bool)) or value is None else str(value)


def register_cli(app) -> None:
    """Register ``flask hyperparams show|validate|diff|amend`` (Hyperparameters.md §2.2)."""
    import click
    from flask.cli import AppGroup

    group = AppGroup("hyperparams", help="Show, validate, diff and amend the learner hyperparameters.")

    def _configured() -> Hyperparams:
        return load_hyperparams(
            app.config.get("HYPERPARAMS_PATH"), overrides=app.config.get("HYPERPARAMS_OVERRIDES")
        )

    @group.command("show")
    def show_cmd():
        """Print the resolved set, one row per key, with its source."""
        hp = getattr(app, "hyperparams", None) or _configured()
        rows = hp.rows()
        width = max(len(key) for key, _, _ in rows)
        click.echo(f"# file: {hp.source_path or '(none; design defaults)'}")
        for key, value, source in rows:
            click.echo(f"{key:<{width}}  {_format_value(value):<24}  {source}")
        click.echo(f"# sha256: {hp.sha256}")
        frozen = latest_frozen()
        if frozen is None:
            click.echo("# frozen: no (cohort warm-up; the file may still be edited)")
        else:
            state = "matches" if frozen.sha256 == hp.sha256 else "DIFFERS FROM"
            click.echo(f"# frozen: version {frozen.version} ({frozen.sha256[:12]}); running set {state} it")

    @group.command("validate")
    @click.argument("file", required=False)
    def validate_cmd(file):
        """Check FILE (default: the configured file) and print its fingerprint."""
        try:
            hp = load_hyperparams(file) if file else _configured()
        except HyperparameterError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"OK {hp.source_path or '(design defaults)'} sha256={hp.sha256}")

    @group.command("diff")
    @click.argument("file", required=False)
    def diff_cmd(file):
        """Compare FILE (default: the configured file) against the newest frozen set; exit 1 if they differ."""
        try:
            hp = load_hyperparams(file) if file else _configured()
        except HyperparameterError as exc:
            raise click.ClickException(str(exc)) from exc
        frozen = latest_frozen()
        if frozen is None:
            click.echo("Nothing is frozen yet; there is nothing to compare against.")
            return
        differences = diff_resolved(frozen.settings or {}, hp.to_dict())
        if not differences:
            click.echo(f"Identical to frozen version {frozen.version} (sha256 {frozen.sha256}).")
            return
        click.echo(f"Differs from frozen version {frozen.version}:")
        for d in differences:
            click.echo(f"  {d['key']}: frozen={_format_value(d['frozen'])} file={_format_value(d['running'])}")
        sys.exit(1)

    @group.command("amend")
    @click.option("--reason", required=True, help="Why the protocol is amended (stored with the new version).")
    @click.option("--file", "file_", required=False, help="Settings file (default: the configured file).")
    def amend_cmd(reason, file_):
        """Record a protocol amendment: store the current file as the next frozen version."""
        try:
            hp = load_hyperparams(file_) if file_ else _configured()
            row = amend(reason, hp=hp)
        except (HyperparameterError, ValueError, RuntimeError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Recorded version {row.version} sha256={row.sha256} in force from {row.in_force_from}.")

    app.cli.add_command(group)
