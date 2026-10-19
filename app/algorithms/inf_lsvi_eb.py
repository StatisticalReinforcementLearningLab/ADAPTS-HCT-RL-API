"""
``RLAlgorithm`` adapter for the ``inf_lsvi_eb`` learner (implementation
contract D1, D11, §3.1; main_new.tex Algorithms 1-4).

The adapter exists so ``current_app.rl_algorithm`` is set for the default
learner. The routes detect it with ``isinstance(..., InfLsviEbAlgorithm)`` and
call the functions of :mod:`app.learner` directly (``build_design_state``,
``standardized_g``, ``learned_probability``, ``run_weekly_update``).

The ``RLAlgorithm`` methods are kept working for code that still uses the
generic interface:

- :meth:`make_state` returns the standardized design state ``g``;
- :meth:`get_action` draws exactly one uniform ``u`` from the deterministic
  sample buffer and returns ``action = int(u < pi)`` (contract D11);
- :meth:`make_reward` scores an outcome upload by contract D6;
- :meth:`update` is not the weekly update: the route must call
  :func:`app.learner.run_weekly_update`; it returns the parameters unchanged.
"""

from __future__ import annotations

from app.algorithms.base import RLAlgorithm
from app.deterministic_sampler import DeterministicSampleStream


class InfLsviEbAlgorithm(RLAlgorithm):
    """Inf-LSVI local fits pooled by empirical Bayes, with the smooth Monte Carlo allocation."""

    name = "inf_lsvi_eb"

    def __init__(
        self,
        seed: int | None = None,
        app=None,
        sampler: DeterministicSampleStream | None = None,
    ):
        super().__init__(seed)
        if sampler is None:
            raise ValueError(
                "InfLsviEbAlgorithm requires a DeterministicSampleStream. Generate one "
                "with `flask init-buffer` and configure SAMPLE_BUFFER_PATH."
            )
        self.seed = seed
        self.app = app
        self.sampler = sampler

    def draw_action(self, action_prob: float, mode: str) -> tuple[int, dict]:
        """
        One minted decision's draw (contract D11): exactly one uniform ``u``
        from the buffer, ``action = int(u < action_prob)``. Returns
        ``(action, random_state)`` with ``{"mode", "u", "sampler_cursor_start",
        "sampler_cursor_end"}``.
        """
        start = self.sampler.cursor()
        u = self.sampler.draw_uniform()
        end = self.sampler.cursor()
        return int(u < float(action_prob)), {
            "mode": mode,
            "u": float(u),
            "sampler_cursor_start": start,
            "sampler_cursor_end": end,
        }

    def get_action(self, group_id, state, parameters, decision_type, decision_idx) -> tuple:
        """
        Draw an action at ``parameters["action_prob"]`` (alias
        ``"probability"``; 0.5 when absent). Returns ``(action, action_prob,
        random_state)``; ``random_state["model_param"]`` carries
        ``parameters["model_param"]`` when given.
        """
        parameters = parameters or {}
        prob = parameters.get("action_prob", parameters.get("probability"))
        prob = 0.5 if prob is None else float(prob)
        mode = str(parameters.get("mode") or ("learned" if "action_prob" in parameters else "fixed"))
        action, random_state = self.draw_action(prob, mode)
        if parameters.get("model_param") is not None:
            random_state["model_param"] = list(parameters["model_param"])
        return action, prob, random_state

    def update(self, old_params, data) -> tuple:
        """Not the weekly update (see :func:`app.learner.run_weekly_update`); returns ``(True, old_params)``."""
        return True, dict(old_params or {})

    def make_state(self, context) -> tuple:
        """
        ``(True, g)`` with the standardized design state of ``context``:
        ``context["design_state"]`` if present, else built from the dyad's
        latest upload (``context["group_id"]``, ``context["decision_type"]``;
        decision time ``context["decision_time"]`` or the upload's timestamp).
        ``(False, message)`` when no state can be built.
        """
        from app import learner

        try:
            decision_type = context["decision_type"]
            design_state = context.get("design_state")
            if design_state is None:
                from app.models import DataUpload

                upload = (
                    DataUpload.query.filter_by(group_id=context["group_id"])
                    .order_by(DataUpload.request_timestamp.desc(), DataUpload.id.desc())
                    .first()
                )
                if upload is None:
                    return False, "no uploaded snapshot for this dyad"
                decision_time = context.get("decision_time") or upload.request_timestamp
                design_state = learner.build_design_state(
                    context["group_id"], decision_type, upload, decision_time
                )
            return True, learner.standardized_g(decision_type, design_state)
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            return False, f"{type(exc).__name__}: {exc}"

    def make_reward(self, user_id, state, action, outcome) -> tuple:
        """
        ``(True, normalized_reward)`` for ``outcome = {"decision_type", "data",
        "aya_week_mean_level"?}`` (contract D6), else ``(False, message)``.
        """
        from app import learner

        try:
            normalized, _ = learner.compute_reward(
                outcome["decision_type"],
                outcome["data"],
                aya_week_mean_level=outcome.get("aya_week_mean_level"),
            )
            return True, normalized
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            return False, f"{type(exc).__name__}: {exc}"
