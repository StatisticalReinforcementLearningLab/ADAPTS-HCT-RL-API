"""Quick diagnostic: run the same sanity-check simulation and dump the final
EB pool (hyper-mean, hyper-covariance) plus how much per-dyad posterior
variance ends up in $\\Delta\\phi^\\top \\Sigma_i^{\\mathrm{post}} \\Delta\\phi$
at a representative state, so we can see why $\\pi$ for fresh dyads sits near
0.5 even when the pool has many dyads of data.

Run from ADAPTS-HCT-RL-API/."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import TestingConfig
TestingConfig.SAMPLE_BUFFER_UNIFORMS = 50_000
TestingConfig.SAMPLE_BUFFER_NORMALS = 2_000_000

import app.routes.update as u
import threading
class _SyncThread(threading.Thread):
    def start(self): self.run()
u.Thread = _SyncThread

import requests as _r
_r.post = lambda *a, **kw: type("R", (), {"status_code": 200, "text": ""})()

from app import create_app, db
from app.models import ModelParameters, Action
from tests.simulate_adapts_hct import run_simulation
import logging
logging.disable(logging.CRITICAL)


def main():
    app = create_app("config.TestingConfig")
    with app.app_context():
        db.create_all()
        run_simulation(
            app.test_client(),
            base_date=datetime.date(2025, 1, 5),
            num_weeks=35,
            num_dyads=25,
            verbose=False,
        )

        for agent in ("aya_message", "cp_message", "dyad_game"):
            hyper = (
                ModelParameters.query
                .filter_by(snapshot_type="hyper", decision_type=agent)
                .order_by(ModelParameters.agent_decision_index.desc())
                .first()
            )
            if hyper is None:
                print(f"\n=== {agent}: no hyper snapshot ===")
                continue
            theta0 = np.asarray(hyper.theta, dtype=np.float64)
            Sigma0 = np.asarray(hyper.covariance, dtype=np.float64)
            D = theta0.size
            print(f"\n=== {agent}  (D={D}, sample_size={hyper.sample_size}) ===")
            print(f"theta_0[:2] (intercept, action) = {theta0[:2]}")
            print(f"theta_0[2:6] (state main first 4): {theta0[2:6]}")
            interact_start = 2 + (D - 2) // 2
            print(f"theta_0[{interact_start}:{interact_start+4}] (action*state first 4): "
                  f"{theta0[interact_start:interact_start+4]}")
            diag = np.diag(Sigma0)
            print(f"Sigma_0 diag, action coef = {diag[1]:.4g}; "
                  f"mean(state-main) = {diag[2:interact_start].mean():.4g}; "
                  f"mean(action*state) = {diag[interact_start:].mean():.4g}")

            # Δφ for a representative state of all zeros + one missing
            # indicator on. Just for a rough sense of the probit-TS damping.
            phi1 = np.zeros(D); phi1[0] = 1; phi1[1] = 1
            phi0 = np.zeros(D); phi0[0] = 1; phi0[1] = 0
            # mimic interaction terms = action-only block
            dphi = phi1 - phi0
            m = dphi @ theta0
            v = dphi @ Sigma0 @ dphi
            from math import erf
            pi = 0.5 * (1 + erf((m / np.sqrt(1 + v)) / np.sqrt(2)))
            print(f"plug-in pop π at zero-state: m={m:.4g}, v={v:.4g}, π={pi:.4g}")

        # A few late EB-policy decisions to see how the per-dyad posteriors look.
        late = (
            Action.query.filter(Action.decision_type == "aya_message")
            .order_by(Action.id.desc())
            .limit(40).all()
        )
        for a in reversed(late[:8]):
            rs = a.random_state or {}
            print(
                f"AYA {a.group_id} d_idx={a.decision_idx} π={a.action_prob:.4g} "
                f"mode={rs.get('mode')} src={rs.get('source')}"
            )


if __name__ == "__main__":
    main()
