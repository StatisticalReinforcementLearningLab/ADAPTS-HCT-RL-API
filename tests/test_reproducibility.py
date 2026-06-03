"""
End-to-end reproducibility tests.

Contract: given the same deterministic buffer + same ordered sequence of
API events, the algorithm must produce byte-identical (action, action_prob)
for every decision.
"""
from __future__ import annotations

import numpy as np
import pytest

from app import create_app, db
from app.deterministic_sampler import DeterministicSampleStream
from tests.conftest import register_group, upload


def _action(client, gid, idx, dt, ts):
    return client.post(
        "/api/v1/action",
        json={"group_id": gid, "timestamp": ts, "decision_idx": idx, "decision_type": dt},
    )


def _setup_nonwarmup_dyad(client, gid):
    """Register enough dyads to clear the cohort gate and rack up 6 cp_message
    decisions for `gid` so it clears the week-1 gate (API-Spec §3.2)."""
    for i in range(5):
        register_group(client, f"seed_{i:02d}")
    register_group(client, gid)
    for k in range(6):
        ts_day = f"2026-01-{5 + k:02d}"
        upload(client, gid, f"{ts_day}T05:00:00", slot="am", day_in_study=k + 1)
        r = _action(client, gid, k, "cp_message", f"{ts_day}T06:00:00")
        assert r.status_code == 201


class TestWarmupDeterminism:
    """Same cursor → same warmup action."""

    def test_warmup_action_is_cursor_deterministic(self):
        s1 = DeterministicSampleStream.fresh(100, 50, seed=11)
        s2 = DeterministicSampleStream.fresh(100, 50, seed=11)
        seq1 = [s1.draw_bernoulli(0.5) for _ in range(30)]
        seq2 = [s2.draw_bernoulli(0.5) for _ in range(30)]
        assert seq1 == seq2

    def test_warmup_action_cursor_is_stamped_on_action(self, client):
        """A warmup action records sampler_cursor_start/end on Action.random_state."""
        from app.models import Action

        register_group(client, "repro_dyad_001")
        upload(client, "repro_dyad_001", "2026-01-06T08:00:00")
        r = _action(client, "repro_dyad_001", 1, "aya_message", "2026-01-06T09:00:00")
        assert r.status_code == 201
        assert r.json["warmup"] is True
        with client.application.app_context():
            row = Action.query.filter_by(
                group_id="repro_dyad_001", decision_idx=1
            ).first()
            assert row.random_state["mode"] == "warmup"
            s = row.random_state["sampler_cursor_start"]
            e = row.random_state["sampler_cursor_end"]
            assert e["uniform"] - s["uniform"] == 1
            assert e["normal"] == s["normal"]  # warmup uses no normals


class TestNonWarmupDeterminism:
    """A non-warmup action under probit-TS consumes ONE uniform and zero normals."""

    def test_standard_action_cursor_stamped(self, client):
        from app.models import Action

        _setup_nonwarmup_dyad(client, "repro_dyad_002")
        upload(client, "repro_dyad_002", "2026-01-12T08:00:00", day_in_study=7)
        r = _action(client, "repro_dyad_002", 1, "aya_message", "2026-01-12T09:00:00")
        assert r.status_code == 201
        assert r.json["warmup"] is False

        with client.application.app_context():
            row = Action.query.filter_by(
                group_id="repro_dyad_002", decision_type="aya_message", decision_idx=1
            ).first()
            assert row.random_state["mode"] == "probit_ts"
            normal_delta = (
                row.random_state["sampler_cursor_end"]["normal"]
                - row.random_state["sampler_cursor_start"]["normal"]
            )
            uniform_delta = (
                row.random_state["sampler_cursor_end"]["uniform"]
                - row.random_state["sampler_cursor_start"]["uniform"]
            )
            assert normal_delta == 0
            assert uniform_delta == 1


class TestClosedFormActionProb:
    """Action prob is the exact probit-TS marginal under the block-diagonal prior."""

    def test_action_prob_matches_closed_form(self, client):
        from app.deterministic_sampler import closed_form_action_prob
        from app.feature_builder import ProtocolRLFeatureBuilder
        from app.algorithms.empirical_bayes import _prior_covariance, ETA_BY_AGENT

        _setup_nonwarmup_dyad(client, "repro_dyad_002")
        upload(client, "repro_dyad_002", "2026-01-12T08:00:00", day_in_study=7)
        r = _action(client, "repro_dyad_002", 1, "aya_message", "2026-01-12T09:00:00")
        data = r.get_json()
        assert data["warmup"] is False

        fb = ProtocolRLFeatureBuilder("aya_message")
        state = np.asarray(data["state"], dtype=np.float64)
        mean = np.zeros(fb.phi_dim)
        cov = _prior_covariance("aya_message")
        eta = ETA_BY_AGENT["aya_message"]
        expected_prob_1 = closed_form_action_prob(
            state, mean, cov, fb.expand_base_to_phi, eta=eta
        )
        if data["action"] == 1:
            assert data["action_prob"] == pytest.approx(expected_prob_1)
        else:
            assert data["action_prob"] == pytest.approx(1.0 - expected_prob_1)


class TestEndToEndReproducibility:
    """Run a fixed event sequence twice on a fresh app each time; the in-memory
    buffer is baked from the same seed, so outputs must match bit-for-bit."""

    def _record_sequence(self):
        app = create_app("config.TestingConfig")
        client = app.test_client()
        register_group(client, "repro_dyad_001")
        register_group(client, "repro_dyad_002")

        outputs: list[tuple] = []
        for idx in range(1, 11):
            gid = "repro_dyad_001" if idx % 2 == 0 else "repro_dyad_002"
            ts = f"2026-01-06T09:00:{idx:02d}"
            upload(client, gid, f"2026-01-06T08:00:{idx:02d}", day_in_study=idx)
            r = _action(client, gid, idx, "aya_message", ts)
            body = r.get_json()
            outputs.append((gid, idx, body["action"], round(body["action_prob"], 12)))

        with app.app_context():
            db.session.remove()
            db.drop_all()
        return outputs

    def test_two_fresh_runs_produce_same_sequence(self):
        first = self._record_sequence()
        second = self._record_sequence()
        assert first == second, (
            "Two runs on identical buffers / events produced different outputs:\n"
            f"first={first}\nsecond={second}"
        )
