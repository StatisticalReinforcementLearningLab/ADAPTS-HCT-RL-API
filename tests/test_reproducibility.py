"""
End-to-end reproducibility tests.

Contract: given the same deterministic buffer + same ordered sequence of
API events, the algorithm must produce byte-identical (action, action_prob)
for every decision.

Two levels are covered:
1. Unit: stamp a fresh, fixed buffer and re-run the same RLSVI action twice;
   the same (action, action_prob, cursor_end) should come out both times.
2. Integration: run a small simulator against the live Flask app twice.
   The second run boots a fresh app (fresh in-memory buffer from the same
   seed via TestingConfig), replays the same events, and the action sequence
   must match bit-for-bit.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from unittest.mock import patch

from app import create_app, db
from app.deterministic_sampler import DeterministicSampleStream
from app.algorithms.empirical_bayes import ThreeAgentEmpiricalBayesAlgorithm


warmup_group = {
    "group_id": "repro_dyad_001",
    "member_list": ["aya_001", "cp_001"],
    "consent_start_date": "2025-01-05",
    "consent_end_date": "2025-04-14",
    "warmup": True,
}

standard_group = {
    "group_id": "repro_dyad_002",
    "member_list": ["aya_002", "cp_002"],
    "consent_start_date": "2025-01-05",
    "consent_end_date": "2025-04-14",
    "warmup": False,
}

aya_ctx_factory = lambda idx: {
    "slot": "am",
    "agent_decision_index": idx,
    "day_in_study": 1 + idx,
    "week_in_study": 1,
    "prior_med_adherence": "miss",
    "aya_diary": {"mood": "miss", "physical": "miss"},
    "relationship_quality_cp": "miss",
    "relationship_quality_aya": "miss",
    "aya_app_engagement": 1,
    "aya_app_burden": 0.0,
    "aya_missing_rate_7d": 1.0,
    "current_game_on": 0,
}


class TestWarmupDeterminism:
    """Same cursor → same warmup action."""

    def test_warmup_action_is_cursor_deterministic(self):
        s1 = DeterministicSampleStream.fresh(100, 50, seed=11)
        s2 = DeterministicSampleStream.fresh(100, 50, seed=11)
        seq1 = [s1.draw_bernoulli(0.5) for _ in range(30)]
        seq2 = [s2.draw_bernoulli(0.5) for _ in range(30)]
        assert seq1 == seq2

    def test_warmup_action_cursor_is_stamped_on_action(self, client):
        """Every warmup action records sampler_cursor_start/end on Action.random_state."""
        client.post("/api/v1/add_group", json=deepcopy(warmup_group))
        from app.models import Action

        payload = {
            "group_id": warmup_group["group_id"],
            "timestamp": "2025-01-06T09:00:00",
            "decision_idx": 1,
            "decision_type": "aya_message",
            "context": aya_ctx_factory(1),
        }
        r = client.post("/api/v1/action", json=payload)
        assert r.status_code == 201
        with client.application.app_context():
            row = Action.query.filter_by(group_id=warmup_group["group_id"], decision_idx=1).first()
            assert row.random_state["mode"] == "warmup"
            s = row.random_state["sampler_cursor_start"]
            e = row.random_state["sampler_cursor_end"]
            assert e["uniform"] - s["uniform"] == 1
            assert e["normal"] == s["normal"]  # warmup uses no normals


class TestNonWarmupDeterminism:
    """Standard action under probit-TS must consume ONE uniform (Bernoulli
    draw) and zero normals."""

    def test_standard_action_cursor_stamped(self, client):
        client.post("/api/v1/add_group", json=deepcopy(standard_group))
        from app.models import Action

        payload = {
            "group_id": standard_group["group_id"],
            "timestamp": "2025-01-06T09:00:00",
            "decision_idx": 1,
            "decision_type": "aya_message",
            "context": aya_ctx_factory(1),
        }
        r = client.post("/api/v1/action", json=payload)
        assert r.status_code == 201

        with client.application.app_context():
            row = Action.query.filter_by(group_id=standard_group["group_id"], decision_idx=1).first()
            assert row.random_state["mode"] == "probit_ts"
            normal_delta = (
                row.random_state["sampler_cursor_end"]["normal"]
                - row.random_state["sampler_cursor_start"]["normal"]
            )
            uniform_delta = (
                row.random_state["sampler_cursor_end"]["uniform"]
                - row.random_state["sampler_cursor_start"]["uniform"]
            )
            assert normal_delta == 0, f"probit-TS should consume 0 normals, got {normal_delta}"
            assert uniform_delta == 1, f"probit-TS should consume 1 uniform, got {uniform_delta}"


class TestClosedFormActionProb:
    """Action prob should be the exact probit-TS marginal under the block-diagonal
    prior, with inverse temperature η from ETA_BY_AGENT."""

    def test_action_prob_matches_closed_form(self, client):
        client.post("/api/v1/add_group", json=deepcopy(standard_group))
        from app.deterministic_sampler import closed_form_action_prob
        from app.feature_builder import ProtocolRLFeatureBuilder
        from app.algorithms.empirical_bayes import _prior_covariance, ETA_BY_AGENT
        import numpy as np

        payload = {
            "group_id": standard_group["group_id"],
            "timestamp": "2025-01-06T09:00:00",
            "decision_idx": 1,
            "decision_type": "aya_message",
            "context": aya_ctx_factory(1),
        }
        r = client.post("/api/v1/action", json=payload)
        data = r.get_json()

        fb = ProtocolRLFeatureBuilder("aya_message")
        state = np.asarray(data["state"], dtype=np.float64)
        # With no priors and no history the learner uses Σ_0^g from the
        # block-diagonal prior with η from ETA_BY_AGENT.
        mean = np.zeros(fb.phi_dim)
        cov = _prior_covariance("aya_message")
        eta = ETA_BY_AGENT["aya_message"]
        expected_prob_1 = closed_form_action_prob(state, mean, cov, fb.expand_base_to_phi, eta=eta)
        if data["action"] == 1:
            assert data["action_prob"] == pytest.approx(expected_prob_1)
        else:
            assert data["action_prob"] == pytest.approx(1.0 - expected_prob_1)


class TestEndToEndReproducibility:
    """Run a sequence of actions twice on a fresh app each time; the
    TestingConfig uses an in-memory buffer with a fixed seed, so the second
    run must produce the same outputs."""

    def _record_sequence(self):
        """Boot a fresh app, fire a fixed event sequence, capture outputs."""
        app = create_app("config.TestingConfig")
        client = app.test_client()
        with app.app_context():
            db.create_all()
        # Stand-in: use a fresh in-memory buffer baked from seed 7 via
        # TestingConfig.SAMPLE_BUFFER_SEED; the app sets up its own sampler.
        client.post("/api/v1/add_group", json=deepcopy(warmup_group))
        client.post("/api/v1/add_group", json=deepcopy(standard_group))

        outputs: list[tuple] = []
        for idx in range(1, 11):
            gid = warmup_group["group_id"] if idx % 2 == 0 else standard_group["group_id"]
            r = client.post(
                "/api/v1/action",
                json={
                    "group_id": gid,
                    "timestamp": f"2025-01-06T09:00:{idx:02d}",
                    "decision_idx": idx,
                    "decision_type": "aya_message",
                    "context": aya_ctx_factory(idx),
                },
            )
            body = r.get_json()
            outputs.append(
                (gid, idx, body["action"], round(body["action_prob"], 12))
            )

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
