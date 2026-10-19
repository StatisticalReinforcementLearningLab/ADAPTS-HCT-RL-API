"""
End-to-end reproducibility tests.

Contract: given the same deterministic buffer + same ordered sequence of
API events, the algorithm must produce byte-identical (action, action_prob)
for every decision.
"""
from __future__ import annotations

import datetime

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


def _setup_learned_dyad(client, gid):
    """Install a real week-2 design policy after both current warm-up gates."""
    from app.models import ModelParameters, PolicyVersion

    for i in range(5):
        register_group(client, f"seed_{i:02d}")
    if gid not in {f"seed_{i:02d}" for i in range(5)}:
        register_group(client, gid)
    assert upload(client, gid, "2026-01-13T08:00:00").status_code == 201
    with client.application.app_context():
        hp = client.application.hyperparams
        version = PolicyVersion(
            "aya_message",
            datetime.date(2026, 1, 12),
            "repro-update",
            PolicyVersion.STATUS_PUBLISHED,
            published_at=datetime.datetime(2026, 1, 12, 3, 5),
            published_at_study=datetime.datetime(2026, 1, 12, 3, 5),
            hyperparams_hash=hp.sha256,
            n_dyads=1,
        )
        db.session.add(version)
        db.session.flush()
        alpha = [0.10, -0.05, 0.08, 0.03]
        covariance = (np.eye(4) * 0.02).tolist()
        posterior = ModelParameters(
            snapshot_type="posterior",
            group_id=gid,
            decision_type="aya_message",
            theta=alpha,
            covariance=covariance,
            study_week_monday=datetime.date(2026, 1, 12),
            policy_version_id=version.id,
            update_rid="repro-update",
            hyperparams_hash=hp.sha256,
        )
        db.session.add(posterior)
        db.session.commit()
    return np.asarray(alpha), np.asarray(covariance)


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
    """A learned design action consumes one uniform and no runtime normals."""

    def test_standard_action_cursor_stamped(self, client):
        from app.models import Action

        _setup_learned_dyad(client, "repro_dyad_002")
        r = _action(client, "repro_dyad_002", 1, "aya_message", "2026-01-13T09:00:00")
        assert r.status_code == 201
        assert r.json["warmup"] is False

        with client.application.app_context():
            row = Action.query.filter_by(
                group_id="repro_dyad_002", decision_type="aya_message", decision_idx=1
            ).first()
            assert row.random_state["mode"] == "learned"
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


class TestDesignActionProb:
    """Action probability is the design's fixed-bank allocation exactly."""

    def test_action_prob_matches_fixed_bank_allocation(self, client):
        from app.learner.features import hf_from_design_state
        from app.learner.policy import probability_at

        alpha, covariance = _setup_learned_dyad(client, "repro_dyad_002")
        r = _action(client, "repro_dyad_002", 1, "aya_message", "2026-01-13T09:00:00")
        data = r.get_json()
        assert data["warmup"] is False

        from app.models import Action
        with client.application.app_context():
            row = Action.query.filter_by(
                group_id="repro_dyad_002", decision_type="aya_message", decision_idx=1
            ).first()
            _, f = hf_from_design_state("aya_message", row.raw_context["design_state"])
            expected, mu, variance = probability_at("aya_message", f, alpha, covariance)
        assert data["action_prob"] == expected
        assert row.mu == mu and row.v == variance


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
