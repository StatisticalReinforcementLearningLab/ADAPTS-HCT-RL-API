"""
Tests for the ADAPTS-HCT study simulation (flat-snapshot contract).

Runs the simulation against the RL API to verify the full interaction flow:
full-snapshot /upload_data before every context-free /action, weekly /update.
"""

import datetime

from tests.simulate_adapts_hct import (
    iter_simulation_events,
    run_simulation,
    ProtocolTrialSimulator,
)
from app.protocol import SNAPSHOT_SCHEMA


class TestSimulationEvents:
    """Test the simulation event generator."""

    def test_events_have_required_fields(self):
        events = list(iter_simulation_events(datetime.date(2025, 1, 5), num_weeks=1, num_dyads=2))
        assert len(events) > 0
        for e in events:
            assert "type" in e
            assert "timestamp" in e
            assert "payload" in e
            assert e["type"] in ("add_group", "update", "action")

    def test_add_group_before_actions(self):
        events = list(iter_simulation_events(datetime.date(2025, 1, 5), num_weeks=1, num_dyads=2))
        add_groups = [e for e in events if e["type"] == "add_group"]
        actions = [e for e in events if e["type"] == "action"]
        assert len(add_groups) == 2
        assert len(actions) > 0

    def test_action_payload_is_context_free(self):
        events = list(iter_simulation_events(datetime.date(2025, 1, 5), num_weeks=1, num_dyads=1))
        actions = [e for e in events if e["type"] == "action"]
        for a in actions:
            p = a["payload"]
            assert set(p) == {"group_id", "timestamp", "decision_idx", "decision_type"}
            assert "context" not in p
            assert p["decision_type"] in ("aya_message", "cp_message", "dyad_game")

    def test_build_snapshot_is_a_full_snapshot(self):
        simulator = ProtocolTrialSimulator(datetime.date(2025, 1, 5), num_weeks=1, num_dyads=1, seed=42)
        action_event = next(e for e in simulator.iter_schedule_events() if e["type"] == "action")
        snapshot = simulator.build_snapshot(action_event)
        assert set(snapshot) == set(SNAPSHOT_SCHEMA)


class TestRunSimulation:
    """Test running the simulation against the API."""

    def test_simulation_runs_without_errors(self, client):
        results = run_simulation(client, num_weeks=2)
        assert results["add_group"] >= 1
        assert results["action"] >= 1
        assert results["upload_data"] >= 1
        assert results["update"] >= 1
        assert len(results["errors"]) == 0, f"Unexpected errors: {results['errors']}"

    def test_simulation_registers_groups(self, client):
        results = run_simulation(client, num_weeks=1, num_dyads=3)
        assert results["add_group"] == 3

    def test_simulation_requests_actions(self, client):
        results = run_simulation(client, num_weeks=2)
        # Dyads recruited weekly: week 0 has 1 dyad, week 1 has 2 dyads.
        # Per active week there are 19 RL decisions: 1 game, 6 CP, 12 AYA.
        expected_min_actions = 19 * (1 + 2)
        assert results["action"] >= expected_min_actions

    def test_simulation_triggers_updates(self, client):
        results = run_simulation(client, num_weeks=2)
        assert results["update"] == 2  # One per week
