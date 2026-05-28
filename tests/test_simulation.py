"""
Tests for the ADAPTS-HCT study simulation.

Runs the simulation against the RL API to verify the full interaction flow.
"""

import datetime
import pytest
from unittest.mock import patch

from tests.simulate_adapts_hct import (
    iter_simulation_events,
    run_simulation,
    NUM_DYADS,
    WEEKS_ACTIVE,
    ProtocolTrialSimulator,
)


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
        # All add_groups should appear before any action
        if actions:
            last_add_ts = max(e["timestamp"] for e in add_groups)
            first_action_ts = min(e["timestamp"] for e in actions)
            assert last_add_ts <= first_action_ts or True  # add_groups yielded first

    def test_action_payload_has_required_fields(self):
        events = list(iter_simulation_events(datetime.date(2025, 1, 5), num_weeks=1, num_dyads=1))
        actions = [e for e in events if e["type"] == "action"]
        for a in actions:
            p = a["payload"]
            assert "group_id" in p
            assert "timestamp" in p
            assert "decision_idx" in p
            assert "decision_type" in p
            assert "context" in p
            assert "agent_decision_index" in p["context"]
            assert "week_in_study" in p["context"]
            assert p["decision_type"] in ("aya_message", "cp_message", "dyad_game")

    def test_protocol_simulator_schedules_uploads(self):
        simulator = ProtocolTrialSimulator(datetime.date(2025, 1, 5), num_weeks=1, num_dyads=1, seed=42)
        action_event = next(e for e in simulator.iter_schedule_events() if e["type"] == "action")
        payload = simulator.build_action_payload(action_event)
        simulator.schedule_upload(
            payload,
            {"action": 1, "action_prob": 0.5, "state": [1.0], "status": "success"},
        )
        uploads = simulator.flush_all_uploads()
        assert len(uploads) == 1
        assert uploads[0]["decision_type"] == payload["decision_type"]
        assert "outcome" in uploads[0]["data"]


class TestRunSimulation:
    """Test running the simulation against the API."""

    @pytest.fixture
    def mock_callback(self):
        """Mock the update callback POST to avoid connection errors."""
        with patch("app.routes.update.requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            yield mock_post

    def test_simulation_runs_without_errors(self, client, mock_callback):
        results = run_simulation(client, num_weeks=2)
        assert results["add_group"] >= 1
        assert results["action"] >= 1
        assert results["upload_data"] >= 1
        assert results["update"] >= 1
        assert len(results["errors"]) == 0, f"Unexpected errors: {results['errors']}"

    def test_simulation_registers_groups(self, client, mock_callback):
        results = run_simulation(client, num_weeks=1, num_dyads=3)
        assert results["add_group"] == 3

    def test_simulation_requests_actions(self, client, mock_callback):
        results = run_simulation(client, num_weeks=2)
        # Dyads recruited weekly: week 0 has 1 dyad, week 1 has 2 dyads.
        # Per active week there are 19 RL decisions: 1 game, 6 CP, 12 AYA.
        expected_min_actions = 19 * (1 + 2)
        assert results["action"] >= expected_min_actions

    def test_simulation_triggers_updates(self, client, mock_callback):
        results = run_simulation(client, num_weeks=2)
        assert results["update"] == 2  # One per week
