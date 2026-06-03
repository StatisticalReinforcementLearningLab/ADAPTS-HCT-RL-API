import numpy as np

from app.feature_builder import ProtocolRLFeatureBuilder, phi_dims_by_decision_type


def test_phi_expands_base_correctly():
    fb = ProtocolRLFeatureBuilder("aya_message")
    ctx = {
        "slot": "am",
        "agent_decision_index": 1,
        "day_in_study": 5,
        "week_in_study": 1,
        "prior_med_adherence": "miss",
        "aya_diary": {"mood": "miss", "physical": "miss"},
        "relationship_quality_cp": "miss",
        "relationship_quality_aya": "miss",
        "aya_app_engagement": 2,
        "aya_app_burden": 1.0,
        "aya_missing_rate_7d": 0.5,
        "current_game_on": 0,
    }
    u = fb.base_vector(ctx)
    phi0 = fb.expand_base_to_phi(u, 0)
    phi1 = fb.expand_base_to_phi(u, 1)
    assert u.shape[0] == fb.base_dim
    assert phi0.shape[0] == fb.phi_dim == phi_dims_by_decision_type()["aya_message"]
    assert np.allclose(phi0[:2], [1.0, 0.0])
    assert np.allclose(phi1[:2], [1.0, 1.0])


def test_each_missable_variable_zeroes_value_terms_when_missing():
    fb = ProtocolRLFeatureBuilder("aya_message")
    ctx = {
        "slot": "am",
        "agent_decision_index": 1,
        "day_in_study": 5,
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
    u = fb.base_vector(ctx)
    # first missable is prior_med_adherence at pair index 3 (0=intercept, 1-2 slot, 3-4 day, 5-6 week, 7-8 prior)
    # intercept + slot(2) + day(2) + week(2) = 7, prior starts at index 7
    assert u[7] == 0.0 and u[8] == 0.0


def test_standardization_applies_to_continuous_variables_only():
    fb = ProtocolRLFeatureBuilder("aya_message")
    ctx = {
        "slot": "am",
        "agent_decision_index": 1,
        "day_in_study": 5,
        "week_in_study": 2,
        "prior_med_adherence": 1,
        "aya_diary": {"mood": 3, "physical": 4},
        "relationship_quality_cp": 3,
        "relationship_quality_aya": 4,
        "aya_app_engagement": 2,
        "aya_app_burden": 2.0,  # scaled => 0.2 inside builder
        "aya_missing_rate_7d": 0.6,
        "current_game_on": 1,
    }
    baselines = {
        "aya_app_burden": {"mu": 0.2, "sigma": 0.1},
        "aya_missing_rate_7d": {"mu": 0.5, "sigma": 0.1},
        # ordinal variables intentionally not in baselines
    }
    u_raw = fb.base_vector(ctx)
    u_std = fb.base_vector(ctx, baselines=baselines)
    # The intercept and ordinal value cells should be unchanged.
    assert np.isclose(u_std[0], u_raw[0])
    # Single shared indicator layout: u = [1, I, v_1*I, ..., v_J*I], so
    # variable k's value sits at position 2 + k.
    names = fb.variable_names
    burden_pair = 2 + names.index("aya_app_burden")
    mr_pair = 2 + names.index("aya_missing_rate_7d")
    assert not np.isclose(u_std[burden_pair], u_raw[burden_pair])
    assert not np.isclose(u_std[mr_pair], u_raw[mr_pair])
    # Expected: (0.2 - 0.2) / 0.1 = 0.0  and  (0.6 - 0.5) / 0.1 = 1.0
    assert np.isclose(u_std[burden_pair], 0.0)
    assert np.isclose(u_std[mr_pair], 1.0)


def test_dyad_game_uses_five_tailoring_vars():
    fb = ProtocolRLFeatureBuilder("dyad_game")
    ctx = {
        "agent_decision_index": 1,
        "week_in_study": 1,
        "aya_app_engagement": 2,
        "cp_app_engagement": 3,
        "aya_app_burden": 0.5,
        "cp_app_burden": 0.2,
        "prior_game_action": "miss",
    }
    u = fb.base_vector(ctx)
    assert u.shape[0] == fb.base_dim
    # The pruned dyad_game state is the five tailoring variables only — no
    # diary summaries, no relationship_quality (API-Spec §5.2).
    assert fb.variable_names == [
        "aya_app_engagement",
        "cp_app_engagement",
        "aya_app_burden",
        "cp_app_burden",
        "prior_game_action",
    ]
    assert "aya_diary_summary" not in fb.variable_names
    assert "cp_diary_summary" not in fb.variable_names
