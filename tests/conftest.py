import pytest
from app import create_app, db
from app.protocol import SNAPSHOT_SCHEMA

@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    app_instance = create_app("config.TestingConfig")

    with app_instance.app_context():
        # Initialize database or other setup here if needed
        yield app_instance
        db.session.remove()
        db.drop_all()  # Drop all tables after the test is completed

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


# --- shared helpers for the flat-snapshot contract (API-Spec §5.1) ---

def full_snapshot(**overrides):
    """A complete /upload_data `data` snapshot (every field present)."""
    snap = {
        "day_in_study": 1,
        "week_in_study": 1,
        "slot": "am",
        "aya_diary_mood": "miss",
        "aya_diary_physical": "miss",
        "aya_app_engagement": 2,
        "aya_app_burden": 0.0,
        "aya_missing_rate_7d": 0.0,
        "previous_med_adherence": "miss",
        "prompted_by_message": False,
        "cp_diary_mood": "miss",
        "cp_app_engagement": 2,
        "cp_app_burden": 0.0,
        "cp_missing_rate_7d": 0.0,
        "daily_diary_completed": False,
        "daily_diary_score": 0.0,
        "relationship_quality_aya": "miss",
        "relationship_quality_cp": "miss",
        "current_game_on": 1,
        "prior_game_action": "miss",
        "aya_diary_summary": "miss",
        "cp_diary_summary": "miss",
        "weekly_survey_completed": False,
        "weekly_relationship_score": 0.0,
    }
    assert set(snap) == set(SNAPSHOT_SCHEMA), "full_snapshot drifted from SNAPSHOT_SCHEMA"
    snap.update(overrides)
    return snap


def register_group(client, group_id, start="2026-01-05", end="2026-04-15"):
    return client.post(
        "/api/v1/add_group",
        json={
            "group_id": group_id,
            "member_list": [f"aya_{group_id}", f"cp_{group_id}"],
            "consent_start_date": start,
            "consent_end_date": end,
        },
    )


def upload(client, group_id, timestamp, **overrides):
    return client.post(
        "/api/v1/upload_data",
        json={
            "group_id": group_id,
            "timestamp": timestamp,
            "data": full_snapshot(**overrides),
        },
    )
