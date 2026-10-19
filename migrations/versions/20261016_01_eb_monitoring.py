"""inf_lsvi_eb learner + monitoring schema

Implements the schema of the implementation contract §2 (main_new.tex
learner; Monitoring-Spec §9; API-Spec §5.10-5.11):

Existing tables:
- ``actions``: ``raw_context`` and ``model_parameters_id`` become nullable
  (no-state fallback / no policy scored); adds ``policy_week``,
  ``policy_version_id``, ``hyperparams_hash``, ``mu``, ``v``, ``upload_id``.
- ``model_parameters``: adds ``study_week_monday``, ``policy_version_id``
  (indexed), ``update_rid``, ``hyperparams_hash``.
- ``study_data``: ``state`` becomes nullable.
- ``model_update_requests``: adds ``study_week_monday``, ``started_at``,
  ``published_at_study``.

New tables: ``policy_versions``, ``hyperparameter_sets``, ``noise_init``,
``group_history``, ``upload_quarantine``, ``monitor_alarms``,
``api_error_events``, ``monitor_runs``, ``update_weeks``, ``monitor_pulls``.

Like the earlier revisions this one is idempotent: every step checks the live
schema first, so it is safe on a database that ``db.create_all()`` already
brought up to date. Base tables that the chain never created (the app used to
create them with ``db.create_all()``) are created here when missing, so
``flask db upgrade`` on an empty database yields the full current schema.
SQLite changes go through ``batch_alter_table``.

Revision ID: 20261016_01
Revises: 20260723_01
Create Date: 2026-10-16
"""
from alembic import op
import sqlalchemy as sa


revision = "20261016_01"
down_revision = "20260723_01"
branch_labels = None
depends_on = None


# --------------------------------------------------------------------------
# Full definitions of tables that may be missing (current schema).
# --------------------------------------------------------------------------


def _base_tables():
    """Tables of the pre-existing schema that no earlier revision creates."""
    return {
        "groups": [
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("group_id", sa.String(length=255), nullable=False, unique=True),
            sa.Column("rid", sa.String(length=255), nullable=True),
            sa.Column("group_info", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        ],
        "model_parameters": [
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("probability_of_action", sa.Float(), nullable=True),
            sa.Column("snapshot_type", sa.String(length=64), nullable=True),
            sa.Column("group_id", sa.String(length=255), nullable=True),
            sa.Column("decision_type", sa.String(length=255), nullable=True),
            sa.Column("agent_decision_index", sa.Integer(), nullable=True),
            sa.Column("sample_size", sa.Integer(), nullable=True),
            sa.Column("feature_dim", sa.Integer(), nullable=True),
            sa.Column("theta", sa.JSON(), nullable=True),
            sa.Column("covariance", sa.JSON(), nullable=True),
            sa.Column("perturbation", sa.JSON(), nullable=True),
            sa.Column("metadata_json", sa.JSON(), nullable=True),
            sa.Column("study_week_monday", sa.Date(), nullable=True),
            sa.Column("policy_version_id", sa.Integer(), nullable=True, index=True),
            sa.Column("update_rid", sa.String(length=255), nullable=True),
            sa.Column("hyperparams_hash", sa.String(length=64), nullable=True),
            sa.Column("timestamp", sa.DateTime(), nullable=False),
        ],
        "actions": [
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("group_id", sa.String(length=255), nullable=False),
            sa.Column("rid", sa.String(length=255), nullable=False, unique=True),
            sa.Column("state", sa.JSON(), nullable=True),
            sa.Column("decision_idx", sa.Integer(), nullable=False),
            sa.Column("decision_type", sa.String(length=255), nullable=False),
            sa.Column("raw_context", sa.JSON(), nullable=True),
            sa.Column("action", sa.Integer(), nullable=False),
            sa.Column("action_prob", sa.Float(), nullable=False),
            sa.Column("is_warmup", sa.Boolean(), nullable=False),
            sa.Column("warmup_reason", sa.String(length=32), nullable=True),
            sa.Column("model_param", sa.JSON(), nullable=True),
            sa.Column("random_state", sa.JSON(), nullable=False),
            sa.Column(
                "model_parameters_id",
                sa.Integer(),
                sa.ForeignKey("model_parameters.id"),
                nullable=True,
            ),
            sa.Column("policy_week", sa.Date(), nullable=True),
            sa.Column("policy_version_id", sa.Integer(), nullable=True),
            sa.Column("hyperparams_hash", sa.String(length=64), nullable=True),
            sa.Column("mu", sa.Float(), nullable=True),
            sa.Column("v", sa.Float(), nullable=True),
            sa.Column("upload_id", sa.Integer(), nullable=True),
            sa.Column("request_timestamp", sa.DateTime(), nullable=False),
            sa.Column("timestamp", sa.DateTime(), nullable=False),
            sa.UniqueConstraint(
                "group_id", "decision_type", "decision_idx", name="uq_action_group_type_idx"
            ),
        ],
        "thompson_sampling_params": [
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("group_id", sa.String(length=255), nullable=False),
            sa.Column("decision_type", sa.String(length=255), nullable=False),
            sa.Column("params", sa.JSON(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("group_id", "decision_type", name="uq_group_decision"),
        ],
        "model_update_requests": [
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("rid", sa.String(length=255), nullable=False, unique=True),
            sa.Column("status", sa.String(length=50), nullable=False),
            sa.Column("request_timestamp", sa.DateTime(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.Column("error_message", sa.String(length=1024), nullable=True),
            sa.Column("study_week_monday", sa.Date(), nullable=True),
            sa.Column("started_at", sa.DateTime(), nullable=True),
            sa.Column("published_at_study", sa.DateTime(), nullable=True),
        ],
        "study_data": [
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("group_id", sa.String(length=255), nullable=False),
            sa.Column("decision_idx", sa.Integer(), nullable=False),
            sa.Column("decision_type", sa.String(length=255), nullable=False),
            sa.Column("action", sa.Integer(), nullable=False),
            sa.Column("action_prob", sa.Float(), nullable=False),
            sa.Column("state", sa.JSON(), nullable=True),
            sa.Column("raw_context", sa.JSON(), nullable=False),
            sa.Column("outcome", sa.JSON(), nullable=False),
            sa.Column("reward", sa.Float(), nullable=True),
            sa.Column("request_timestamp", sa.DateTime(), nullable=False),
            sa.Column("derived_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint(
                "group_id", "decision_type", "decision_idx", name="uq_study_group_type_idx"
            ),
        ],
        "update_reproducibility_snapshots": [
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("rid", sa.String(length=255), nullable=False),
            sa.Column("model_parameters_id", sa.Integer(), nullable=True),
            sa.Column("snapshot_dir", sa.String(length=2048), nullable=False),
            sa.Column("data_uploads_count", sa.Integer(), nullable=False),
            sa.Column("actions_count", sa.Integer(), nullable=False),
            sa.Column("groups_count", sa.Integer(), nullable=False),
            sa.Column("total_bytes", sa.BigInteger(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        ],
    }


def _new_tables():
    """Tables introduced by this revision (contract §2)."""
    return {
        "policy_versions": [
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("decision_type", sa.String(length=255), nullable=False),
            sa.Column("study_week_monday", sa.Date(), nullable=False),
            sa.Column("update_rid", sa.String(length=255), nullable=False),
            sa.Column("status", sa.String(length=16), nullable=False),
            sa.Column("reason", sa.Text(), nullable=True),
            sa.Column("published_at", sa.DateTime(), nullable=True),
            sa.Column("published_at_study", sa.DateTime(), nullable=True),
            sa.Column("hyperparams_hash", sa.String(length=64), nullable=True),
            sa.Column("n_dyads", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Index("ix_policy_versions_type_week", "decision_type", "study_week_monday"),
        ],
        "hyperparameter_sets": [
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("version", sa.Integer(), nullable=False, unique=True),
            sa.Column("sha256", sa.String(length=64), nullable=False),
            sa.Column("settings", sa.JSON(), nullable=False),
            sa.Column("frozen_at", sa.DateTime(), nullable=False),
            sa.Column("in_force_from", sa.DateTime(), nullable=False),
            sa.Column("reason", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        ],
        "noise_init": [
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("decision_type", sa.String(length=255), nullable=False),
            sa.Column("sigma_sq", sa.Float(), nullable=False),
            sa.Column("sigma_bar_sq", sa.Float(), nullable=False),
            sa.Column("tau0_act", sa.Float(), nullable=False),
            sa.Column("tau0_mod", sa.Float(), nullable=False),
            sa.Column("source", sa.String(length=32), nullable=False),
            sa.Column("n_week1_rewards", sa.Integer(), nullable=True),
            sa.Column("s2", sa.Float(), nullable=True),
            sa.Column("clip_used", sa.Boolean(), nullable=False),
            sa.Column("study_week_monday", sa.Date(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        ],
        "group_history": [
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("group_id", sa.String(length=255), nullable=False, index=True),
            sa.Column("member_list", sa.JSON(), nullable=False),
            sa.Column("consent_start_date", sa.String(length=10), nullable=False),
            sa.Column("consent_end_date", sa.String(length=10), nullable=False),
            sa.Column("rid", sa.String(length=255), nullable=True),
            sa.Column("recorded_at", sa.DateTime(), nullable=False),
        ],
        "upload_quarantine": [
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("rid", sa.String(length=255), nullable=False),
            sa.Column("group_id", sa.String(length=255), nullable=True),
            sa.Column("raw_body", sa.LargeBinary(), nullable=False),
            sa.Column("validation_error", sa.Text(), nullable=False),
            sa.Column("payload_sha256", sa.String(length=64), nullable=False),
            sa.Column("payload_bytes", sa.Integer(), nullable=False),
            sa.Column("received_at", sa.DateTime(), nullable=False),
        ],
        "monitor_alarms": [
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("event_id", sa.String(length=64), nullable=False, unique=True),
            sa.Column("item_id", sa.String(length=64), nullable=False),
            sa.Column("severity", sa.String(length=16), nullable=False),
            sa.Column("owner", sa.String(length=16), nullable=False),
            sa.Column("raised_at", sa.DateTime(), nullable=False, index=True),
            sa.Column("source", sa.String(length=32), nullable=False),
            sa.Column("run_id", sa.Integer(), nullable=True),
            sa.Column("group_id", sa.String(length=255), nullable=True),
            sa.Column("decision_type", sa.String(length=255), nullable=True),
            sa.Column("decision_idx", sa.Integer(), nullable=True),
            sa.Column("rid", sa.String(length=255), nullable=True),
            sa.Column("message", sa.Text(), nullable=False),
            sa.Column("value", sa.Float(), nullable=True),
            sa.Column("threshold", sa.Float(), nullable=True),
            sa.Column("detail", sa.JSON(), nullable=False),
            sa.Column("fallback_used", sa.String(length=255), nullable=True),
            sa.Column("note", sa.Text(), nullable=True),
            sa.Column("note_by", sa.String(length=255), nullable=True),
            sa.Column("noted_at", sa.DateTime(), nullable=True),
            sa.Column("check_list_version", sa.String(length=16), nullable=False),
            sa.Column("dedup_key", sa.String(length=255), nullable=True, unique=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        ],
        "api_error_events": [
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("event_id", sa.String(length=64), nullable=False, unique=True),
            sa.Column("code", sa.String(length=64), nullable=False),
            sa.Column("endpoint", sa.String(length=255), nullable=False),
            sa.Column("http_status", sa.Integer(), nullable=False),
            sa.Column("title", sa.String(length=255), nullable=True),
            sa.Column("message", sa.Text(), nullable=True),
            sa.Column("rid", sa.String(length=255), nullable=True),
            sa.Column("group_id", sa.String(length=255), nullable=True),
            sa.Column("decision_type", sa.String(length=255), nullable=True),
            sa.Column("decision_idx", sa.Integer(), nullable=True),
            sa.Column("payload_sha256", sa.String(length=64), nullable=True),
            sa.Column("payload_bytes", sa.Integer(), nullable=True),
            sa.Column("occurred_at", sa.DateTime(), nullable=False, index=True),
        ],
        "monitor_runs": [
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("kind", sa.String(length=16), nullable=False),
            sa.Column("detail", sa.JSON(), nullable=True),
            sa.Column("item_id", sa.String(length=64), nullable=True),
            sa.Column("window_start", sa.DateTime(), nullable=True),
            sa.Column("window_end", sa.DateTime(), nullable=True),
            sa.Column("started_at", sa.DateTime(), nullable=False),
            sa.Column("finished_at", sa.DateTime(), nullable=True),
            sa.Column("status", sa.String(length=16), nullable=False),
            sa.Column("n_alarms", sa.Integer(), nullable=False),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("logical_now", sa.DateTime(), nullable=True),
        ],
        "update_weeks": [
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("study_week_monday", sa.Date(), nullable=False, unique=True),
            sa.Column("rids", sa.JSON(), nullable=False),
            sa.Column("status", sa.String(length=16), nullable=False),
            sa.Column("first_call_at", sa.DateTime(), nullable=True),
            sa.Column("finished_at", sa.DateTime(), nullable=True),
            sa.Column("ready_at", sa.DateTime(), nullable=True),
            sa.Column("missing_closing_uploads", sa.JSON(), nullable=True),
            sa.Column("c7_raised", sa.Boolean(), nullable=False),
        ],
        "monitor_pulls": [
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("rid", sa.String(length=255), nullable=False, unique=True),
            sa.Column("window_start", sa.DateTime(), nullable=True),
            sa.Column("returned_count", sa.Integer(), nullable=True),
            sa.Column("status", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        ],
    }


# Columns added to tables that already exist: table -> [(name, type)].
_ADDED_COLUMNS = {
    "actions": [
        ("policy_week", sa.Date()),
        ("policy_version_id", sa.Integer()),
        ("hyperparams_hash", sa.String(length=64)),
        ("mu", sa.Float()),
        ("v", sa.Float()),
        ("upload_id", sa.Integer()),
    ],
    "model_parameters": [
        ("study_week_monday", sa.Date()),
        ("policy_version_id", sa.Integer()),
        ("update_rid", sa.String(length=255)),
        ("hyperparams_hash", sa.String(length=64)),
    ],
    "model_update_requests": [
        ("study_week_monday", sa.Date()),
        ("started_at", sa.DateTime()),
        ("published_at_study", sa.DateTime()),
    ],
}

# Columns that become nullable: table -> [(name, existing type)].
_RELAXED_COLUMNS = {
    "actions": [("raw_context", sa.JSON()), ("model_parameters_id", sa.Integer())],
    "study_data": [("state", sa.JSON())],
}

_MP_POLICY_INDEX = "ix_model_parameters_policy_version_id"

# Creation order honours the actions -> model_parameters foreign key.
_BASE_ORDER = (
    "groups",
    "model_parameters",
    "actions",
    "thompson_sampling_params",
    "model_update_requests",
    "study_data",
    "update_reproducibility_snapshots",
)


def _columns(inspector, table):
    return {c["name"]: c for c in inspector.get_columns(table)}


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    # 1. Base tables missing from an empty database.
    base = _base_tables()
    for name in _BASE_ORDER:
        if name not in tables:
            op.create_table(name, *base[name])
            tables.add(name)
        else:
            _upgrade_existing(name, inspector)

    # 2. New tables.
    for name, items in _new_tables().items():
        if name not in tables:
            op.create_table(name, *items)


def _upgrade_existing(name, inspector):
    """Add the new columns / relax nullability / add indexes on an existing table."""
    cols = _columns(inspector, name)
    adds = [(c, t) for c, t in _ADDED_COLUMNS.get(name, []) if c not in cols]
    relax = [(c, t) for c, t in _RELAXED_COLUMNS.get(name, []) if c in cols and not cols[c]["nullable"]]
    need_index = False
    if name == "model_parameters":
        existing_indexes = {ix["name"] for ix in inspector.get_indexes(name)}
        need_index = _MP_POLICY_INDEX not in existing_indexes
    if not (adds or relax or need_index):
        return
    with op.batch_alter_table(name) as batch_op:
        for col_name, col_type in adds:
            batch_op.add_column(sa.Column(col_name, col_type, nullable=True))
        for col_name, col_type in relax:
            batch_op.alter_column(col_name, existing_type=col_type, nullable=True)
        if need_index:
            batch_op.create_index(_MP_POLICY_INDEX, ["policy_version_id"], unique=False)


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    for name in reversed(list(_new_tables())):
        if name in tables:
            op.drop_table(name)

    for name in ("model_update_requests", "model_parameters", "actions"):
        if name not in tables:
            continue
        cols = _columns(inspector, name)
        drops = [c for c, _ in _ADDED_COLUMNS[name] if c in cols]
        drop_index = name == "model_parameters" and _MP_POLICY_INDEX in {
            ix["name"] for ix in inspector.get_indexes(name)
        }
        if not (drops or drop_index):
            continue
        with op.batch_alter_table(name) as batch_op:
            if drop_index:
                batch_op.drop_index(_MP_POLICY_INDEX)
            for col_name in drops:
                batch_op.drop_column(col_name)

    # Restore NOT NULL only where no NULL rows were written meanwhile.
    for name, relaxed in _RELAXED_COLUMNS.items():
        if name not in tables:
            continue
        cols = _columns(inspector, name)
        restore = []
        for col_name, col_type in relaxed:
            if col_name not in cols or not cols[col_name]["nullable"]:
                continue
            n_null = bind.execute(
                sa.text(f"SELECT COUNT(*) FROM {name} WHERE {col_name} IS NULL")
            ).scalar()
            if not n_null:
                restore.append((col_name, col_type))
        if restore:
            with op.batch_alter_table(name) as batch_op:
                for col_name, col_type in restore:
                    batch_op.alter_column(col_name, existing_type=col_type, nullable=False)
