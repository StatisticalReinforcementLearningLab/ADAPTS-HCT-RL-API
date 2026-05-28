"""merge empirical_bayes_snapshots into model_parameters

Absorbs the EB snapshot table into ``model_parameters``:
- Adds snapshot_type / group_id / decision_type / agent_decision_index /
  sample_size / feature_dim / theta / covariance / perturbation / metadata_json
  columns to ``model_parameters``.
- Makes ``probability_of_action`` nullable (it's NULL for EB snapshot rows).
- Drops the standalone ``empirical_bayes_snapshots`` table.

Revision ID: 20260528_02
Revises: 20260528_01
Create Date: 2026-05-28
"""
from alembic import op
import sqlalchemy as sa


revision = "20260528_02"
down_revision = "20260528_01"
branch_labels = None
depends_on = None


_NEW_COLUMNS = [
    ("snapshot_type", sa.String(length=64), True),
    ("group_id", sa.String(length=255), True),
    ("decision_type", sa.String(length=255), True),
    ("agent_decision_index", sa.Integer(), True),
    ("sample_size", sa.Integer(), True),
    ("feature_dim", sa.Integer(), True),
    ("theta", sa.JSON(), True),
    ("covariance", sa.JSON(), True),
    ("perturbation", sa.JSON(), True),
    ("metadata_json", sa.JSON(), True),
]


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "model_parameters" in inspector.get_table_names():
        existing_cols = {c["name"] for c in inspector.get_columns("model_parameters")}
        with op.batch_alter_table("model_parameters") as batch_op:
            for name, col_type, nullable in _NEW_COLUMNS:
                if name not in existing_cols:
                    batch_op.add_column(sa.Column(name, col_type, nullable=nullable))
            # probability_of_action is now nullable (NULL for EB snapshot rows).
            batch_op.alter_column("probability_of_action", existing_type=sa.Float(), nullable=True)

    if "empirical_bayes_snapshots" in inspector.get_table_names():
        op.drop_table("empirical_bayes_snapshots")


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "empirical_bayes_snapshots" not in inspector.get_table_names():
        op.create_table(
            "empirical_bayes_snapshots",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("snapshot_type", sa.String(length=64), nullable=False),
            sa.Column("group_id", sa.String(length=255), nullable=True),
            sa.Column("decision_type", sa.String(length=255), nullable=False),
            sa.Column("agent_decision_index", sa.Integer(), nullable=False),
            sa.Column("sample_size", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("feature_dim", sa.Integer(), nullable=False),
            sa.Column("theta", sa.JSON(), nullable=False),
            sa.Column("covariance", sa.JSON(), nullable=False),
            sa.Column("perturbation", sa.JSON(), nullable=True),
            sa.Column("metadata_json", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )

    if "model_parameters" in inspector.get_table_names():
        existing_cols = {c["name"] for c in inspector.get_columns("model_parameters")}
        with op.batch_alter_table("model_parameters") as batch_op:
            batch_op.alter_column(
                "probability_of_action",
                existing_type=sa.Float(),
                nullable=False,
            )
            for name, _, _ in reversed(_NEW_COLUMNS):
                if name in existing_cols:
                    batch_op.drop_column(name)
