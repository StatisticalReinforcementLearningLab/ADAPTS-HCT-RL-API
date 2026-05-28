"""add groups.warmup column and standardization_baselines table

Adds:
- ``groups.warmup`` boolean column (default False) — flagged on the first
  five enrolled dyads so the algorithm bypasses the learner and returns a
  Bernoulli(0.5) action while we accrue the data used to seed the EB prior.
- ``standardization_baselines`` table — one row per
  ``(group_id, decision_type, variable_name)`` storing the per-dyad week-1
  mean and standard deviation used to standardize continuous state inputs.

Revision ID: 20260421_01
Revises:
Create Date: 2026-04-21
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260421_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # 1. groups.warmup
    if "groups" in inspector.get_table_names():
        existing_cols = {c["name"] for c in inspector.get_columns("groups")}
        if "warmup" not in existing_cols:
            op.add_column(
                "groups",
                sa.Column(
                    "warmup",
                    sa.Boolean(),
                    nullable=False,
                    server_default=sa.false(),
                ),
            )
            # Drop the server_default once the column is populated so future
            # inserts must specify it explicitly.
            with op.batch_alter_table("groups") as batch_op:
                batch_op.alter_column("warmup", server_default=None)

    # 2. standardization_baselines
    if "standardization_baselines" not in inspector.get_table_names():
        op.create_table(
            "standardization_baselines",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("group_id", sa.String(length=255), nullable=False),
            sa.Column("decision_type", sa.String(length=255), nullable=False),
            sa.Column("variable_name", sa.String(length=255), nullable=False),
            sa.Column("mu", sa.Float(), nullable=False),
            sa.Column("sigma", sa.Float(), nullable=False),
            sa.Column(
                "sample_size", sa.Integer(), nullable=False, server_default="0"
            ),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint(
                "group_id",
                "decision_type",
                "variable_name",
                name="uq_baseline_group_dt_var",
            ),
        )


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "standardization_baselines" in inspector.get_table_names():
        op.drop_table("standardization_baselines")

    if "groups" in inspector.get_table_names():
        existing_cols = {c["name"] for c in inspector.get_columns("groups")}
        if "warmup" in existing_cols:
            with op.batch_alter_table("groups") as batch_op:
                batch_op.drop_column("warmup")
