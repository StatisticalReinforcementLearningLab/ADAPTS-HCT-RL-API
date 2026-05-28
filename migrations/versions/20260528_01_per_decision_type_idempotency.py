"""make decision_idx idempotency per (group_id, decision_type)

Adds composite unique constraints so that the idempotency key on both
``actions`` and ``study_data`` is ``(group_id, decision_type, decision_idx)``
rather than ``(group_id, decision_idx)``. The three agents (aya_message,
cp_message, dyad_game) have independent per-dyad decision counters.

Revision ID: 20260528_01
Revises: 20260421_01
Create Date: 2026-05-28
"""
from alembic import op
import sqlalchemy as sa


revision = "20260528_01"
down_revision = "20260421_01"
branch_labels = None
depends_on = None


_ACTION_CONSTRAINT = "uq_action_group_type_idx"
_STUDY_CONSTRAINT = "uq_study_group_type_idx"


def _existing_unique_names(inspector, table: str) -> set[str]:
    return {uc["name"] for uc in inspector.get_unique_constraints(table)}


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "actions" in inspector.get_table_names():
        existing = _existing_unique_names(inspector, "actions")
        if _ACTION_CONSTRAINT not in existing:
            with op.batch_alter_table("actions") as batch_op:
                batch_op.create_unique_constraint(
                    _ACTION_CONSTRAINT,
                    ["group_id", "decision_type", "decision_idx"],
                )

    if "study_data" in inspector.get_table_names():
        existing = _existing_unique_names(inspector, "study_data")
        if _STUDY_CONSTRAINT not in existing:
            with op.batch_alter_table("study_data") as batch_op:
                batch_op.create_unique_constraint(
                    _STUDY_CONSTRAINT,
                    ["group_id", "decision_type", "decision_idx"],
                )


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "study_data" in inspector.get_table_names():
        existing = _existing_unique_names(inspector, "study_data")
        if _STUDY_CONSTRAINT in existing:
            with op.batch_alter_table("study_data") as batch_op:
                batch_op.drop_constraint(_STUDY_CONSTRAINT, type_="unique")

    if "actions" in inspector.get_table_names():
        existing = _existing_unique_names(inspector, "actions")
        if _ACTION_CONSTRAINT in existing:
            with op.batch_alter_table("actions") as batch_op:
                batch_op.drop_constraint(_ACTION_CONSTRAINT, type_="unique")
