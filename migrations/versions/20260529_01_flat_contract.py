"""flat upload_data contract + server-side warm-up + update-derived rewards

Implements the API-Spec redesign:
- Adds the ``data_uploads`` table (append-only flat snapshots; §6.3).
- Adds ``actions.is_warmup`` / ``actions.warmup_reason`` (server-side warm-up; §3.2/§6.2).
- Adds ``study_data.derived_at`` (update-derived rows; §6.4).
- Drops ``groups.warmup`` (warm-up is no longer a per-dyad host flag).
- Drops ``model_update_requests.callback_url`` (no callback; §3.4).
- Renames ``update_reproducibility_snapshots.study_data_count`` ->
  ``data_uploads_count`` (the snapshot now copies data_uploads; §6.9).

Revision ID: 20260529_01
Revises: 20260528_02
Create Date: 2026-05-29
"""
from alembic import op
import sqlalchemy as sa


revision = "20260529_01"
down_revision = "20260528_02"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    # 1. data_uploads table.
    if "data_uploads" not in tables:
        op.create_table(
            "data_uploads",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("group_id", sa.String(length=255), nullable=False),
            sa.Column("data", sa.JSON(), nullable=False),
            sa.Column("request_timestamp", sa.DateTime(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )

    # 2. actions warm-up columns.
    if "actions" in tables:
        cols = {c["name"] for c in inspector.get_columns("actions")}
        with op.batch_alter_table("actions") as batch_op:
            if "is_warmup" not in cols:
                batch_op.add_column(
                    sa.Column(
                        "is_warmup", sa.Boolean(), nullable=False, server_default=sa.false()
                    )
                )
            if "warmup_reason" not in cols:
                batch_op.add_column(
                    sa.Column("warmup_reason", sa.String(length=32), nullable=True)
                )

    # 3. study_data.derived_at.
    if "study_data" in tables:
        cols = {c["name"] for c in inspector.get_columns("study_data")}
        if "derived_at" not in cols:
            with op.batch_alter_table("study_data") as batch_op:
                batch_op.add_column(sa.Column("derived_at", sa.DateTime(), nullable=True))

    # 4. Drop groups.warmup.
    if "groups" in tables:
        cols = {c["name"] for c in inspector.get_columns("groups")}
        if "warmup" in cols:
            with op.batch_alter_table("groups") as batch_op:
                batch_op.drop_column("warmup")

    # 5. Drop model_update_requests.callback_url.
    if "model_update_requests" in tables:
        cols = {c["name"] for c in inspector.get_columns("model_update_requests")}
        if "callback_url" in cols:
            with op.batch_alter_table("model_update_requests") as batch_op:
                batch_op.drop_column("callback_url")

    # 6. Rename update_reproducibility_snapshots.study_data_count.
    if "update_reproducibility_snapshots" in tables:
        cols = {c["name"] for c in inspector.get_columns("update_reproducibility_snapshots")}
        if "study_data_count" in cols and "data_uploads_count" not in cols:
            with op.batch_alter_table("update_reproducibility_snapshots") as batch_op:
                batch_op.alter_column(
                    "study_data_count", new_column_name="data_uploads_count"
                )


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "update_reproducibility_snapshots" in tables:
        cols = {c["name"] for c in inspector.get_columns("update_reproducibility_snapshots")}
        if "data_uploads_count" in cols and "study_data_count" not in cols:
            with op.batch_alter_table("update_reproducibility_snapshots") as batch_op:
                batch_op.alter_column(
                    "data_uploads_count", new_column_name="study_data_count"
                )

    if "model_update_requests" in tables:
        cols = {c["name"] for c in inspector.get_columns("model_update_requests")}
        if "callback_url" not in cols:
            with op.batch_alter_table("model_update_requests") as batch_op:
                batch_op.add_column(
                    sa.Column(
                        "callback_url",
                        sa.String(length=1024),
                        nullable=False,
                        server_default="",
                    )
                )

    if "groups" in tables:
        cols = {c["name"] for c in inspector.get_columns("groups")}
        if "warmup" not in cols:
            with op.batch_alter_table("groups") as batch_op:
                batch_op.add_column(
                    sa.Column(
                        "warmup", sa.Boolean(), nullable=False, server_default=sa.false()
                    )
                )

    if "study_data" in tables:
        cols = {c["name"] for c in inspector.get_columns("study_data")}
        if "derived_at" in cols:
            with op.batch_alter_table("study_data") as batch_op:
                batch_op.drop_column("derived_at")

    if "actions" in tables:
        cols = {c["name"] for c in inspector.get_columns("actions")}
        with op.batch_alter_table("actions") as batch_op:
            if "warmup_reason" in cols:
                batch_op.drop_column("warmup_reason")
            if "is_warmup" in cols:
                batch_op.drop_column("is_warmup")

    if "data_uploads" in tables:
        op.drop_table("data_uploads")
