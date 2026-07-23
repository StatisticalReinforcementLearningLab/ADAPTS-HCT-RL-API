"""shared rid response field + /action model_param

Implements the API-Spec §2 preamble change that promotes ``rid`` to a common,
always-present response field on every endpoint, plus the flat model-parameter
replay vector on /action:

- Adds ``groups.rid`` (per-call rid of the most recent /register_group; §5.1).
- Adds ``data_uploads.rid`` (unique per-call rid; §5.3).
- Adds ``actions.model_param`` (JSON flat vector the learner scored to produce
  action_prob, so the idempotent /action replay can return it; §2.2/§5.2).
- Renames ``model_update_requests.update_id`` -> ``rid`` (§5.6).
- Renames ``update_reproducibility_snapshots.update_id`` -> ``rid`` (§5.9).

Revision ID: 20260723_01
Revises: 20260529_01
Create Date: 2026-07-23
"""
from alembic import op
import sqlalchemy as sa


revision = "20260723_01"
down_revision = "20260529_01"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    # 1. groups.rid
    if "groups" in tables:
        cols = {c["name"] for c in inspector.get_columns("groups")}
        if "rid" not in cols:
            with op.batch_alter_table("groups") as batch_op:
                batch_op.add_column(sa.Column("rid", sa.String(length=255), nullable=True))

    # 2. data_uploads.rid (unique among non-null values)
    if "data_uploads" in tables:
        cols = {c["name"] for c in inspector.get_columns("data_uploads")}
        if "rid" not in cols:
            with op.batch_alter_table("data_uploads") as batch_op:
                batch_op.add_column(sa.Column("rid", sa.String(length=255), nullable=True))
                batch_op.create_unique_constraint("uq_data_uploads_rid", ["rid"])

    # 3. actions.model_param
    if "actions" in tables:
        cols = {c["name"] for c in inspector.get_columns("actions")}
        if "model_param" not in cols:
            with op.batch_alter_table("actions") as batch_op:
                batch_op.add_column(sa.Column("model_param", sa.JSON(), nullable=True))

    # 4. model_update_requests.update_id -> rid
    if "model_update_requests" in tables:
        cols = {c["name"] for c in inspector.get_columns("model_update_requests")}
        if "update_id" in cols and "rid" not in cols:
            with op.batch_alter_table("model_update_requests") as batch_op:
                batch_op.alter_column("update_id", new_column_name="rid")

    # 5. update_reproducibility_snapshots.update_id -> rid
    if "update_reproducibility_snapshots" in tables:
        cols = {c["name"] for c in inspector.get_columns("update_reproducibility_snapshots")}
        if "update_id" in cols and "rid" not in cols:
            with op.batch_alter_table("update_reproducibility_snapshots") as batch_op:
                batch_op.alter_column("update_id", new_column_name="rid")


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "update_reproducibility_snapshots" in tables:
        cols = {c["name"] for c in inspector.get_columns("update_reproducibility_snapshots")}
        if "rid" in cols and "update_id" not in cols:
            with op.batch_alter_table("update_reproducibility_snapshots") as batch_op:
                batch_op.alter_column("rid", new_column_name="update_id")

    if "model_update_requests" in tables:
        cols = {c["name"] for c in inspector.get_columns("model_update_requests")}
        if "rid" in cols and "update_id" not in cols:
            with op.batch_alter_table("model_update_requests") as batch_op:
                batch_op.alter_column("rid", new_column_name="update_id")

    if "actions" in tables:
        cols = {c["name"] for c in inspector.get_columns("actions")}
        if "model_param" in cols:
            with op.batch_alter_table("actions") as batch_op:
                batch_op.drop_column("model_param")

    if "data_uploads" in tables:
        cols = {c["name"] for c in inspector.get_columns("data_uploads")}
        if "rid" in cols:
            with op.batch_alter_table("data_uploads") as batch_op:
                batch_op.drop_constraint("uq_data_uploads_rid", type_="unique")
                batch_op.drop_column("rid")

    if "groups" in tables:
        cols = {c["name"] for c in inspector.get_columns("groups")}
        if "rid" in cols:
            with op.batch_alter_table("groups") as batch_op:
                batch_op.drop_column("rid")
