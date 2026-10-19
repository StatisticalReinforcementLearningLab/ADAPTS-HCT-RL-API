"""Store rejected upload bodies byte-for-byte.

Revision ID: 20261019_01
Revises: 20261016_01
Create Date: 2026-10-19
"""

from alembic import op
import sqlalchemy as sa


revision = "20261019_01"
down_revision = "20261016_01"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "upload_quarantine" not in inspector.get_table_names():
        return
    columns = {column["name"]: column for column in inspector.get_columns("upload_quarantine")}
    raw = columns.get("raw_body")
    if raw is None or isinstance(raw["type"], sa.LargeBinary):
        return
    # Existing Text values contain the original UTF-8 request text. SQLite's
    # CAST(... AS BLOB) and the equivalent batch migration on other engines
    # preserve those encoded bytes while changing future writes to binary.
    dialect_options = {}
    if bind.dialect.name == "postgresql":
        dialect_options["postgresql_using"] = "convert_to(raw_body, 'UTF8')"
    with op.batch_alter_table("upload_quarantine") as batch_op:
        batch_op.alter_column(
            "raw_body",
            existing_type=raw["type"],
            type_=sa.LargeBinary(),
            existing_nullable=False,
            **dialect_options,
        )


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "upload_quarantine" not in inspector.get_table_names():
        return
    columns = {column["name"]: column for column in inspector.get_columns("upload_quarantine")}
    raw = columns.get("raw_body")
    if raw is None or isinstance(raw["type"], sa.Text):
        return
    dialect_options = {}
    if bind.dialect.name == "postgresql":
        dialect_options["postgresql_using"] = "convert_from(raw_body, 'UTF8')"
    with op.batch_alter_table("upload_quarantine") as batch_op:
        batch_op.alter_column(
            "raw_body",
            existing_type=raw["type"],
            type_=sa.Text(),
            existing_nullable=False,
            **dialect_options,
        )
