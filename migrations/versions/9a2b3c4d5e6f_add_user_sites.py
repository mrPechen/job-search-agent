"""add user_sites

Revision ID: 9a2b3c4d5e6f
Revises: ea5108fb8c54
Create Date: 2026-08-25 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "9a2b3c4d5e6f"
down_revision: Union[str, Sequence[str], None] = "ea5108fb8c54"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_sites",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("domain", sa.String(length=255), nullable=False),
        sa.Column(
            "added_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "domain", name="uq_user_sites_user_domain"),
    )
    op.create_index(
        op.f("ix_user_sites_user_id"), "user_sites", ["user_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_user_sites_user_id"), table_name="user_sites")
    op.drop_table("user_sites")
