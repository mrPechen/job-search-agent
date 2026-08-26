"""align embedding dim to 768

Revision ID: 2aa35da2c3ec
Revises: 9a2b3c4d5e6f
Create Date: 2026-08-26 18:52:35.128812

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '2aa35da2c3ec'
down_revision: Union[str, Sequence[str], None] = '9a2b3c4d5e6f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        "ALTER TABLE documents ALTER COLUMN embedding "
        "TYPE vector(768) USING embedding::vector(768)"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(
        "ALTER TABLE documents ALTER COLUMN embedding "
        "TYPE vector(1536) USING embedding::vector(1536)"
    )
