"""Strengthen cross-table integrity and normalize SIS role keys.

The updates retain role ids and grants, so existing sessions and parent-facing services
continue to resolve the same people and permissions after the key normalization.
"""
from alembic import op
import sqlalchemy as sa


revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None


ROLE_KEYS = {
    "system_admin": "admin",
    "principal": "school_manager",
    "year_supervisor": "floor_supervisor",
}


def upgrade() -> None:
    # Update only legacy rows. The role id remains stable, preserving all grants and audit
    # references without using a numeric id as an authorization key.
    bind = op.get_bind()
    codes = set(bind.execute(sa.text("SELECT code FROM roles")).scalars())
    collisions = [f"{old} -> {new}" for old, new in ROLE_KEYS.items() if old in codes and new in codes]
    if collisions:
        raise RuntimeError(
            "Cannot normalize duplicated role keys without an administrator decision: "
            + ", ".join(collisions)
        )
    for old, new in ROLE_KEYS.items():
        op.execute(sa.text("UPDATE roles SET code=:new WHERE code=:old").bindparams(new=new, old=old))


def downgrade() -> None:
    for old, new in ROLE_KEYS.items():
        op.execute(sa.text("UPDATE roles SET code=:old WHERE code=:new").bindparams(new=new, old=old))
