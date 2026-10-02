"""The wire contract, as an integrator reads it.

One module, `contract.py`, because this service has one contract and splitting it by
router would put `StudentRef` in whichever file happened to need it first.
"""
from records.api.schemas.contract import *  # noqa: F401,F403
