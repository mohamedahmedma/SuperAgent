"""The shared kernel: rules both contexts need, and which belong to neither.

A leaf by contract - it may not import `domain.agent` or `domain.corpus`. Anything that
would need to is not shared, it belongs to one of them.

The first residents are likely `school_week.py` and the language rule that
`agent/chat/language.py` and `indexing/language_check.py` are currently two copies of.
"""
