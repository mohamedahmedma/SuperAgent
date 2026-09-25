"""The cache behind RAG_FIX_PLAN item 34, on its own: bounded by size and by age."""
import unittest

from backend.infra.auth import KnownUsers


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class KnownUsersTests(unittest.TestCase):
    def test_an_added_user_is_known(self):
        known = KnownUsers()
        known.add("a")
        self.assertIn("a", known)
        self.assertNotIn("b", known)

    def test_an_entry_expires_after_its_ttl(self):
        clock = Clock()
        known = KnownUsers(ttl_seconds=60, clock=clock)
        known.add("a")
        clock.now += 59
        self.assertIn("a", known)
        clock.now += 2
        self.assertNotIn("a", known)
        self.assertEqual(0, len(known))

    def test_the_least_recently_seen_is_evicted_at_capacity(self):
        known = KnownUsers(capacity=2)
        known.add("a")
        known.add("b")
        self.assertIn("a", known)  # a is now the most recently seen
        known.add("c")
        self.assertIn("a", known)
        self.assertNotIn("b", known)
        self.assertIn("c", known)

    def test_clear_forgets_everything(self):
        known = KnownUsers()
        known.add("a")
        known.clear()
        self.assertNotIn("a", known)


if __name__ == "__main__":
    unittest.main()
