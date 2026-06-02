import unittest

from competition_config import (
    TARGET_LEAGUE_IDS,
    get_competition_flags,
)


class CompetitionConfigTest(unittest.TestCase):
    def test_default_includes_requested_international_competitions(self):
        self.assertIn(27, TARGET_LEAGUE_IDS)
        self.assertIn(31, TARGET_LEAGUE_IDS)

    def test_world_cup_flags(self):
        flags = get_competition_flags(27, "World Cup 2026")
        self.assertTrue(flags["is_international"])
        self.assertTrue(flags["is_world_cup"])
        self.assertFalse(flags["is_friendly"])
        self.assertEqual(flags["competition_type"], "world_cup")

    def test_friendly_flags(self):
        flags = get_competition_flags(31, "International Friendly Games")
        self.assertTrue(flags["is_international"])
        self.assertTrue(flags["is_friendly"])
        self.assertFalse(flags["is_world_cup"])
        self.assertEqual(flags["competition_type"], "friendly")


if __name__ == "__main__":
    unittest.main()
