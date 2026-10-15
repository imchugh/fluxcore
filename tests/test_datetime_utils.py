"""Tests for fluxcore.infrastructure.datetime_utils."""

import unittest
from datetime import timedelta

from fluxcore.infrastructure.datetime_utils import get_standard_timezone


class TestGetStandardTimezone(unittest.TestCase):
    def test_southern_hemisphere_dst_zone(self):
        tz = get_standard_timezone("Australia/Sydney")
        self.assertEqual(tz.utcoffset(None), timedelta(hours=10))

    def test_northern_hemisphere_dst_zone(self):
        tz = get_standard_timezone("Europe/Berlin")
        self.assertEqual(tz.utcoffset(None), timedelta(hours=1))

    def test_zone_without_dst(self):
        tz = get_standard_timezone("Australia/Darwin")
        self.assertEqual(tz.utcoffset(None), timedelta(hours=9, minutes=30))


if __name__ == "__main__":
    unittest.main(verbosity=2)
