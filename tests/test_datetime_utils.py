"""Tests for fluxcore.infrastructure.datetime_utils."""

import unittest
from datetime import datetime, timedelta

from fluxcore.infrastructure.datetime_utils import (
    floor_to_interval,
    get_standard_timezone,
)


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


class TestFloorToInterval(unittest.TestCase):
    def test_already_aligned_timestamp_is_a_no_op(self):
        dt = datetime(2026, 8, 25, 12, 0, 0)
        self.assertEqual(floor_to_interval(dt, 30), dt)

    def test_mid_interval_timestamp_floors_down(self):
        dt = datetime(2026, 8, 25, 12, 6, 8, 744667)
        self.assertEqual(floor_to_interval(dt, 30), datetime(2026, 8, 25, 12, 0, 0))

    def test_floors_to_previous_interval_boundary_not_nearest(self):
        dt = datetime(2026, 8, 25, 12, 29, 59)
        self.assertEqual(floor_to_interval(dt, 30), datetime(2026, 8, 25, 12, 0, 0))

    def test_midnight_crossing(self):
        dt = datetime(2026, 8, 25, 0, 10, 0)
        self.assertEqual(floor_to_interval(dt, 30), datetime(2026, 8, 25, 0, 0, 0))

    def test_non_divisor_interval(self):
        dt = datetime(2026, 8, 25, 13, 40, 0)
        self.assertEqual(floor_to_interval(dt, 15), datetime(2026, 8, 25, 13, 30, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
