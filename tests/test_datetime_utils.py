"""Tests for infrastructure.datetime_utils.floor_to_interval."""

import unittest
from datetime import datetime

from infrastructure.datetime_utils import floor_to_interval


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
