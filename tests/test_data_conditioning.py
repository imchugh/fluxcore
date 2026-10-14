"""Tests for fluxcore/infrastructure/data_conditioning.py."""

import unittest

import numpy as np
import pandas as pd

from fluxcore.infrastructure.data_conditioning import (
    condition_dataframe,
    infer_interval,
)


def _frame(times):
    idx = pd.DatetimeIndex(pd.to_datetime(times), name="time")
    return pd.DataFrame({"x": np.arange(len(idx), dtype=float)}, index=idx)


class ConditionDataframeTestCase(unittest.TestCase):
    def test_single_record_with_interval_out(self):
        # An incremental tail with one new record: must not try to infer.
        out = condition_dataframe(_frame(["2026-10-14 05:30"]), interval_out=30)

        self.assertEqual(len(out), 1)
        self.assertEqual(out.index[0], pd.Timestamp("2026-10-14 05:30"))

    def test_pads_missing_steps(self):
        out = condition_dataframe(
            _frame(["2026-10-14 00:00", "2026-10-14 01:30"]), interval_out=30
        )

        self.assertEqual(len(out), 4)
        self.assertTrue(out["x"].iloc[1:3].isna().all())

    def test_infers_interval_when_none_given(self):
        # Gaps of 10, 10 and 30 min: the median (10) wins over the gap.
        out = condition_dataframe(
            _frame(
                [
                    "2026-10-14 00:00",
                    "2026-10-14 00:10",
                    "2026-10-14 00:20",
                    "2026-10-14 00:50",
                ]
            )
        )

        self.assertEqual(out.index.freq, pd.Timedelta(minutes=10))
        self.assertEqual(len(out), 6)

    def test_single_record_without_any_interval_raises_clearly(self):
        with self.assertRaisesRegex(ValueError, "fewer than two"):
            condition_dataframe(_frame(["2026-10-14 05:30"]))


class InferIntervalTestCase(unittest.TestCase):
    def test_median_ignores_occasional_gap(self):
        df = _frame(
            [
                "2026-10-14 00:00",
                "2026-10-14 00:30",
                "2026-10-14 01:00",
                "2026-10-14 03:00",
            ]
        )
        self.assertEqual(infer_interval(df), 30)


if __name__ == "__main__":
    unittest.main()
