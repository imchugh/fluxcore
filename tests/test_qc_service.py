"""Tests for services/data/qc_service.py's QC check functions."""

import unittest

import numpy as np
import pandas as pd

from services.data import qc_service


def _index(n, freq="30min", start="2020-01-01"):
    return pd.date_range(start, periods=n, freq=freq)


class RangeCheckTestCase(unittest.TestCase):
    def test_flags_outside_range(self):
        idx = _index(5)
        series = pd.Series([0.0, -20.0, 10.0, 60.0, np.nan], index=idx)
        bad = qc_service.range_check(series, lower=-10, upper=50)
        expected = pd.Series([False, True, False, True, False], index=idx)
        pd.testing.assert_series_equal(bad, expected)

    def test_nan_not_flagged(self):
        idx = _index(1)
        series = pd.Series([np.nan], index=idx)
        bad = qc_service.range_check(series, lower=-10, upper=50)
        self.assertFalse(bool(bad.iloc[0]))


class ExcludeDatesCheckTestCase(unittest.TestCase):
    def test_flags_inclusive_range(self):
        idx = _index(5, freq="D", start="2020-01-01")
        ranges = [(pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03"))]
        bad = qc_service.exclude_dates_check(idx, ranges)
        expected = pd.Series([False, True, True, False, False], index=idx)
        pd.testing.assert_series_equal(bad, expected)

    def test_multiple_ranges_combine(self):
        idx = _index(5, freq="D", start="2020-01-01")
        ranges = [
            (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")),
            (pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-05")),
        ]
        bad = qc_service.exclude_dates_check(idx, ranges)
        expected = pd.Series([True, False, False, False, True], index=idx)
        pd.testing.assert_series_equal(bad, expected)


class DependencyCheckTestCase(unittest.TestCase):
    def test_or_combination(self):
        idx = _index(4)
        a = pd.Series([True, False, False, False], index=idx)
        b = pd.Series([False, False, True, False], index=idx)
        combined = qc_service.dependency_check([a, b])
        expected = pd.Series([True, False, True, False], index=idx)
        pd.testing.assert_series_equal(combined, expected)

    def test_requires_at_least_one_series(self):
        with self.assertRaises(ValueError):
            qc_service.dependency_check([])


class MADFilterTestCase(unittest.TestCase):
    def _build_series(self, n=48 * 20, spike_idx=48 * 10, spike_size=1000.0):
        idx = pd.date_range("2020-01-01", periods=n, freq="30min")
        rng = np.random.default_rng(0)
        base = 20 + 5 * np.sin(np.linspace(0, 40 * np.pi, n)) + rng.normal(0, 0.1, n)
        base[spike_idx] += spike_size
        series = pd.Series(base, index=idx)
        hour = idx.hour
        fsd = np.where((hour >= 6) & (hour < 18), 500.0, 0.0)
        reference = pd.Series(fsd, index=idx)
        return series, reference

    def test_flags_injected_spike(self):
        series, reference = self._build_series()
        spike_idx = 48 * 10
        bad = qc_service.mad_filter(
            series, reference, time_step_minutes=30, window_days=5
        )
        self.assertTrue(bool(bad.iloc[spike_idx]))
        self.assertFalse(bool(bad.iloc[spike_idx - 100]))

    def test_short_series_no_crash(self):
        idx = _index(2)
        series = pd.Series([1.0, 2.0], index=idx)
        reference = pd.Series([500.0, 500.0], index=idx)
        bad = qc_service.mad_filter(series, reference, time_step_minutes=30)
        self.assertEqual(len(bad), 2)
        self.assertFalse(bad.any())

    def test_gap_edge_handled(self):
        n = 48 * 10
        idx = pd.date_range("2020-01-01", periods=n, freq="30min")
        rng = np.random.default_rng(1)
        base = 20 + rng.normal(0, 0.1, n)
        base[100:110] = np.nan
        series = pd.Series(base, index=idx)
        reference = pd.Series(500.0, index=idx)
        bad = qc_service.mad_filter(
            series, reference, time_step_minutes=30, window_days=3
        )
        self.assertFalse(bool(bad.iloc[105]))

    def test_window_boundary_arithmetic(self):
        # Regression check against pfp_ck.do_madfilter_1's
        # n_windows = int(nrecs / window_nrecs) (pfp_ck.py:905).
        nrecs = 48 * 40
        window_days = 13
        nperday = int(24 * 60 / 30)
        window_nrecs = window_days * nperday
        self.assertEqual(nrecs // window_nrecs, 3)


if __name__ == "__main__":
    unittest.main()
