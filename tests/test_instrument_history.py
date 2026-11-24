"""Tests for fluxcore/services/metadata/instrument_history.py."""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from fluxcore.services.metadata.instrument_history import (
    InstrumentHistory,
    InstrumentPeriod,
)

CHANGE = datetime(2025, 6, 1, 0, 0)
AFTER_CHANGE = datetime(2025, 6, 1, 0, 30)
YEAR_2025 = (datetime(2025, 1, 1, 0, 30), datetime(2026, 1, 1, 0, 0))


def _spec(instrument, begin=None, end=None):
    return SimpleNamespace(
        instrument=instrument, begin=begin, end=end, canonical_name="Var"
    )


def _changeover(first="Vaisala HMP155", second="Campbell HMP"):
    return InstrumentHistory.from_specs(
        [_spec(first, end=CHANGE), _spec(second, begin=AFTER_CHANGE)]
    )


def _compound_irga_change():
    return InstrumentHistory.from_specs(
        [
            _spec({"sonic_anemometer": "CSAT3B", "irga": "LI-7500"}, end=CHANGE),
            _spec(
                {"sonic_anemometer": "CSAT3B", "irga": "LI-7500RS"},
                begin=AFTER_CHANGE,
            ),
        ]
    )


class FromSpecsTestCase(unittest.TestCase):
    def test_unchanged_instrument_is_one_open_period(self):
        history = InstrumentHistory.from_specs([_spec("Kipp&Zonen CNR4")])
        self.assertEqual(len(history.roles), 1)
        self.assertIsNone(history.roles[0].role)
        self.assertEqual(
            history.roles[0].periods, (InstrumentPeriod("Kipp&Zonen CNR4", None, None),)
        )
        self.assertEqual(history.current(), "Kipp&Zonen CNR4")

    def test_consecutive_inputs_from_one_instrument_are_one_period(self):
        """A units change on one sensor (A, A, then B) gives A one period."""
        history = InstrumentHistory.from_specs(
            [
                _spec("A", end=datetime(2020, 1, 1)),
                _spec("A", begin=datetime(2020, 1, 1, 0, 30), end=CHANGE),
                _spec("B", begin=AFTER_CHANGE),
            ]
        )
        self.assertEqual(
            history.roles[0].periods,
            (
                InstrumentPeriod("A", None, CHANGE),
                InstrumentPeriod("B", AFTER_CHANGE, None),
            ),
        )

    def test_compound_roles_keep_config_order(self):
        history = _compound_irga_change()
        self.assertEqual([r.role for r in history.roles], ["sonic_anemometer", "irga"])
        self.assertEqual([r.changed for r in history.roles], [False, True])
        self.assertEqual(history.current(), "CSAT3B,LI-7500RS")

    def test_mixed_single_and_compound_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "mix single and compound"):
            InstrumentHistory.from_specs(
                [_spec("A", end=CHANGE), _spec({"irga": "B"}, begin=AFTER_CHANGE)]
            )


class AttrFormTestCase(unittest.TestCase):
    def test_round_trip(self):
        for history in (_changeover(), _compound_irga_change()):
            self.assertEqual(InstrumentHistory.from_attr(history.to_attr()), history)

    def test_order_survives_a_zarr_round_trip(self):
        """Zarr alphabetises dict keys, so a dict keyed by instrument would
        come back Campbell-first; the list form keeps time order."""
        history = _changeover(first="Vaisala HMP155", second="Campbell HMP")
        idx = pd.date_range("2025-01-01", periods=2, freq="30min", name="time")
        ds = xr.Dataset({"Ta": ("time", np.zeros(2))}, coords={"time": idx})
        ds.Ta.attrs["instrument_history"] = history.to_attr()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "store.zarr"
            ds.to_zarr(path)
            back = xr.open_zarr(path).Ta.attrs["instrument_history"]
        self.assertEqual(InstrumentHistory.from_attr(back), history)

    def test_older_attr_forms_are_rejected(self):
        older = [
            {"Sensor A": {"start_date": None, "end_date": None}},
            [["irga", {"Sensor A": {"start_date": None, "end_date": None}}]],
            None,
        ]
        for attr in older:
            with self.assertRaisesRegex(ValueError, "Rebuild the store"):
                InstrumentHistory.from_attr(attr)


class NetcdfAttrsTestCase(unittest.TestCase):
    def test_unchanged_instrument_has_no_history(self):
        history = InstrumentHistory.from_specs([_spec("Kipp&Zonen CNR4")])
        self.assertEqual(history.netcdf_attrs(*YEAR_2025), ("Kipp&Zonen CNR4", None))

    def test_changeover_year_lists_both_periods_in_time_order(self):
        instrument, history = _changeover().netcdf_attrs(*YEAR_2025)
        self.assertEqual(instrument, "Campbell HMP")
        self.assertEqual(
            history,
            "(Vaisala HMP155,2025-01-01T00:30:00,2025-06-01T00:00:00)"
            "|(Campbell HMP,2025-06-01T00:30:00,2026-01-01T00:00:00)",
        )

    def test_year_with_one_period_has_no_history(self):
        window = (datetime(2024, 1, 1, 0, 30), datetime(2025, 1, 1))
        self.assertEqual(_changeover().netcdf_attrs(*window), ("Vaisala HMP155", None))

    def test_compound_history_lists_only_changed_roles(self):
        instrument, history = _compound_irga_change().netcdf_attrs(*YEAR_2025)
        self.assertEqual(instrument, "CSAT3B,LI-7500RS")
        self.assertEqual(
            history,
            "irga>(LI-7500,2025-01-01T00:30:00,2025-06-01T00:00:00)"
            "|(LI-7500RS,2025-06-01T00:30:00,2026-01-01T00:00:00)",
        )

    def test_compound_year_before_a_change_keeps_its_segment(self):
        window = (datetime(2024, 1, 1, 0, 30), datetime(2025, 1, 1))
        instrument, history = _compound_irga_change().netcdf_attrs(*window)
        self.assertEqual(instrument, "CSAT3B,LI-7500")
        self.assertEqual(
            history, "irga>(LI-7500,2024-01-01T00:30:00,2025-01-01T00:00:00)"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
