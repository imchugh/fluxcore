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
    instrument_uri_map,
)

CHANGE = datetime(2025, 6, 1, 0, 0)
AFTER_CHANGE = datetime(2025, 6, 1, 0, 30)
YEAR_2025 = (datetime(2025, 1, 1, 0, 30), datetime(2026, 1, 1, 0, 0))


def _spec(instrument, begin=None, end=None, uri=None, calculate_from=None):
    return SimpleNamespace(
        instrument=instrument,
        instrument_uri=uri,
        begin=begin,
        end=end,
        canonical_name="Var",
        calculate_from=calculate_from,
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


class InstrumentUriTestCase(unittest.TestCase):
    YEAR_2024 = (datetime(2024, 1, 1, 0, 30), datetime(2025, 1, 1))

    def _logger_change(self):
        specs = [
            _spec("CR3000", end=CHANGE, uri="uri:cr3000"),
            _spec("CR1000X", begin=AFTER_CHANGE, uri="uri:cr1000x"),
        ]
        return InstrumentHistory.from_specs(specs), instrument_uri_map(specs)

    def _irga_change(self, sonic_uri="uri:csat"):
        specs = [
            _spec(
                {"sonic_anemometer": "CSAT3B", "irga": "LI-7500"},
                end=CHANGE,
                uri={"sonic_anemometer": sonic_uri, "irga": "uri:7500"},
            ),
            _spec(
                {"sonic_anemometer": "CSAT3B", "irga": "LI-7500RS"},
                begin=AFTER_CHANGE,
                uri={"sonic_anemometer": sonic_uri, "irga": "uri:7500rs"},
            ),
        ]
        return InstrumentHistory.from_specs(specs), instrument_uri_map(specs)

    def test_map_covers_every_instrument_in_the_history(self):
        self.assertEqual(
            self._logger_change()[1], {"CR3000": "uri:cr3000", "CR1000X": "uri:cr1000x"}
        )
        self.assertEqual(
            self._irga_change()[1],
            {"CSAT3B": "uri:csat", "LI-7500": "uri:7500", "LI-7500RS": "uri:7500rs"},
        )

    def test_each_year_gets_the_uri_of_its_own_instrument(self):
        history, uris = self._logger_change()
        self.assertEqual(history.netcdf_uri(uris, *self.YEAR_2024), "uri:cr3000")
        self.assertEqual(history.netcdf_uri(uris, *YEAR_2025), "uri:cr1000x")

    def test_compound_uris_line_up_with_the_instruments(self):
        history, uris = self._irga_change()
        self.assertEqual(history.netcdf_uri(uris, *self.YEAR_2024), "uri:csat,uri:7500")
        self.assertEqual(history.netcdf_uri(uris, *YEAR_2025), "uri:csat,uri:7500rs")

    def test_an_unresolved_uri_keeps_its_position(self):
        history, uris = self._irga_change(sonic_uri=None)
        self.assertEqual(history.netcdf_attrs(*YEAR_2025)[0], "CSAT3B,LI-7500RS")
        self.assertEqual(history.netcdf_uri(uris, *YEAR_2025), ",uri:7500rs")

    def test_no_resolved_uri_omits_the_attr(self):
        history = InstrumentHistory.from_specs([_spec("A")])
        self.assertIsNone(history.netcdf_uri({"A": None}, *YEAR_2025))
        self.assertIsNone(history.netcdf_uri(None, *YEAR_2025))

    def test_an_older_store_string_passes_through(self):
        history, _ = self._logger_change()
        self.assertEqual(
            history.netcdf_uri("uri:cr1000x", *self.YEAR_2024), "uri:cr1000x"
        )

    def test_map_survives_a_zarr_round_trip(self):
        _, uris = self._irga_change(sonic_uri=None)
        idx = pd.date_range("2025-01-01", periods=2, freq="30min", name="time")
        ds = xr.Dataset({"Fco2": ("time", np.zeros(2))}, coords={"time": idx})
        ds.Fco2.attrs["instrument_uri"] = uris
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "store.zarr"
            ds.to_zarr(path)
            back = xr.open_zarr(path).Fco2.attrs["instrument_uri"]
        self.assertEqual(back, uris)


if __name__ == "__main__":
    unittest.main(verbosity=2)
