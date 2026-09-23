"""Tests for orchestration/qc_pipeline.py."""

import unittest

import numpy as np
import pandas as pd
import xarray as xr

from orchestration import qc_pipeline
from services.metadata import qc_config_schema
from services.metadata.qc_config_schema import (
    RangeCheckSpec,
    SiteQCConfig,
    VariableQCSpec,
)


def _build_dataset(n=10, time_step=30):
    idx = pd.date_range("2020-01-01", periods=n, freq=f"{time_step}min")
    lat, lon = [0.0], [0.0]

    def _var(values):
        return (
            ("time", "latitude", "longitude"),
            np.array(values, dtype=float).reshape(n, 1, 1),
        )

    ta = [10.0] * n
    ta[3] = 200.0  # out of range
    fco2 = [1.0] * n

    ds = xr.Dataset(
        {
            "Ta_Av": _var(ta),
            "Ta_Av_QCFlag": (
                ("time", "latitude", "longitude"),
                np.zeros((n, 1, 1), dtype=int),
            ),
            "Fco2": _var(fco2),
            "crs": 0,
        },
        coords={"time": idx, "latitude": lat, "longitude": lon},
    )
    ds.attrs["time_step"] = time_step
    return ds


class ApplyQCTestCase(unittest.TestCase):
    def test_range_check_masks_and_flags(self):
        ds = _build_dataset()
        qc_config = SiteQCConfig(
            site_name="TestSite",
            variables={
                "Ta_Av": VariableQCSpec(range_check=RangeCheckSpec(lower=-10, upper=50)),
            },
        )
        out = qc_pipeline.apply_qc(ds, qc_config)

        ta_values = out["Ta_Av"].squeeze(("latitude", "longitude")).values
        self.assertTrue(np.isnan(ta_values[3]))
        self.assertFalse(np.isnan(ta_values[0]))

        flags = out["Ta_Av_QCFlag"].squeeze(("latitude", "longitude")).values
        self.assertEqual(flags[3], qc_pipeline.QC_FLAG_CODES["range_check"])
        self.assertEqual(flags[0], 0)

    def test_unconfigured_variable_passes_through(self):
        ds = _build_dataset()
        qc_config = SiteQCConfig(
            site_name="TestSite",
            variables={
                "Ta_Av": VariableQCSpec(range_check=RangeCheckSpec(lower=-10, upper=50)),
            },
        )
        out = qc_pipeline.apply_qc(ds, qc_config)
        xr.testing.assert_identical(out["Fco2"], ds["Fco2"])

    def test_chained_dependency_propagates(self):
        ds = _build_dataset()
        qc_config = SiteQCConfig(
            site_name="TestSite",
            variables={
                "Ta_Av": VariableQCSpec(range_check=RangeCheckSpec(lower=-10, upper=50)),
                "Fco2": VariableQCSpec(dependency_check=["Ta_Av"]),
            },
        )
        out = qc_pipeline.apply_qc(ds, qc_config)

        fco2_values = out["Fco2"].squeeze(("latitude", "longitude")).values
        self.assertTrue(np.isnan(fco2_values[3]))

        flags = out["Fco2_QCFlag"].squeeze(("latitude", "longitude")).values
        self.assertEqual(flags[3], qc_pipeline.QC_FLAG_CODES["dependency_check"])

    def test_missing_bit_set_for_nan_input(self):
        ds = _build_dataset()
        ds["Ta_Av"][5, 0, 0] = np.nan
        qc_config = SiteQCConfig(
            site_name="TestSite",
            variables={"Ta_Av": VariableQCSpec()},
        )
        out = qc_pipeline.apply_qc(ds, qc_config)
        flags = out["Ta_Av_QCFlag"].squeeze(("latitude", "longitude")).values
        self.assertEqual(flags[5], qc_pipeline.QC_FLAG_CODES["missing"])

    def test_later_check_wins_when_multiple_fail(self):
        ds = _build_dataset()  # Ta_Av[3] = 200.0, out of range
        qc_config = SiteQCConfig(
            site_name="TestSite",
            variables={
                "Ta_Av": VariableQCSpec(
                    range_check=RangeCheckSpec(lower=-10, upper=50),
                    exclude_dates=[
                        ("2020-01-01T01:30:00", "2020-01-01T01:30:00"),
                    ],
                ),
            },
        )
        out = qc_pipeline.apply_qc(ds, qc_config)
        flags = out["Ta_Av_QCFlag"].squeeze(("latitude", "longitude")).values
        # Index 3 fails both range_check and exclude_dates. PyFluxPro's own
        # execution order runs exclude_dates after range_check and
        # unconditionally overwrites -- so its code wins, not a sum of both.
        self.assertEqual(flags[3], qc_pipeline.QC_FLAG_CODES["exclude_dates"])

    def test_dependency_check_can_overwrite_missing(self):
        ds = _build_dataset()
        ds["Fco2"][3, 0, 0] = np.nan  # Fco2 itself missing at index 3
        qc_config = SiteQCConfig(
            site_name="TestSite",
            variables={
                "Ta_Av": VariableQCSpec(range_check=RangeCheckSpec(lower=-10, upper=50)),
                "Fco2": VariableQCSpec(dependency_check=["Ta_Av"]),
            },
        )
        out = qc_pipeline.apply_qc(ds, qc_config)
        flags = out["Fco2_QCFlag"].squeeze(("latitude", "longitude")).values
        # Fco2[3] is itself NaN (would be "missing"=1) AND its precursor
        # Ta_Av[3] fails range_check -- dependency_check only tests the
        # precursor's flag, not Fco2's own value, and (matching PyFluxPro's
        # general unconditional-overwrite pattern) overwrites regardless.
        self.assertEqual(flags[3], qc_pipeline.QC_FLAG_CODES["dependency_check"])


class ApplyQCWithResolvedDefaultsIntegrationTestCase(unittest.TestCase):
    """apply_qc no longer knows about defaults at all -- range_defaults and
    _dependency_defaults.yml resolution now happen upstream, in
    qc_config_schema.resolve_qc_config, before apply_qc ever runs (see
    tests/test_qc_config_schema.py for resolve_qc_config's own tests).
    This is the one integration behavior worth covering here: a merged-in
    default range_check on a dependency_check *source* has to correctly
    propagate through dependency_graph_order()/resolved_bad to whatever
    depends on it, exactly as if it had been explicitly configured -- easy
    to get subtly wrong even though no per-check logic in apply_qc itself
    changed.
    """

    def test_default_range_check_on_dependency_source_propagates(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="30min")
        diag = np.array([0.0, 0.0, 0.0, 5000.0, 0.0]).reshape(5, 1, 1)
        fco2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0]).reshape(5, 1, 1)
        ds = xr.Dataset(
            {
                "Diag_SONIC": (("time", "latitude", "longitude"), diag),
                "Fco2": (("time", "latitude", "longitude"), fco2),
                "crs": 0,
            },
            coords={"time": idx, "latitude": [0.0], "longitude": [0.0]},
        )
        ds.attrs["time_step"] = 30

        site_config = SiteQCConfig(site_name="TestSite", variables={})
        dependency_defaults = {
            "Fco2": VariableQCSpec(dependency_check=["Diag_SONIC"]),
        }
        range_defaults = {"Diag": {"SONIC": [0, 1000]}}

        merged = qc_config_schema.resolve_qc_config(
            site_config, dependency_defaults, range_defaults, ds
        )
        out = qc_pipeline.apply_qc(ds, merged)

        fco2_flags = out["Fco2_QCFlag"].squeeze(("latitude", "longitude")).values
        self.assertEqual(fco2_flags[3], qc_pipeline.QC_FLAG_CODES["dependency_check"])
        self.assertEqual(fco2_flags[0], 0)


if __name__ == "__main__":
    unittest.main()
