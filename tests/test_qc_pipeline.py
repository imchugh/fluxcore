"""Tests for orchestration/qc_pipeline.py."""

import unittest

import numpy as np
import pandas as pd
import xarray as xr

from orchestration import qc_pipeline
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
        self.assertEqual(flags[3], qc_pipeline.QC_FLAG_BITS["range_check"])
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
        self.assertEqual(flags[3], qc_pipeline.QC_FLAG_BITS["dependency_check"])

    def test_missing_bit_set_for_nan_input(self):
        ds = _build_dataset()
        ds["Ta_Av"][5, 0, 0] = np.nan
        qc_config = SiteQCConfig(
            site_name="TestSite",
            variables={"Ta_Av": VariableQCSpec()},
        )
        out = qc_pipeline.apply_qc(ds, qc_config)
        flags = out["Ta_Av_QCFlag"].squeeze(("latitude", "longitude")).values
        self.assertEqual(flags[5], qc_pipeline.QC_FLAG_BITS["missing"])


class ApplyQCRangeDefaultsTestCase(unittest.TestCase):
    """Tests for apply_qc's default-fallback path (range_defaults arg)."""

    def test_default_applies_to_unconfigured_variable(self):
        ds = _build_dataset()
        ds["Fco2"][4, 0, 0] = 999.0  # out of range under the default, not the fixture's value
        qc_config = SiteQCConfig(site_name="TestSite", variables={})

        out = qc_pipeline.apply_qc(ds, qc_config, range_defaults={"Fco2": [-50, 50]})

        fco2 = out["Fco2"].squeeze(("latitude", "longitude")).values
        self.assertTrue(np.isnan(fco2[4]))
        self.assertFalse(np.isnan(fco2[0]))

        flags = out["Fco2_QCFlag"].squeeze(("latitude", "longitude")).values
        self.assertEqual(flags[4], qc_pipeline.QC_FLAG_BITS["range_check"])
        self.assertEqual(flags[0], 0)

    def test_explicit_config_takes_precedence_over_default(self):
        ds = _build_dataset()  # Ta_Av[3] = 200.0
        qc_config = SiteQCConfig(
            site_name="TestSite",
            variables={
                "Ta_Av": VariableQCSpec(range_check=RangeCheckSpec(lower=-10, upper=50)),
            },
        )
        # A much looser default that would NOT flag index 3 on its own —
        # if this were applied instead of/on top of the explicit config,
        # index 3 would not end up masked.
        out = qc_pipeline.apply_qc(
            ds, qc_config, range_defaults={"Ta": {"Av": [-1000, 1000]}}
        )
        ta = out["Ta_Av"].squeeze(("latitude", "longitude")).values
        self.assertTrue(np.isnan(ta[3]))

    def test_statistic_keyed_default_uses_attrs_statistic_type(self):
        ds = _build_dataset()
        ds["Ta_Av"].attrs["statistic_type"] = "average"
        ds["Ta_Av"][4, 0, 0] = 999.0
        qc_config = SiteQCConfig(site_name="TestSite", variables={})

        out = qc_pipeline.apply_qc(
            ds, qc_config, range_defaults={"Ta": {"Av": [-10, 50]}}
        )
        ta = out["Ta_Av"].squeeze(("latitude", "longitude")).values
        self.assertTrue(np.isnan(ta[4]))
        self.assertFalse(np.isnan(ta[0]))

    def test_qualifier_keyed_default_resolves_by_name(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="30min")
        values = np.array([1.0, 1.0, 1.0, 5000.0, 1.0]).reshape(5, 1, 1)
        ds = xr.Dataset(
            {"Diag_IRGA": (("time", "latitude", "longitude"), values), "crs": 0},
            coords={"time": idx, "latitude": [0.0], "longitude": [0.0]},
        )
        ds.attrs["time_step"] = 30
        qc_config = SiteQCConfig(site_name="TestSite", variables={})

        out = qc_pipeline.apply_qc(
            ds, qc_config, range_defaults={"Diag": {"IRGA": [0, 1000]}}
        )
        values_out = out["Diag_IRGA"].squeeze(("latitude", "longitude")).values
        self.assertTrue(np.isnan(values_out[3]))
        self.assertFalse(np.isnan(values_out[0]))

    def test_no_matching_default_passes_through(self):
        ds = _build_dataset()
        qc_config = SiteQCConfig(site_name="TestSite", variables={})
        out = qc_pipeline.apply_qc(ds, qc_config, range_defaults={"SomeOtherQuantity": [0, 1]})
        xr.testing.assert_identical(out["Fco2"], ds["Fco2"])

    def test_no_range_defaults_arg_behaves_as_before(self):
        ds = _build_dataset()
        qc_config = SiteQCConfig(site_name="TestSite", variables={})
        out = qc_pipeline.apply_qc(ds, qc_config)
        xr.testing.assert_identical(out["Fco2"], ds["Fco2"])


if __name__ == "__main__":
    unittest.main()
