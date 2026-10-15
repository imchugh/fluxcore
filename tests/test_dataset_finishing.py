"""Tests for dataset_finishing.assign_global_attrs's global-attribute layering."""

import unittest

import xarray as xr

from fluxcore.orchestration import dataset_finishing
from fluxcore.orchestration.dataset_finishing import assign_global_attrs


def _dataset():
    return xr.Dataset(attrs={"site_name": "TestSite"})


class AssignGlobalAttrsTestCase(unittest.TestCase):
    def test_core_attrs_applied_without_global_attrs(self):
        out = assign_global_attrs(_dataset())

        for key, value in dataset_finishing.NC_GLOBAL_ATTRS.items():
            self.assertEqual(out.attrs[key], value)
        self.assertEqual(out.attrs["Conventions"], "CF-1.8")
        self.assertEqual(out.attrs["processing_level"], "L1")
        self.assertIn("date_created", out.attrs)

    def test_global_attrs_add_to_and_override_core_attrs(self):
        out = assign_global_attrs(
            _dataset(),
            global_attrs={
                "license_name": "Other licence",
                "metadata_link": "http://example.org/<site>/",
            },
        )

        self.assertEqual(out.attrs["license_name"], "Other licence")
        self.assertEqual(out.attrs["metadata_link"], "http://example.org/TestSite/")
        self.assertEqual(out.attrs["featureType"], "timeSeries")

    def test_processing_level_overrides_both_sources(self):
        out = assign_global_attrs(
            _dataset(),
            global_attrs={"processing_level": "L1"},
            processing_level="L2",
        )

        self.assertEqual(out.attrs["processing_level"], "L2")


if __name__ == "__main__":
    unittest.main()
