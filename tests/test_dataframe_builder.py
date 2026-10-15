"""Tests for fluxcore/orchestration/dataframe_builder.py's assembly steps."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from fluxcore.orchestration import dataframe_builder


def _registry(*raw_names, group="g"):
    return {
        f"{raw}_gp0": SimpleNamespace(
            raw_name=raw, alias=f"{raw}_gp0", file_group=group
        )
        for raw in raw_names
    }


def _file_group(files, expected, group="g"):
    return SimpleNamespace(
        group=group,
        variables_by_file={f: None for f in files},
        expected_variables=set(expected),
    )


def _frame(start, n, **cols):
    idx = pd.date_range(start, periods=n, freq="30min", name="time")
    return pd.DataFrame(
        {k: np.full(n, v, dtype=float) for k, v in cols.items()}, index=idx
    )


class BuildFileGroupDataframeTestCase(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(
            dataframe_builder, "_apply_conversions", side_effect=lambda df, **_: df
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _build(self, frames_by_file, expected, start_date=None):
        return dataframe_builder._build_file_group_dataframe(
            file_group=_file_group(frames_by_file, expected),
            loader=lambda f: frames_by_file[f],
            specs_by_alias=_registry(*expected),
            start_date=start_date,
            time_step=30,
        )

    def test_configured_variable_missing_from_recent_file_is_nan_filled(self):
        # Tpanel only exists in the old backup file; an incremental read from
        # a start date after it ends must still carry the column (all NaN).
        frames = {
            "backup.dat": _frame("2026-01-01", 4, Ta=1.0, Tpanel=20.0),
            "master.dat": _frame("2026-10-14 10:00", 3, Ta=2.0),
        }
        out = self._build(
            frames, ["Ta", "Tpanel"], start_date=pd.Timestamp("2026-10-14 10:00")
        )

        self.assertEqual(sorted(out.columns), ["Ta_gp0", "Tpanel_gp0"])
        self.assertEqual(len(out), 3)
        self.assertTrue(out["Tpanel_gp0"].isna().all())

    def test_full_read_keeps_historical_values(self):
        frames = {
            "backup.dat": _frame("2026-01-01", 2, Ta=1.0, Tpanel=20.0),
            "master.dat": _frame("2026-01-01 01:00", 2, Ta=2.0),
        }
        out = self._build(frames, ["Ta", "Tpanel"])

        self.assertEqual(out["Tpanel_gp0"].notna().sum(), 2)

    def test_group_with_no_rows_since_start_keeps_columns(self):
        frames = {"master.dat": _frame("2026-01-01", 2, Ta=1.0)}
        out = self._build(frames, ["Ta"], start_date=pd.Timestamp("2026-10-14"))

        self.assertEqual(list(out.columns), ["Ta_gp0"])
        self.assertEqual(len(out), 0)


class FluxExtentTruncationTestCase(unittest.TestCase):
    """Step 5: output is truncated to the flux file group's extent."""

    def _build(self, frames_by_group, start_date=None):
        groups = {name: SimpleNamespace(file_format="TOA5") for name in frames_by_group}
        patches = [
            mock.patch.object(dataframe_builder.raw_data_loader, "get_data_loader"),
            mock.patch.object(
                dataframe_builder,
                "_build_file_group_dataframe",
                side_effect=lambda file_group, **_: frames_by_group[
                    next(k for k, v in groups.items() if v is file_group)
                ],
            ),
            mock.patch.object(
                dataframe_builder,
                "_merge_overlapping_variables",
                side_effect=lambda df, **_: df,
            ),
            mock.patch.object(
                dataframe_builder,
                "_rename_to_canonical",
                side_effect=lambda df, **_: df,
            ),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        return dataframe_builder.build_dataframe(
            file_groups=groups,
            specs_by_alias={},
            start_date=start_date,
            flux_file="flux",
            time_step=30,
        )

    def test_full_build_trims_ancillary_to_the_flux_extent(self):
        out = self._build(
            {
                "flux": _frame("2026-01-02 00:00", 4, Fc=1.0),
                "met": _frame("2026-01-01 22:00", 12, Ta=2.0),
            }
        )
        self.assertEqual(out.index.min(), pd.Timestamp("2026-01-02 00:00"))
        self.assertEqual(out.index.max(), pd.Timestamp("2026-01-02 01:30"))

    def test_tail_keeps_rows_before_the_first_new_flux_record(self):
        # A flux gap right after the checkpoint: the tail must still start
        # at start_date, as a full build would, not at the first flux record.
        start = pd.Timestamp("2026-01-02 00:00")
        met = _frame("2026-01-02 00:00", 6, Ta=2.0)
        flux = _frame("2026-01-02 01:30", 2, Fc=1.0).reindex(met.index)
        out = self._build({"flux": flux.dropna(), "met": met}, start_date=start)
        self.assertEqual(out.index.min(), start)
        self.assertEqual(out.index.max(), pd.Timestamp("2026-01-02 02:00"))

    def test_tail_starts_at_start_date_even_with_no_earlier_rows(self):
        # Only the flux group has rows, and they start after a gap: the tail
        # is still padded back to start_date, as a full build's grid would be.
        start = pd.Timestamp("2026-01-02 00:00")
        out = self._build(
            {
                "flux": _frame("2026-01-02 01:30", 2, Fc=1.0),
                "met": _frame("2026-01-02 00:00", 0, Ta=2.0),
            },
            start_date=start,
        )
        self.assertEqual(out.index.min(), start)
        self.assertEqual(len(out), 5)
        self.assertEqual(out["Fc"].notna().sum(), 2)

    def test_tail_ends_at_the_last_flux_record(self):
        out = self._build(
            {
                "flux": _frame("2026-01-02 00:00", 2, Fc=1.0),
                "met": _frame("2026-01-02 00:00", 6, Ta=2.0),
            },
            start_date=pd.Timestamp("2026-01-02 00:00"),
        )
        self.assertEqual(out.index.max(), pd.Timestamp("2026-01-02 00:30"))

    def test_tail_without_new_flux_records_is_empty(self):
        no_flux = _frame("2026-01-02 00:00", 0, Fc=1.0)
        out = self._build(
            {"flux": no_flux, "met": _frame("2026-01-02 00:00", 6, Ta=2.0)},
            start_date=pd.Timestamp("2026-01-02 00:00"),
        )
        self.assertEqual(len(out), 0)


if __name__ == "__main__":
    unittest.main()
