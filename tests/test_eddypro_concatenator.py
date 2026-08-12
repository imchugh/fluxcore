"""Tests for eddypro_concatenator.select_new_slaves.

Covers the pre-filter's use of raw_data_loader.peek_last_timestamp,
including the case where the master has no peekable record (header-only)
— peek_last_timestamp returns None there, and select_new_slaves must treat
that as "nothing to filter against" rather than raising or dropping every
candidate.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from infrastructure import file_io
from services.data import eddypro_concatenator as epc

EDDYPRO_HEADERS = [
    ["DATAH", "filename", "date", "time", "Val1"],
    ["", "", "", "", "W/m^2"],
]


def _eddypro_frame(timestamps: list[pd.Timestamp]) -> pd.DataFrame:
    n = len(timestamps)
    return pd.DataFrame(
        {
            "DATAH": ["DATA"] * n,
            "filename": ["f.txt"] * n,
            "date": [ts.strftime("%Y-%m-%d") for ts in timestamps],
            "time": [ts.strftime("%H:%M:%S") for ts in timestamps],
            "Val1": [float(i) for i in range(n)],
        }
    )


def _write_master(path: Path, timestamps: list[pd.Timestamp]) -> None:
    file_io.write_eddypro_csv(
        file_path=path, headers=EDDYPRO_HEADERS, data=_eddypro_frame(timestamps)
    )


class TestSelectNewSlaves(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_filters_candidates_older_than_master(self):
        master = self.dir / "master.txt"
        _write_master(master, list(pd.date_range("2023-06-01", "2023-06-10", freq="h")))

        candidates = [
            self.dir / "2023-06-05_EP-Summary.txt",  # covered by master, dropped
            self.dir / "2023-06-10_EP-Summary.txt",  # master's last day, kept
            self.dir / "2023-06-12_EP-Summary.txt",  # newer than master, kept
            self.dir / "not-a-date_EP-Summary.txt",  # unparseable name, kept
        ]

        kept = epc.select_new_slaves(master=master, candidates=candidates)

        self.assertEqual(
            {p.name for p in kept},
            {
                "2023-06-10_EP-Summary.txt",
                "2023-06-12_EP-Summary.txt",
                "not-a-date_EP-Summary.txt",
            },
        )

    def test_header_only_master_keeps_every_candidate(self):
        master = self.dir / "master.txt"
        _write_master(master, [])  # no data rows -> peek_last_timestamp is None

        candidates = [
            self.dir / "2023-06-05_EP-Summary.txt",
            self.dir / "2099-01-01_EP-Summary.txt",
        ]

        kept = epc.select_new_slaves(master=master, candidates=candidates)

        self.assertEqual({p.name for p in kept}, {p.name for p in candidates})


if __name__ == "__main__":
    unittest.main()
