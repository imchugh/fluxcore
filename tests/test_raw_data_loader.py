"""Tests for raw_data_loader's start_date-aware fast read path.

load_raw_data_since is a performance-only variant of load_raw_data (tail-peek
+ binary search to skip parsing rows a start_date filter would discard
anyway). The tests below treat `load_raw_data(...)` trimmed to
`df.index >= start_date` as the golden reference and assert the fast path
always produces the same result.

Fixtures are built with the production TOA5/EddyPro writers
(file_io.write_toa5_csv / write_eddypro_csv) so file content matches real
quoting/formatting conventions rather than hand-crafted text.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from domain.constants import DATA_TIME_FORMAT
from infrastructure import file_io
from services.data import raw_data_loader as rdl

TOA5_HEADERS = [
    ["TOA5", "TestStation", "CR6", "12345", "CR6.Std.99", "CPU:test.CR6", "0", "flux"],
    ["TIMESTAMP", "RECORD", "Val1", "Val2"],
    ["TS", "RN", "W/m^2", "degC"],
    ["", "", "Avg", "Avg"],
]

EDDYPRO_HEADERS = [
    ["DATAH", "filename", "date", "time", "Val1", "Val2"],
    ["", "", "", "", "W/m^2", "degC"],
]


def _toa5_frame(timestamps: list[pd.Timestamp]) -> pd.DataFrame:
    n = len(timestamps)
    return pd.DataFrame(
        {
            "TIMESTAMP": [ts.strftime(DATA_TIME_FORMAT) for ts in timestamps],
            "RECORD": list(range(n)),
            "Val1": [float(i) for i in range(n)],
            "Val2": [float(i) * 0.5 for i in range(n)],
        }
    )


def _write_toa5(path: Path, timestamps: list[pd.Timestamp]) -> None:
    file_io.write_toa5_csv(
        file_path=path, headers=TOA5_HEADERS, data=_toa5_frame(timestamps)
    )


def _eddypro_frame(timestamps: list[pd.Timestamp]) -> pd.DataFrame:
    n = len(timestamps)
    return pd.DataFrame(
        {
            "DATAH": ["DATA"] * n,
            "filename": ["f.txt"] * n,
            "date": [ts.strftime("%Y-%m-%d") for ts in timestamps],
            "time": [ts.strftime("%H:%M:%S") for ts in timestamps],
            "Val1": [float(i) for i in range(n)],
            "Val2": [float(i) * 0.5 for i in range(n)],
        }
    )


def _write_eddypro(path: Path, timestamps: list[pd.Timestamp]) -> None:
    file_io.write_eddypro_csv(
        file_path=path, headers=EDDYPRO_HEADERS, data=_eddypro_frame(timestamps)
    )


def _corrupt_lines(
    path: Path, start_line: int, count: int, n_header_lines: int = 4
) -> None:
    """Overwrite `count` data lines starting at `start_line` with garbage text."""
    lines = path.read_text().splitlines(keepends=True)
    for i in range(start_line, start_line + count):
        idx = n_header_lines + i
        if idx < len(lines):
            lines[idx] = "GARBAGE,NOT,A,VALID,TOA5,ROW\n"
    path.write_text("".join(lines))


class TestLoadRawDataSinceTOA5(unittest.TestCase):
    """Golden-equivalence matrix: load_raw_data_since vs. full read + trim."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls._tmp.name) / "master.dat"
        cls.timestamps = list(pd.date_range("2022-01-01", "2024-01-01", freq="h"))
        _write_toa5(cls.path, cls.timestamps)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _assert_matches_golden(self, start_date):
        full = rdl.load_raw_data(file_path=self.path, file_format="TOA5")
        expected = full[full.index >= start_date]
        actual = rdl.load_raw_data_since(
            file_path=self.path, file_format="TOA5", start_date=start_date
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_matches_full_read_mid_history(self):
        self._assert_matches_golden(pd.Timestamp("2023-06-15 12:00:00"))

    def test_matches_full_read_before_first_record(self):
        self._assert_matches_golden(pd.Timestamp("2020-01-01"))

    def test_matches_full_read_on_exact_existing_timestamp(self):
        self._assert_matches_golden(self.timestamps[1000])

    def test_matches_full_read_on_nonexistent_instant(self):
        self._assert_matches_golden(self.timestamps[1000] + pd.Timedelta(minutes=17))

    def test_matches_full_read_at_year_boundary(self):
        self._assert_matches_golden(pd.Timestamp("2023-01-01 00:00:00"))

    def test_empty_after_last_record(self):
        # The tail-peek short-circuit returns a bare pd.DataFrame() here
        # (no parse at all) rather than a correctly-columned empty frame —
        # fine, since _build_file_group_dataframe only checks `.empty`
        # before discarding it, never its columns/dtypes.
        start = self.timestamps[-1] + pd.Timedelta(days=1)
        actual = rdl.load_raw_data_since(
            file_path=self.path, file_format="TOA5", start_date=start
        )
        self.assertTrue(actual.empty)


class TestLoadRawDataSinceGap(unittest.TestCase):
    """A data gap straddling the calendar-year boundary."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls._tmp.name) / "master.dat"
        before = list(pd.date_range("2022-01-01", "2022-10-01", freq="h"))
        after = list(pd.date_range("2023-03-01", "2023-08-01", freq="h"))
        cls.timestamps = before + after
        _write_toa5(cls.path, cls.timestamps)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_start_date_inside_gap_matches_full_read(self):
        start = pd.Timestamp("2023-01-01")
        full = rdl.load_raw_data(file_path=self.path, file_format="TOA5")
        expected = full[full.index >= start]
        actual = rdl.load_raw_data_since(
            file_path=self.path, file_format="TOA5", start_date=start
        )
        pd.testing.assert_frame_equal(actual, expected)


class TestLoadRawDataSinceEdgeCases(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_headers_only_file_returns_empty(self):
        path = self.dir / "empty.dat"
        _write_toa5(path, [])
        actual = rdl.load_raw_data_since(
            file_path=path, file_format="TOA5", start_date=pd.Timestamp("2023-01-01")
        )
        self.assertTrue(actual.empty)

    def test_single_row_file(self):
        path = self.dir / "single.dat"
        ts = [pd.Timestamp("2023-06-01 00:00:00")]
        _write_toa5(path, ts)
        full = rdl.load_raw_data(file_path=path, file_format="TOA5")

        with self.subTest("start before the only record"):
            start = pd.Timestamp("2023-01-01")
            expected = full[full.index >= start]
            actual = rdl.load_raw_data_since(
                file_path=path, file_format="TOA5", start_date=start
            )
            pd.testing.assert_frame_equal(actual, expected)

        with self.subTest("start after the only record"):
            start = pd.Timestamp("2024-01-01")
            actual = rdl.load_raw_data_since(
                file_path=path, file_format="TOA5", start_date=start
            )
            self.assertTrue(actual.empty)

    def test_file_smaller_than_search_margin(self):
        path = self.dir / "small.dat"
        ts = list(pd.date_range("2023-01-01", periods=20, freq="h"))
        _write_toa5(path, ts)
        start = ts[10]
        full = rdl.load_raw_data(file_path=path, file_format="TOA5")
        expected = full[full.index >= start]
        actual = rdl.load_raw_data_since(
            file_path=path, file_format="TOA5", start_date=start
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_duplicate_timestamps_straddling_start_date(self):
        path = self.dir / "dup.dat"
        base = list(pd.date_range("2022-01-01", "2023-12-01", freq="h"))
        start = base[5000]
        ts = base[:5000] + [start] + base[5000:]  # duplicate exactly at start_date
        _write_toa5(path, ts)
        full = rdl.load_raw_data(file_path=path, file_format="TOA5")
        expected = full[full.index >= start]
        actual = rdl.load_raw_data_since(
            file_path=path, file_format="TOA5", start_date=start
        )
        pd.testing.assert_frame_equal(actual, expected)


class TestLoadRawDataSinceCorruptLines(unittest.TestCase):
    """A block of malformed lines near the middle of the file.

    Both the fast path and load_raw_data use on_bad_lines="skip", so this
    also exercises the binary search's "ambiguous probe" branch whenever a
    probe happens to land inside the corrupted block.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls._tmp.name) / "corrupt.dat"
        cls.timestamps = list(pd.date_range("2022-01-01", "2024-01-01", freq="h"))
        _write_toa5(cls.path, cls.timestamps)
        _corrupt_lines(cls.path, start_line=len(cls.timestamps) // 2, count=25)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_matches_full_read_with_corrupt_lines_present(self):
        start = self.timestamps[len(self.timestamps) // 2 - 200]
        full = rdl.load_raw_data(file_path=self.path, file_format="TOA5")
        expected = full[full.index >= start]
        actual = rdl.load_raw_data_since(
            file_path=self.path, file_format="TOA5", start_date=start
        )
        pd.testing.assert_frame_equal(actual, expected)


class TestLoadRawDataSinceEddyPro(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls._tmp.name) / "master.txt"
        cls.timestamps = list(pd.date_range("2022-01-01", "2024-01-01", freq="h"))
        _write_eddypro(cls.path, cls.timestamps)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_matches_full_read_mid_history(self):
        start = pd.Timestamp("2023-06-15 12:00:00")
        full = rdl.load_raw_data(file_path=self.path, file_format="EddyPro")
        expected = full[full.index >= start]
        actual = rdl.load_raw_data_since(
            file_path=self.path, file_format="EddyPro", start_date=start
        )
        pd.testing.assert_frame_equal(actual, expected)


class TestLoadRawDataSinceFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls._tmp.name) / "master.dat"
        cls.timestamps = list(pd.date_range("2022-01-01", "2024-01-01", freq="h"))
        _write_toa5(cls.path, cls.timestamps)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_fallback_matches_golden_when_fast_path_raises(self):
        start = self.timestamps[500]
        full = rdl.load_raw_data(file_path=self.path, file_format="TOA5")
        expected = full[full.index >= start]
        with mock.patch.object(
            file_io, "find_seek_offset", side_effect=RuntimeError("boom")
        ):
            actual = rdl.load_raw_data_since(
                file_path=self.path, file_format="TOA5", start_date=start
            )
        pd.testing.assert_frame_equal(actual, expected)

    def test_read_csv_kwargs_always_take_fallback(self):
        # usecols alone would break the fast path's headerless column
        # alignment, so any read_csv_kwargs must skip the fast path
        # entirely rather than attempt (and mis-resolve) it.
        start = self.timestamps[500]
        full = rdl.load_raw_data(
            file_path=self.path, file_format="TOA5", usecols=["TIMESTAMP", "Val1"]
        )
        expected = full[full.index >= start]
        with mock.patch.object(
            file_io,
            "find_seek_offset",
            side_effect=AssertionError("fast path should be skipped"),
        ):
            actual = rdl.load_raw_data_since(
                file_path=self.path,
                file_format="TOA5",
                start_date=start,
                usecols=["TIMESTAMP", "Val1"],
            )
        pd.testing.assert_frame_equal(actual, expected)


class TestGetDataAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls._tmp.name) / "master.dat"
        cls.timestamps = list(pd.date_range("2022-01-01", "2024-01-01", freq="h"))
        _write_toa5(cls.path, cls.timestamps)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_dispatches_to_load_raw_data_since_when_start_date_given(self):
        start = self.timestamps[500]
        loader = rdl.get_data_adapter(system_type="CSI", start_date=start)
        full = rdl.load_raw_data(file_path=self.path, file_format="TOA5")
        expected = full[full.index >= start]
        pd.testing.assert_frame_equal(loader(self.path), expected)

    def test_dispatches_to_load_raw_data_when_start_date_none(self):
        loader = rdl.get_data_adapter(system_type="CSI")
        expected = rdl.load_raw_data(file_path=self.path, file_format="TOA5")
        pd.testing.assert_frame_equal(loader(self.path), expected)


if __name__ == "__main__":
    unittest.main()
