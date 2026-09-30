"""Unit tests for orchestration/incremental_zarr.py."""

import dataclasses
import pathlib
import tempfile
import unittest
from unittest import mock

import pandas as pd

from orchestration import incremental_zarr


@dataclasses.dataclass(frozen=True)
class _Cfg:
    a: int
    b: dict


class HashConfigTestCase(unittest.TestCase):
    def test_same_content_same_hash_regardless_of_dict_key_order(self):
        cfg1 = _Cfg(a=1, b={"x": 1, "y": 2})
        cfg2 = _Cfg(a=1, b={"y": 2, "x": 1})
        self.assertEqual(
            incremental_zarr.hash_config(cfg1), incremental_zarr.hash_config(cfg2)
        )

    def test_different_content_different_hash(self):
        cfg1 = _Cfg(a=1, b={"x": 1})
        cfg2 = _Cfg(a=2, b={"x": 1})
        self.assertNotEqual(
            incremental_zarr.hash_config(cfg1), incremental_zarr.hash_config(cfg2)
        )

    def test_argument_order_matters(self):
        self.assertNotEqual(
            incremental_zarr.hash_config("a", "b"),
            incremental_zarr.hash_config("b", "a"),
        )

    def test_set_hashes_independent_of_iteration_order(self):
        cfg1 = _Cfg(a=1, b={"s": {"alpha", "beta", "gamma"}})
        cfg2 = _Cfg(a=1, b={"s": {"gamma", "alpha", "beta"}})
        self.assertEqual(
            incremental_zarr.hash_config(cfg1), incremental_zarr.hash_config(cfg2)
        )


class IncrementalUpdateTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store_path = pathlib.Path(self._tmp.name) / "store.zarr"
        self.logger = mock.Mock()

    def test_seeds_when_store_missing(self):
        full_rebuild = mock.Mock(return_value=self.store_path)
        produce_tail = mock.Mock()

        result = incremental_zarr.incremental_update(
            store_path=self.store_path,
            config_hash="h1",
            produce_tail=produce_tail,
            full_rebuild=full_rebuild,
            logger=self.logger,
            label="test",
        )

        self.assertEqual(result, self.store_path)
        full_rebuild.assert_called_once()
        produce_tail.assert_not_called()

    def test_rebuilds_on_config_hash_mismatch(self):
        self.store_path.mkdir()
        full_rebuild = mock.Mock(return_value=self.store_path)
        produce_tail = mock.Mock()

        with mock.patch("orchestration.incremental_zarr.xr.open_zarr") as open_zarr:
            open_zarr.return_value = mock.Mock(attrs={"config_hash": "old"})
            result = incremental_zarr.incremental_update(
                store_path=self.store_path,
                config_hash="new",
                produce_tail=produce_tail,
                full_rebuild=full_rebuild,
                logger=self.logger,
                label="test",
            )

        self.assertEqual(result, self.store_path)
        full_rebuild.assert_called_once()
        produce_tail.assert_not_called()

    def test_appends_tail_when_hash_matches(self):
        self.store_path.mkdir()
        full_rebuild = mock.Mock()
        tail_ds = mock.Mock()
        tail_ds.sizes = {"time": 3}
        produce_tail = mock.Mock(return_value=tail_ds)

        with (
            mock.patch("orchestration.incremental_zarr.xr.open_zarr") as open_zarr,
            mock.patch(
                "orchestration.incremental_zarr.last_store_timestamp"
            ) as last_ts,
            mock.patch(
                "orchestration.incremental_zarr.file_io.append_zarr"
            ) as append_zarr,
        ):
            open_zarr.return_value = mock.Mock(attrs={"config_hash": "h1"})
            last_ts.return_value = pd.Timestamp("2020-01-01")

            result = incremental_zarr.incremental_update(
                store_path=self.store_path,
                config_hash="h1",
                produce_tail=produce_tail,
                full_rebuild=full_rebuild,
                logger=self.logger,
                label="test",
            )

        self.assertEqual(result, self.store_path)
        produce_tail.assert_called_once_with(pd.Timestamp("2020-01-01"))
        append_zarr.assert_called_once_with(ds=tail_ds, store_path=self.store_path)
        full_rebuild.assert_not_called()

    def test_empty_tail_is_a_noop(self):
        self.store_path.mkdir()
        full_rebuild = mock.Mock()
        empty_ds = mock.Mock()
        empty_ds.sizes = {"time": 0}
        produce_tail = mock.Mock(return_value=empty_ds)

        with (
            mock.patch("orchestration.incremental_zarr.xr.open_zarr") as open_zarr,
            mock.patch(
                "orchestration.incremental_zarr.last_store_timestamp"
            ) as last_ts,
            mock.patch(
                "orchestration.incremental_zarr.file_io.append_zarr"
            ) as append_zarr,
        ):
            open_zarr.return_value = mock.Mock(attrs={"config_hash": "h1"})
            last_ts.return_value = pd.Timestamp("2020-01-01")

            result = incremental_zarr.incremental_update(
                store_path=self.store_path,
                config_hash="h1",
                produce_tail=produce_tail,
                full_rebuild=full_rebuild,
                logger=self.logger,
                label="test",
            )

        self.assertEqual(result, self.store_path)
        append_zarr.assert_not_called()
        full_rebuild.assert_not_called()

    def test_exception_falls_back_to_full_rebuild(self):
        self.store_path.mkdir()
        full_rebuild = mock.Mock(return_value=self.store_path)

        def _boom(checkpoint):
            raise RuntimeError("boom")

        with mock.patch("orchestration.incremental_zarr.xr.open_zarr") as open_zarr:
            open_zarr.return_value = mock.Mock(attrs={"config_hash": "h1"})

            result = incremental_zarr.incremental_update(
                store_path=self.store_path,
                config_hash="h1",
                produce_tail=_boom,
                full_rebuild=full_rebuild,
                logger=self.logger,
                label="test",
            )

        self.assertEqual(result, self.store_path)
        full_rebuild.assert_called_once()


if __name__ == "__main__":
    unittest.main()
