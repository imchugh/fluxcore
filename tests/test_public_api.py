"""fluxcore's declared public API: every name in __all__ resolves."""

import unittest

import fluxcore


class PublicApiTestCase(unittest.TestCase):
    def test_every_exported_name_resolves(self):
        for name in fluxcore.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(fluxcore, name))

    def test_no_duplicates(self):
        self.assertEqual(len(fluxcore.__all__), len(set(fluxcore.__all__)))

    def test_star_import_matches_all(self):
        namespace = {}
        exec("from fluxcore import *", namespace)
        exported = {k for k in namespace if not k.startswith("__")}
        self.assertEqual(exported, set(fluxcore.__all__))
