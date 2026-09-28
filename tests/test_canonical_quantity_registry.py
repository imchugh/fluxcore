"""Tests for services/metadata/core/canonical_quantity_registry.py."""

import unittest

from domain.enums import StatisticType, VariableType
from services.metadata.core.canonical_quantity_registry import (
    build_canonical_quantity_registry,
)


class CanonicalQuantityRegistryTestCase(unittest.TestCase):
    def setUp(self):
        build_canonical_quantity_registry.cache_clear()
        self.addCleanup(build_canonical_quantity_registry.cache_clear)
        self.registry = build_canonical_quantity_registry()

    def test_real_registry_loads(self):
        self.assertTrue(self.registry.has_quantity("Fco2"))
        self.assertGreater(len(self.registry.quantities), 50)

    def test_base_metadata_holds_no_value_range(self):
        # Value ranges live only in configs/qc/_range_defaults.yml.
        base = self.registry.get_base_metadata("Fco2")
        self.assertFalse(hasattr(base, "valid_min"))
        self.assertFalse(hasattr(base, "valid_max"))

    def test_resolve_metadata_for_each_variable_form(self):
        for kwargs in (
            {"variable_type": VariableType.QUALITY_FLAG},
            {"variable_type": VariableType.COUNTER},
            {
                "variable_type": VariableType.CONTINUOUS,
                "statistic_type": StatisticType.VAR,
            },
        ):
            with self.subTest(kwargs=kwargs):
                quantity = "Ux" if "statistic_type" in kwargs else "Diag"
                meta = self.registry.resolve_metadata(quantity=quantity, **kwargs)
                self.assertFalse(hasattr(meta, "valid_max"))


if __name__ == "__main__":
    unittest.main()
