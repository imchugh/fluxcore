"""Tests for the generic-core SiteMetadata file loader (no TERN dependency).

Fixture is the real AliceSpringsMulga record from configs/site_metadata.yml,
unwrapped from its {site_name: <record>} multi-site mapping into a
standalone single-site file — the shape a non-TERN caller would write by
hand, demonstrated with real production data rather than an invented one.
"""

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from services.metadata.site_metadata_builder import build_site_metadata_from_file

REAL_RECORD_YAML = """\
id: https://w3id.org/tern/resources/fd2809a2-e705-4961-b09c-7e9ae1ffa87e
fluxnet_id: AU-ASM
date_commissioned: '2010-09-01'
latitude: '-22.2828'
longitude: '133.2493'
elevation: '606.0'
time_step: '30.0'
freq_hz: '10.0'
canopy_height: '4.4'
soil: Acidic, Red Dermosol
tower_height: '13.7'
vegetation: 'Mulga canopy: Acacia aneura and A. aptaneura, with understorey forbs,
  shrubs, and grasses.'
site_name: AliceSpringsMulga
"""

MISSING_LATITUDE_YAML = """\
time_step: '30.0'
freq_hz: '10.0'
site_name: AliceSpringsMulga
longitude: '133.2493'
"""

MINIMAL_YAML = """\
site_name: TestSite
latitude: '1.0'
longitude: '2.0'
time_step: '30.0'
freq_hz: '10.0'
"""


class SiteMetadataBuilderTestCase(unittest.TestCase):
    def _write_and_build(self, yaml_text):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "site_metadata.yml"
            path.write_text(yaml_text)
            return build_site_metadata_from_file(path)

    def test_builds_from_real_record_with_no_tern_dependency(self):
        metadata = self._write_and_build(REAL_RECORD_YAML)

        self.assertEqual(metadata.site_name, "AliceSpringsMulga")
        self.assertEqual(metadata.latitude, -22.2828)
        self.assertEqual(metadata.time_step, 30)
        self.assertEqual(metadata.freq_hz, 10)
        self.assertEqual(metadata.n_samples, 30 * 60 * 10)

    def test_missing_required_field_raises(self):
        with self.assertRaises(ValidationError):
            self._write_and_build(MISSING_LATITUDE_YAML)

    def test_float_string_time_step_coerces_to_int(self):
        metadata = self._write_and_build(MINIMAL_YAML)

        self.assertEqual(metadata.time_step, 30)
        self.assertIsInstance(metadata.time_step, int)


if __name__ == "__main__":
    unittest.main(verbosity=2)
