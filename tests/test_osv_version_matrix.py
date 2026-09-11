from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stages.osv_version_matrix import build_osv_version_matrix


class OsvVersionMatrixTests(unittest.TestCase):
    def test_build_matrix_filters_missing_pypi_and_expands_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            osv_dir = root / "osv"
            pypi_dir = root / "pypi"
            osv_dir.mkdir(parents=True, exist_ok=True)
            pypi_dir.mkdir(parents=True, exist_ok=True)

            # package keep-me exists in local pypi cache
            keep_payload = {
                "releases": {
                    "0.9.0": [],
                    "1.0.0": [],
                    "1.5.0": [],
                    "2.0.0": [],
                    "2.1.0": [],
                }
            }
            (pypi_dir / "pypi#keep-me.json").write_text(json.dumps(keep_payload), encoding="utf-8")

            osv_payload = {
                "id": "OSV-TEST-1",
                "affected": [
                    {
                        "package": {"name": "keep-me"},
                        "versions": ["0.9.0"],
                        "ranges": [
                            {
                                "type": "ECOSYSTEM",
                                "events": [{"introduced": "1.0.0"}, {"fixed": "2.0.0"}],
                            }
                        ],
                    },
                    {
                        "package": {"name": "removed-malicious-pkg"},
                        "versions": ["1.0.0"],
                    },
                ],
            }
            (osv_dir / "entry.json").write_text(json.dumps(osv_payload), encoding="utf-8")

            result = build_osv_version_matrix(osv_dir=str(osv_dir), pypi_info_dir=str(pypi_dir))
            versions = result["packages"]["keep-me"]["versions"]

            self.assertIn("0.9.0", versions)
            self.assertIn("1.0.0", versions)
            self.assertIn("1.5.0", versions)
            self.assertNotIn("2.0.0", versions)  # fixed is exclusive
            self.assertEqual(versions["1.0.0"]["vuln_count"], 1)
            self.assertEqual(result["stats"]["packages_in_matrix"], 1)
            self.assertGreater(result["stats"]["ignored_missing_package_in_pypi_info"], 0)

    def test_alias_deduplication(self) -> None:
        """GHSA and PYSEC aliases for same vuln should count as 1, not 2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            osv_dir = root / "osv"
            pypi_dir = root / "pypi"
            osv_dir.mkdir(parents=True, exist_ok=True)
            pypi_dir.mkdir(parents=True, exist_ok=True)

            pypi_payload = {"releases": {"1.0.0": [{"upload_time_iso_8601": "2020-01-01T00:00:00Z"}]}}
            (pypi_dir / "pypi#alias-pkg.json").write_text(json.dumps(pypi_payload), encoding="utf-8")

            # Same vuln, two files (GHSA + PYSEC aliases)
            ghsa = {
                "id": "GHSA-jw8x-6495-233v",
                "aliases": ["CVE-2024-5206", "PYSEC-2024-110"],
                "affected": [{"package": {"name": "alias-pkg"}, "versions": ["1.0.0"]}],
            }
            pysec = {
                "id": "PYSEC-2024-110",
                "aliases": ["GHSA-jw8x-6495-233v", "CVE-2024-5206"],
                "affected": [{"package": {"name": "alias-pkg"}, "versions": ["1.0.0"]}],
            }
            (osv_dir / "GHSA-jw8x-6495-233v.json").write_text(json.dumps(ghsa), encoding="utf-8")
            (osv_dir / "PYSEC-2024-110.json").write_text(json.dumps(pysec), encoding="utf-8")

            result = build_osv_version_matrix(osv_dir=str(osv_dir), pypi_info_dir=str(pypi_dir))
            versions = result["packages"]["alias-pkg"]["versions"]
            self.assertEqual(versions["1.0.0"]["vuln_count"], 1)
            self.assertEqual(len(versions["1.0.0"]["osv_ids"]), 1)


if __name__ == "__main__":
    unittest.main()
