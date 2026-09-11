from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stages.vuln_checker import (
    check_vulnerability,
    enrich_cves_with_local_dump,
    osv_range_mode_stats,
)


class VulnCheckTests(unittest.TestCase):
    def test_range_matching_fixed_exclusive(self) -> None:
        osv_index = {
            "demo-pkg": {
                "by_version": {},
                "ranges": [
                    {
                        "osv_id": "OSV-1",
                        "cve_ids": ["CVE-2024-1111"],
                        "events": [{"introduced": "1.0.0"}, {"fixed": "2.0.0"}],
                    }
                ],
                "vuln_ids": ["OSV-1"],
            }
        }
        cve_cache = {"CVE-2024-1111": {"severity": "HIGH", "score": 7.5}}

        hit = check_vulnerability("demo-pkg", "1.5.0", osv_index, cve_cache)
        miss = check_vulnerability("demo-pkg", "2.0.0", osv_index, cve_cache)

        self.assertTrue(hit.is_vulnerable)
        self.assertEqual(hit.max_severity, "HIGH")
        self.assertFalse(miss.is_vulnerable)

    def test_range_matching_last_affected_inclusive(self) -> None:
        osv_index = {
            "demo-pkg": {
                "by_version": {},
                "ranges": [
                    {
                        "osv_id": "OSV-2",
                        "cve_ids": ["CVE-2024-2222"],
                        "events": [{"introduced": "0"}, {"last_affected": "1.2.3"}],
                    }
                ],
                "vuln_ids": ["OSV-2"],
            }
        }
        cve_cache = {"CVE-2024-2222": {"severity": "MEDIUM", "score": 5.0}}

        hit = check_vulnerability("demo-pkg", "1.2.3", osv_index, cve_cache)
        miss = check_vulnerability("demo-pkg", "1.2.4", osv_index, cve_cache)

        self.assertTrue(hit.is_vulnerable)
        self.assertFalse(miss.is_vulnerable)

    def test_local_cve_dump_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cve_dir = root / "2024" / "53xxx"
            cve_dir.mkdir(parents=True, exist_ok=True)
            cve_path = cve_dir / "CVE-2024-9999.json"
            payload = {
                "containers": {
                    "cna": {
                        "metrics": [
                            {
                                "cvssV3_1": {
                                    "baseScore": 9.8,
                                    "baseSeverity": "CRITICAL",
                                }
                            }
                        ]
                    }
                }
            }
            cve_path.write_text(json.dumps(payload), encoding="utf-8")

            cache = enrich_cves_with_local_dump(
                ["CVE-2024-9999"],
                cve_dump_dir=str(root),
                cve_cache_path=str(root / "cache.json"),
                cve_file_index_cache_path=str(root / "index.json"),
            )
            self.assertEqual(cache["CVE-2024-9999"]["severity"], "CRITICAL")
            self.assertEqual(cache["CVE-2024-9999"]["score"], 9.8)

    def test_osv_range_mode_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            advisory_a = {
                "id": "OSV-A",
                "affected": [
                    {
                        "package": {"name": "a"},
                        "ranges": [{"type": "ECOSYSTEM", "events": [{"introduced": "0"}, {"fixed": "1.0.0"}]}],
                        "versions": [],
                    }
                ],
            }
            advisory_b = {
                "id": "OSV-B",
                "affected": [
                    {
                        "package": {"name": "b"},
                        "versions": ["1.0.0"],
                        "ranges": [],
                    }
                ],
            }
            (root / "a.json").write_text(json.dumps(advisory_a), encoding="utf-8")
            (root / "b.json").write_text(json.dumps(advisory_b), encoding="utf-8")

            stats = osv_range_mode_stats(str(root))
            self.assertEqual(stats["total_advisories"], 2)
            self.assertEqual(stats["advisories_with_ranges"], 1)
            self.assertEqual(stats["advisories_with_versions"], 1)


if __name__ == "__main__":
    unittest.main()
