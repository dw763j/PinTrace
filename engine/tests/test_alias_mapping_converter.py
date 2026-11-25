from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_builder.alias_mapping_converter import build_alias_mapping_for_targets
from stages.utils import get_pkg_name


class AliasMappingConverterTests(unittest.TestCase):
    def test_revision_package_alias_override(self) -> None:
        self.assertEqual(get_pkg_name("ckeditor_uploader", {}), "django-ckeditor")

    def test_convert_pipreqs_mapping_for_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mapping_path = root / "mapping.json"
            out_path = root / "alias.json"
            payload = {
                "mapping": {
                    "cv2": "opencv-python",
                    "yaml": "pyyaml",
                    "requests": "requests",
                }
            }
            mapping_path.write_text(json.dumps(payload), encoding="utf-8")

            stats = build_alias_mapping_for_targets(
                mapping_path=str(mapping_path),
                target_packages={"opencv-python", "requests"},
                out_path=str(out_path),
            )
            alias_payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertIn("opencv-python", alias_payload)
            self.assertEqual(alias_payload["opencv-python"], ["cv2"])
            self.assertNotIn("requests", alias_payload)  # self alias should be filtered
            self.assertEqual(stats["canonical_with_aliases"], 1)


if __name__ == "__main__":
    unittest.main()
