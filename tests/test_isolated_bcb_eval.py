from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate.d1.isolated_eval import extract_pinned_requirements


class IsolatedBcbEvalTests(unittest.TestCase):
    def test_extract_pinned_requirements_with_mapping(self) -> None:
        code = "\n".join(
            [
                "import cv2  # VERSION=4.9.0.80",
                "from bs4 import BeautifulSoup  # VERSION=4.12.3",
                "import random",
                "import numpy as np  # VERSION=1.26.4",
            ]
        )
        mapping = {"cv2": "opencv-python", "bs4": "beautifulsoup4"}
        specs = extract_pinned_requirements(
            code,
            mapping=mapping,
            python_version="3.12",
        )
        self.assertIn("opencv-python==4.9.0.80", specs)
        self.assertIn("beautifulsoup4==4.12.3", specs)
        self.assertIn("numpy==1.26.4", specs)
        # stdlib import without VERSION should not appear
        self.assertNotIn("random==", " ".join(specs))


if __name__ == "__main__":
    unittest.main()
