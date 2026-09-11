from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stages.compact_checker import _infer_test_utilized_tpl


def test_infer_test_utilized_tpl_detects_pytz_outside_specified_tpl() -> None:
    test_code = "\n".join(
        [
            "import unittest",
            "from datetime import datetime",
            "from decimal import Decimal",
            "import numpy as np",
            "import pytz",
            "",
            "class TestCases(unittest.TestCase):",
            "    def test_x(self):",
            "        assert datetime is not None and Decimal is not None and np is not None and pytz is not None",
        ]
    )
    # specified TPL only contains numpy (stdlib imports should be ignored)
    pkg_specs = ["numpy"]
    extra = _infer_test_utilized_tpl(test_code, pkg_specs=pkg_specs)
    assert "pytz" in extra
    assert "numpy" not in extra

if __name__ == "__main__":
    test_infer_test_utilized_tpl_detects_pytz_outside_specified_tpl()
    print("All tests passed")