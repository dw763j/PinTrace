from __future__ import annotations

from evaluate.build_canonical_tables import _vulnerability_join_key
from evaluate.generate_analysis_tables import _funnel_row


def test_vulnerability_join_uses_isolated_constraint_version():
    dependency = {
        "pypi_name": "Django",
        "resolved_version": None,
        "constraint_resolved_version": "5.2.1",
        "resolution_origin": "model_constrained_resolver",
        "dependency_scope": "direct",
    }
    finding = {
        "pypi_name": "django",
        "version": "5.2.1",
        "resolution_origin": "model_constrained_resolver",
        "dependency_scope": "direct",
    }
    assert _vulnerability_join_key(dependency) == _vulnerability_join_key(finding)


def test_table_funnels_keep_exact_and_range_separate():
    rows = [
        {
            "security_status": "available",
            "has_model_constraint": True,
            "has_valid_model_constraint": True,
            "has_model_cve": True,
            "has_model_exact_constraint": True,
            "has_valid_model_exact_constraint": True,
            "has_model_exact_cve": True,
            "has_model_range_constraint": False,
            "has_valid_model_range_constraint": False,
            "has_model_range_cve": False,
        },
        {
            "security_status": "available",
            "has_model_constraint": True,
            "has_valid_model_constraint": True,
            "has_model_cve": False,
            "has_model_exact_constraint": False,
            "has_valid_model_exact_constraint": False,
            "has_model_exact_cve": False,
            "has_model_range_constraint": True,
            "has_valid_model_range_constraint": True,
            "has_model_range_cve": False,
        },
    ]
    group = ("d2", "requirements.txt", "model", "full")
    exact = _funnel_row(group, rows, "exact")
    range_row = _funnel_row(group, rows, "range")
    assert (exact["N"], exact["S"], exact["P"], exact["V"]) == (2, 1, 1, 1)
    assert (range_row["N"], range_row["S"], range_row["P"], range_row["V"]) == (2, 1, 1, 0)
