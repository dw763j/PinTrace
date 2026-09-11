from __future__ import annotations

from stages.metrics import aggregate_metrics


def _record(task_id: str, per_lib: list[dict], findings: list[dict] | None) -> dict:
    record = {
        "task_id": task_id,
        "per_lib": per_lib,
        "compat_results": {"ty": {"is_compatible": True, "errors": []}},
    }
    if findings is not None:
        record["vuln_findings"] = findings
    return record


def test_constraint_funnel_counts_and_identity():
    exact = {
        "pypi_name": "requests",
        "specified_version": "2.31.0",
        "resolved_version": "2.31.0",
        "version_exists": True,
        "declaration_source": "inline_annotation",
        "constraint_kind": "exact",
        "resolution_origin": "model_exact",
    }
    invalid_range = {
        "pypi_name": "numpy",
        "specified_version": None,
        "version_spec": ">=9999",
        "resolved_version": None,
        "version_exists": False,
        "declaration_source": "manifest",
        "constraint_kind": "range",
        "resolution_origin": "unresolved",
    }
    bare = {
        "pypi_name": "pandas",
        "specified_version": None,
        "version_spec": None,
        "resolved_version": "2.2.0",
        "version_exists": True,
        "declaration_source": "code_import_only",
        "constraint_kind": "bare",
        "resolution_origin": "unconstrained_resolver",
    }
    vulnerable = {
        "pypi_name": "requests",
        "version": "2.31.0",
        "is_vulnerable": True,
        "constraint_kind": "exact",
        "resolution_origin": "model_exact",
        "dependency_scope": "direct",
    }
    records = [
        _record("a", [exact], [vulnerable]),
        _record("b", [invalid_range], []),
        _record("c", [bare], []),
    ]
    summary = aggregate_metrics(records, "test-model")
    funnel = summary["constraint_funnel"]
    assert (funnel["N"], funnel["S"], funnel["P"], funnel["V"]) == (3, 2, 1, 1)
    product = (
        funnel["incidence"]["rate"]
        * funnel["validity"]["rate"]
        * funnel["conditional_cve"]["rate"]
    )
    assert product == funnel["unconditional_attributable"]["rate"]
    exact_funnel = summary["constraint_funnels_by_kind"]["exact"]
    range_funnel = summary["constraint_funnels_by_kind"]["range"]
    assert (exact_funnel["N"], exact_funnel["S"], exact_funnel["P"], exact_funnel["V"]) == (3, 1, 1, 1)
    assert (range_funnel["N"], range_funnel["S"], range_funnel["P"], range_funnel["V"]) == (3, 1, 0, 0)


def test_security_skip_is_unavailable_not_zero():
    summary = aggregate_metrics([_record("a", [], None)], "test-model")
    assert summary["vulnerability"]["status"] == "unavailable"
    assert summary["vulnerability"]["task_vuln_exposure"] is None
    assert summary["constraint_funnel"]["V"] is None
    assert summary["constraint_funnel"]["conditional_cve"]["rate"] is None
    assert summary["constraint_funnels_by_kind"]["exact"]["V"] is None
    assert summary["constraint_funnels_by_kind"]["range"]["V"] is None
