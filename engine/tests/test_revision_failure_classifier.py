from __future__ import annotations

from stages.failure_classifier import classify_compatibility_layer


def _ty(rule: str, message: str = "diagnostic") -> dict:
    return {"is_compatible": False, "errors": [{"rule": rule, "message": message}]}


def test_revision_compatibility_layers():
    assert classify_compatibility_layer({"is_compatible": True, "errors": []}) == "compatible"
    assert classify_compatibility_layer(_ty("ty-install-error")) == "environment_build"
    assert classify_compatibility_layer(_ty("compat-runtime-error")) == "infrastructure"
    assert classify_compatibility_layer(_ty("invalid-syntax")) == "code_or_syntax"
    assert classify_compatibility_layer(_ty("unresolved-import")) == "import_or_initialization"
    assert classify_compatibility_layer(_ty("unresolved-attribute")) == "api_mismatch_candidate"
    assert classify_compatibility_layer(
        {"is_compatible": True, "errors": []}, {"status": "fail"}
    ) == "dynamic_semantic"
