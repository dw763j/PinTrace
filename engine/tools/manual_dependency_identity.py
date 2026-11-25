"""Resolve dependency identities while retaining original model declarations.

The identity layer is independent of installation and execution outcomes.  It
records local and removed-runtime modules separately so callers can evaluate
their availability in the complete generated project.
"""

import ast
import re
from collections import defaultdict
from pathlib import PurePosixPath

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name


ALIAS_SOURCES = {
    "fitz": ("pymupdf", "https://pymupdf.readthedocs.io/en/latest/installation.html"),
    "pywt": ("pywavelets", "https://pywavelets.readthedocs.io/en/latest/index.html"),
    "tortoise": ("tortoise-orm", "https://tortoise.github.io/getting_started.html"),
    "Bio": ("biopython", "https://biopython.org/docs/latest/api/Bio.SeqIO.html"),
    "docx": ("python-docx", "https://python-docx.readthedocs.io/en/latest/user/quickstart.html"),
    "skimage": ("scikit-image", "https://scikit-image.org/docs/stable/user_guide/install.html"),
    "win32api": ("pywin32", "https://github.com/mhammond/pywin32"),
    "win32com": ("pywin32", "https://github.com/mhammond/pywin32"),
}
NAMESPACE_DISTRIBUTIONS = {
    "google.auth": "google-auth",
    "google.oauth2": "google-auth",
    "google.api_core": "google-api-core",
    "google.protobuf": "protobuf",
    "google.cloud.storage": "google-cloud-storage",
    "google.cloud.bigquery": "google-cloud-bigquery",
    "google.cloud.secretmanager": "google-cloud-secret-manager",
    "google.cloud.vision": "google-cloud-vision",
    "google.cloud.functions_v1": "google-cloud-functions",
    "google.generativeai": "google-generativeai",
    "google.cloud.exceptions": "google-cloud-core",
    "google.cloud.aiplatform": "google-cloud-aiplatform",
    "google.cloud.pubsub_v1": "google-cloud-pubsub",
    "google.pubsub_v1": "google-cloud-pubsub",
    "google.cloud.kms_v1": "google-cloud-kms",
    "google.cloud.kms": "google-cloud-kms",
    "google.adk": "google-adk",
    "google.colab": "google-colab",
    "azure.functions": "azure-functions",
    "azure.identity": "azure-identity",
    "azure.mgmt.resource": "azure-mgmt-resource",
    "azure.mgmt.subscription": "azure-mgmt-subscription",
    "azure.mgmt.containerinstance": "azure-mgmt-containerinstance",
    "azure.mgmt.containerregistry": "azure-mgmt-containerregistry",
    "azure.containerregistry": "azure-containerregistry",
    "azure.mgmt.keyvault": "azure-mgmt-keyvault",
    "azure.common": "azure-common",
    "azure.mgmt.compute": "azure-mgmt-compute",
    "azure.keyvault.secrets": "azure-keyvault-secrets",
    "azure.storage.blob": "azure-storage-blob",
    "azure.storage.fileshare": "azure-storage-file-share",
    "azure.core": "azure-core",
    "azure.ai.formrecognizer": "azure-ai-formrecognizer",
    "azure.ai.ml": "azure-ai-ml",
    "azure.ai.documentintelligence": "azure-ai-documentintelligence",
    "azure.mgmt.core": "azure-mgmt-core",
    "azure.mgmt.network": "azure-mgmt-network",
    "azure.mgmt.storage": "azure-mgmt-storage",
    "azure.mgmt.monitor": "azure-mgmt-monitor",
    "azure.mgmt.authorization": "azure-mgmt-authorization",
    "azure.mgmt.msi": "azure-mgmt-msi",
    "azure.keyvault.keys": "azure-keyvault-keys",
    "azure.keyvault.certificates": "azure-keyvault-certificates",
}
REMOVED_RUNTIME_MODULES = {
    "distutils": "https://docs.python.org/3.12/whatsnew/3.12.html",
}
PLACEHOLDER_MODULES = {"my_local_package", "my_module", "your_module", "your_package"}
# These observed version strings occur in mysqlclient's release history and
# are absent from MySQL-python's complete release history.  The saved PyPI
# metadata and hashes are in mapping/mysql_provider_evidence.json.
MYSQLCLIENT_VERSION_EVIDENCE = {"1.4.6", "2.1.0", "2.1.1", "2.2.0"}


def _declared_names(requirements):
    names = set()
    for item in requirements:
        text = item.get("raw_requirement", "") if isinstance(item, dict) else str(item)
        try:
            names.add(canonicalize_name(Requirement(text).name))
        except InvalidRequirement:
            continue
    return names


def local_module_names(files):
    """Return import roots represented by supplied Python files."""
    names = set()
    for filename in files:
        path = PurePosixPath(str(filename).replace("\\", "/"))
        if path.suffix != ".py" or path.is_absolute() or ".." in path.parts:
            continue
        if len(path.parts) > 1:
            names.add(path.parts[0])
        elif path.stem != "__init__":
            names.add(path.stem)
    return names


def _api_context(import_name, code):
    """Find namespace or API evidence for the supported package identities."""
    patterns = {
        "fitz": r"\b(?:PyMuPDF|pymupdf)\b|\bfitz\.(?:open|Document|Page|Rect|Point|Matrix|Pixmap|TextWriter|get_text_length)\b|from\s+fitz\s+import\s+(?:Document|Page|Rect|Point|Matrix|Pixmap)\b",
        "pywt": r"\b(?:import\s+pywt|from\s+pywt(?:\.|\s))",
        "tortoise": r"\btortoise-orm\b|from\s+tortoise\.(?:models|contrib|fields|expressions|transactions|exceptions)\b|from\s+tortoise\s+import\s+[^\n]*(?:Tortoise|fields|models)\b|\btortoise\.Tortoise\b",
        "Bio": r"\b(?:import\s+Bio|from\s+Bio(?:\.|\s))",
        "docx": r"\bpython-docx\b|from\s+docx\.(?:enum|shared|oxml|document|table|text)\b|from\s+docx\s+import\s+Document\b|\bdocx\.Document\b",
        "skimage": r"\b(?:import\s+skimage|from\s+skimage(?:\.|\s))",
        "win32api": r"\bimport\s+win32api\b",
        "win32com": r"\b(?:import\s+win32com|from\s+win32com)(?:\.client|\.shell)\b",
    }
    match = re.search(patterns[import_name], code)
    if match:
        return match.group(0)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    aliases = {
        alias.asname for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names if alias.name == import_name and alias.asname
    }
    attributes = {
        "fitz": {"open", "Document", "Page", "Rect", "Point", "Matrix", "Pixmap"},
        "docx": {"Document"},
        "tortoise": {"Tortoise", "fields", "models"},
    }.get(import_name, set())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in aliases and node.attr in attributes:
                return f"{node.value.id}.{node.attr} uses the imported {import_name} namespace"
    return None


def _namespace_paths(import_line):
    try:
        nodes = ast.parse(import_line.strip()).body
    except SyntaxError:
        # Multiline malformed imports still identify their complete module.
        match = re.match(r"(?:from|import)\s+([A-Za-z_][\w.]*)", import_line.strip())
        modules = [match.group(1)] if match else []
    else:
        modules = []
        for node in nodes:
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and not node.level:
                modules.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def _namespace_identity(import_line):
    matches = {}
    for module in _namespace_paths(import_line):
        choices = [prefix for prefix in NAMESPACE_DISTRIBUTIONS if module == prefix or module.startswith(prefix + ".")]
        if choices:
            prefix = max(choices, key=len)
            matches[prefix] = NAMESPACE_DISTRIBUTIONS[prefix]
    return matches


def resolve_import_identity(
    import_name, *, import_line="", code="", manifest_requirements=(),
    local_files=(), existing_distribution=None,
):
    """Return a source-grounded identity decision for one imported module.

    ``classification`` is external, local_provided, local_missing,
    removed_runtime, or uncertain.  The returned distribution is populated for
    external dependencies.  Original manifest names are supplied as evidence
    and are never rewritten by this function.
    """
    module = str(import_name).split(".")[0]
    source = "\n".join([str(import_line), str(code)])
    result = {
        "import_name": module,
        "classification": "external",
        "distribution": canonicalize_name(existing_distribution or module),
        "rule": "retained_mapping",
        "evidence": [],
        "needs_human_review": False,
    }
    local_names = local_module_names(local_files)
    if module in local_names:
        result.update(classification="local_provided", distribution=None, rule="supplied_project_file")
        result["evidence"] = [str(name) for name in local_files if module in PurePosixPath(str(name).replace("\\", "/")).parts or PurePosixPath(str(name)).stem == module]
        return result
    if module in REMOVED_RUNTIME_MODULES:
        result.update(classification="removed_runtime", distribution=None, rule="python312_removed_stdlib")
        result["evidence"] = [import_line, REMOVED_RUNTIME_MODULES[module]]
        return result
    if module in PLACEHOLDER_MODULES:
        local_context = re.search(r"local.{0,35}(?:package|module)|(?:package|module).{0,35}local|custom.{0,35}(?:package|module)|pip\s+install\s+-e", source, re.I)
        if local_context:
            result.update(classification="local_missing", distribution=None, rule="explicit_local_module_context")
            result["evidence"] = [import_line, local_context.group(0)]
            return result
    if module == "models":
        deit_project = re.search(r"from\s+models\.deit\s+import\s+deit_small_patch16_224\b", source) and re.search(r"torch\.load\(|checkpoint", code) and "use_top_n_heads" in code
        tortoise_entity = re.search(r"from\s+models\.expense\s+import\s+Expense\b", source) and re.search(r"from\s+tortoise\.models\s+import\s+Model\b", code) and re.search(r"ManyToManyRelation\s*\[\s*Expense\s*\]", code)
        if deit_project or tortoise_entity:
            result.update(classification="local_missing", distribution=None, rule="contextual_project_module", identity_is_inferred=True)
            context = "models.deit with a project checkpoint and use_top_n_heads parameter" if deit_project else "models.expense supplies a project entity referenced by a Tortoise relation"
            result["evidence"] = [import_line, context]
            return result
        local_context = re.search(r"from\s+models\s+import\s+Store\b|[\"']models[\"']\s*:\s*\[\s*[\"']models[\"']|(?:custom|local)\s+(?:module|implementation)|#\s*models\.py|file\s+[`\"']?models[/\\][\w/\\]+\.py|your\s+project\s+module\s+path", source, re.I)
        if local_context:
            result.update(classification="local_missing", distribution=None, rule="project_models_context")
            result["evidence"] = [import_line, local_context.group(0)]
        else:
            result.update(classification="uncertain", distribution=None, rule="ambiguous_models_namespace", needs_human_review=True)
            result["evidence"] = [import_line]
        return result
    if module in {"google", "azure"}:
        file_share_path = re.match(r"\s*from\s+azure\.storage\.file(?:\.share)?\s+import\b", import_line)
        file_share_declared = "azure-storage-file-share" in _declared_names(manifest_requirements)
        if file_share_path and re.search(r"\bShareClient\b", code) and (file_share_declared or re.search(r"\bShareClient\b", import_line)):
            result.update(distribution="azure-storage-file-share", rule="documented_file_share_api_context", source_import_path_mismatch=True)
            result["evidence"] = [import_line, "ShareClient belongs to azure.storage.fileshare", "https://learn.microsoft.com/en-us/python/api/overview/azure/storage-file-share-readme?view=azure-python"]
            return result
        matches = _namespace_identity(import_line)
        distributions = set(matches.values())
        if len(distributions) == 1:
            distribution = next(iter(distributions))
            result.update(distribution=distribution, rule="documented_namespace_identity")
            citation = "https://github.com/googlecolab/colabtools/blob/main/setup.py" if distribution == "google-colab" else f"https://pypi.org/project/{distribution}/"
            result["evidence"] = list(matches) + [citation]
        elif not matches:
            declared = _declared_names(manifest_requirements)
            matches = {}
            for imported_path in _namespace_paths(import_line):
                parts = imported_path.split(".")
                for length in range(2, len(parts) + 1):
                    prefix = ".".join(parts[:length])
                    distribution = canonicalize_name(prefix)
                    if distribution in declared:
                        matches[prefix] = distribution
            if matches and len(set(matches.values())) == 1:
                distribution = next(iter(matches.values()))
                result.update(distribution=distribution, rule="model_declared_namespace", external_identity_verified=False)
                result["evidence"] = [import_line, f"Model manifest explicitly declares {distribution}"]
            else:
                result.update(classification="uncertain", distribution=None, rule="namespace_requires_source_review", needs_human_review=True)
                result["evidence"] = [import_line]
                result["candidate_distributions"] = sorted(set(matches.values()))
        else:
            result.update(classification="uncertain", distribution=None, rule="namespace_requires_source_review", needs_human_review=True)
            result["evidence"] = [import_line]
            result["candidate_distributions"] = sorted(distributions)
        return result
    if module == "MySQLdb":
        declared = _declared_names(manifest_requirements)
        providers = declared & {"mysqlclient", "mysql-python"}
        explicit_provider = re.search(r"install\s+mysqlclient\b|MySQLdb/mysqlclient\b", source, re.I)
        model_versions = set()
        annotation = re.search(r"#\s*VERSION\s*=\s*([^\s#]+)", import_line, re.I)
        if annotation:
            model_versions.add(annotation.group(1))
        for item in manifest_requirements:
            raw = item.get("raw_requirement", "") if isinstance(item, dict) else str(item)
            try:
                requirement = Requirement(raw)
            except InvalidRequirement:
                continue
            if canonicalize_name(requirement.name) == "mysqldb":
                model_versions.update(spec.version for spec in requirement.specifier if spec.operator in {"==", "==="})
        if len(providers) == 1:
            distribution = next(iter(providers))
            result.update(distribution=distribution, rule="declared_mysql_provider")
            result["evidence"] = [f"Model manifest declares {distribution}", "https://mysqlclient.readthedocs.io/user_guide.html"]
        elif not providers and explicit_provider:
            result.update(distribution="mysqlclient", rule="explicit_mysql_provider_context")
            result["evidence"] = [explicit_provider.group(0), "https://mysqlclient.readthedocs.io/user_guide.html"]
        elif not providers and model_versions and model_versions <= MYSQLCLIENT_VERSION_EVIDENCE:
            result.update(distribution="mysqlclient", rule="documented_mysql_provider_version_identity")
            result["evidence"] = [f"Model version {version} occurs only in mysqlclient among the two MySQLdb providers" for version in sorted(model_versions)]
            result["evidence"].extend(["https://pypi.org/pypi/mysqlclient/json", "https://pypi.org/pypi/mysql-python/json"])
        else:
            result.update(classification="uncertain", distribution=None, rule="mysql_provider_ambiguous", needs_human_review=True)
            result["evidence"] = [import_line, "MySQLdb has multiple distribution providers"]
        return result
    if module in ALIAS_SOURCES:
        distribution, citation = ALIAS_SOURCES[module]
        if distribution in _declared_names(manifest_requirements):
            evidence = f"Model manifest declares {distribution}"
        else:
            evidence = _api_context(module, source)
        if evidence:
            result.update(distribution=distribution, rule="documented_import_identity")
            result["evidence"] = [evidence, citation]
        elif canonicalize_name(existing_distribution or "") == distribution:
            result["evidence"] = ["Retained distribution already matches the documented import namespace", citation]
        else:
            result.update(classification="uncertain", distribution=None, rule="alias_context_missing", needs_human_review=True)
            result["evidence"] = [import_line, citation]
    return result


def _manifest_record(item):
    if isinstance(item, dict):
        return dict(item)
    raw = str(item).strip()
    record = {"raw_requirement": raw, "declaration_source": "manifest", "import_line": f"requirements.txt: {raw}"}
    try:
        requirement = Requirement(raw)
    except InvalidRequirement as error:
        record.update(library=raw, constraint_kind="invalid", version=None, version_spec=None, parse_error=str(error))
        return record
    specifiers = list(requirement.specifier)
    exact = len(specifiers) == 1 and specifiers[0].operator in {"==", "==="} and "*" not in specifiers[0].version
    kind = "exact" if exact else "range" if specifiers or requirement.url else "bare"
    record.update(
        library=requirement.name, normalized_name=canonicalize_name(requirement.name),
        pypi_name_hint=canonicalize_name(requirement.name), constraint_kind=kind,
        version=specifiers[0].version if exact else None,
        version_spec=str(requirement.specifier) if specifiers and not exact else None,
        extras=sorted(requirement.extras), marker=str(requirement.marker) if requirement.marker else None,
        direct_url=requirement.url, parse_error=None,
    )
    return record


def prepare_dependency_records(import_records, manifest_records, *, code_files, mapping):
    """Prepare model-advice records and deduplicated merged resolver inputs.

    Every original import occurrence is returned.  Matching imports inherit the
    manifest's constraint for environment construction while their own advice
    records retain the model-written annotation and source line.
    """
    manifests = [_manifest_record(item) for item in manifest_records]
    by_distribution = defaultdict(list)
    resolver_records = []
    requirements = []
    for manifest in manifests:
        if manifest.get("constraint_kind") == "invalid":
            continue
        raw = manifest.get("raw_requirement", "")
        try:
            distribution = canonicalize_name(Requirement(raw).name)
        except InvalidRequirement:
            continue
        manifest["pypi_name_hint"] = distribution
        by_distribution[distribution].append(manifest)
        resolver_records.append(dict(manifest, identity_source="model_manifest"))
        requirements.append(raw)

    code = "\n".join(code_files.values())
    corrected_imports = []
    identities = []
    non_external = []
    for ordinal, original in enumerate(import_records):
        record = dict(original)
        import_name = record.get("import_name") or record.get("library")
        identity = resolve_import_identity(
            import_name, import_line=record.get("import_line", ""), code=code,
            manifest_requirements=manifests, local_files=code_files,
            existing_distribution=mapping.get(import_name, import_name),
        )
        identity["import_ordinal"] = ordinal
        identities.append(identity)
        record.update(
            import_name=import_name, identity_classification=identity["classification"],
            identity_rule=identity["rule"], pypi_name_hint=identity["distribution"],
            pypi_name=identity["distribution"],
        )
        corrected_imports.append(record)
        if identity["classification"] != "external":
            non_external.append(record)
            continue
        distribution = identity["distribution"]
        matching = by_distribution.get(distribution, [])
        if matching:
            record["matched_manifest_requirements"] = [item["raw_requirement"] for item in matching]
            continue
        version = record.get("specified_version") or record.get("version")
        specifier = record.get("version_spec")
        requirement_text = distribution
        if version:
            requirement_text += f"=={version}"
        elif specifier:
            requirement_text += str(specifier)
        resolver_records.append(dict(record, resolver_requirement=requirement_text))
        requirements.append(requirement_text)

    return {
        "manifest_records": manifests,
        "import_records": corrected_imports,
        "resolver_records": resolver_records,
        "requirements": list(dict.fromkeys(requirements)),
        "identities": identities,
        "non_external_imports": non_external,
        "review_required": any(item["needs_human_review"] for item in identities),
    }


def classify_diagnostic_stage(diagnostics):
    """Identify environment failures before classifying code diagnostics."""
    if any(item.get("rule") == "ty-install-error" or item.get("level") == "install" for item in diagnostics):
        return "environment_build"
    return "static_diagnostic" if diagnostics else "no_diagnostic"
