import os
"""Reproduce the response-level and declaration-level RQ2 table data."""
import csv
import hashlib
import json
from pathlib import Path
from prepare_advisory_model_table import write_advisory_model_table


def prepare_rq2(root):
    root = Path(root)
    analysis, output = root / "analysis", root / "tables"
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}

    def read(name):
        source = analysis / name
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        with source.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def write(name, headers, data, description):
        text = description + "\n\n| " + " | ".join(headers) + " |\n"
        text += "| " + " | ".join(["---"]*len(headers)) + " |\n"
        text += "".join("| " + " | ".join(map(str, row)) + " |\n" for row in data)
        (output / f"{name}.md").write_text(text, encoding="utf-8")

    all_funnels = read("security_funnels.csv")
    funnel = [r for r in all_funnels if r["model"] == "All observations"]
    complete = [r for r in read("all_constraint_validity.csv") if r["model"] == "All observations"]
    criteria, security = [], []
    for mode, kind, condition, form in [("requirements.txt", "exact", "MR", "Exact"),
                                         ("requirements.txt", "range", "MR", "Ranged"),
                                         ("inline", "exact", "VA", "Exact")]:
        r = next(r for r in funnel if r["mode"] == mode and r["kind"] == kind and r["security_scope"] == "published_by_cutoff")
        n, s, p, v = [int(r[key]) for key in ["N", "S", "P", "V"]]
        assert n == 9270 and 0 <= v <= p <= s <= n
        criteria.append(["At least one", condition, form, s, p, round(100*p/s, 2)])
        security.append([condition, form, p, v, round(100*v/n, 2), round(100*v/p, 2)])
    for mode, condition in [("requirements.txt", "MR"), ("inline", "VA")]:
        r = next(r for r in complete if r["mode"] == mode)
        s, a = int(r["S_assessable"]), int(r["A"])
        criteria.append(["Every declaration", condition, "All non-bare", s, a, round(100*a/s, 2)])
    def format_row(row):
        return [f"{cell:,}" if isinstance(cell, int) else f"{cell:.2f}" if isinstance(cell, float) else cell for cell in row]
    write("rq2_validity", ["Criterion", "Condition", "Form", "Assessed", "Passing", "Rate (%)"],
          [format_row(r) for r in criteria],
          "Clean PinTrace; 9,270 responses per condition. Criteria use different eligible response populations. Every-declaration validity excludes two VA responses with unresolved identities.")
    model_security = write_advisory_model_table(root, all_funnels)
    releases = read("release_existence_temporal.csv")
    states = [("absent_release", "Release absent from retained history"),
              ("metadata_unavailable", "Package metadata unavailable"),
              ("published_by_cutoff", "Published by cutoff"),
              ("after_cutoff", "First retained upload after cutoff"),
              ("release_without_dated_artifact", "Release without dated artifact")]
    release_rows = []
    for state, label in states:
        counts = [sum(int(r["occurrences"]) for r in releases if r["mode"] == mode and r["state"] == state)
                  for mode in ["requirements.txt", "inline"]]
        release_rows.append([label]+counts)
        if state == "published_by_cutoff":
            failed = [sum(int(r["invalid_occurrences"]) for r in releases if r["mode"] == mode and r["state"] == state)
                      for mode in ["requirements.txt", "inline"]]
            release_rows.extend([["  Individually satisfiable"]+[n-f for n, f in zip(counts, failed)],
                                 ["  Individually unsatisfiable"]+failed])
    write("rq2_release_states", ["Publication state", "MR exact", "VA exact"],
          [format_row(r) for r in release_rows],
          "Active identified exact-declaration occurrences in clean PinTrace. The two indented rows partition Published by cutoff.")
    weights = read("package_standardization.csv")
    weighting = []
    for mode, condition in [("requirements.txt", "MR"), ("inline", "VA")]:
        for metric, scope in [("exact_published_by_cutoff", "Cutoff"), ("exact_associated", "Snapshot")]:
            r = next(r for r in weights if r["mode"] == mode and r["metric"] == metric)
            weighting.append([condition, scope]+[round(float(r[key]), 2) for key in [
                "unweighted_response_pct", "equal_target_package_pct", "candidate_incidence_standardized_pct"]])
    write("rq2_weighting", ["Condition", "Scope", "Observed", "Equal-package", "Candidate-weighted"],
          [format_row(r) for r in weighting],
          "Overall exact-advice association rates (%). Package means cover 261 target packages. Candidate weights use target incidence in 55,711 pre-balanced records.")
    data = dict(sources_sha256=hashes, unit_by_table={"validity": "response", "security": "response",
                "release_states": "declaration occurrence", "weighting": "package-weighted response rate"},
                validity=criteria, security=model_security["rows"], security_aggregate=security,
                release_states=release_rows, weighting=weighting)
    (output / "rq2_display_data.json").write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    return data


if __name__ == "__main__":
    prepare_rq2(Path(os.environ.get("PINTRACE_WORKSPACE", Path(__file__).resolve().parents[1])))
    print("Reproduced four RQ2 tables from corrected analysis outputs.")
