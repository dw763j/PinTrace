import os
"""Generate the fixed-code comparison table from checked response counts."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(os.environ.get("PINTRACE_WORKSPACE", Path(__file__).resolve().parents[1]))


def build_table(root=ROOT):
    source = root / "experiments/fixed_code/outcomes.csv"
    with source.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {(r["track"], r["strategy"]): r for r in rows}
    assert len(indexed) == 6
    lines = [
        "% Evidence: supplement/fixed_code/outcomes.csv; scripts/prepare_fixed_code_table.py. All numeric cells count saved responses.",
        r"\begin{table}[htbp]",
        r"\centering\footnotesize",
        r"\caption{\revised{Fixed-code policy outcomes on the controlled samples.",
        r"Cells give response counts.",
        r"Recovered and Regressed count transitions from nonpass to pass and from pass to nonpass, respectively.",
        r"Pass + no match counts passes whose complete dependency graphs have no advisory match under the retained snapshot.",
        r"Dashes identify the baseline reference.}}",
        r"\label{tab:fixed_code_comparison}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xrrrrr@{}}",
        r"\toprule",
        r"\revised{\textbf{Policy}} & \revised{\textbf{Installed}} & \revised{\textbf{Pass}} & \revised{\textbf{Recovered}} & \revised{\textbf{Regressed}} & \revised{\textbf{Pass + no match}} \\",
    ]
    names = {"original": "Original constraints", "unpin": "Remove exact constraints", "neighbor": "Verified neighbor"}
    for track, dataset, oracle in (("d2", r"\myso", "static validation"), ("d1", "BigCodeBench", "task tests")):
        baseline = indexed[track, "original"]
        total, passing = int(baseline["N"]), int(baseline["oracle_pass"])
        failing = total - passing
        lines.append(r"\midrule")
        heading = f"{dataset}: {oracle}, $N={total}$ ({passing} baseline passes, {failing} nonpasses)"
        lines.append(r"\multicolumn{6}{@{}l}{\revised{" + heading + r"}} \\")
        for strategy in ("original", "unpin", "neighbor"):
            row = indexed[track, strategy]
            assert int(row["N"]) == total
            assert int(row["oracle_pass"]) == passing + int(row["recovered"]) - int(row["regressed"])
            cells = [names[strategy], row["installed"], row["oracle_pass"],
                     "--" if strategy == "original" else row["recovered"],
                     "--" if strategy == "original" else row["regressed"], row["oracle_pass_without_association"]]
            lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    output = root / "tables/fixed_code_policies.tex"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {"source": str(source.relative_to(root)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "output": str(output.relative_to(root)), "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "unit": "Saved response", "rows": rows, "block_denominators": {"PinTrace": 80, "BigCodeBench": 40}}
    (root / "analysis/fixed_code_policy_table_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(build_table())
