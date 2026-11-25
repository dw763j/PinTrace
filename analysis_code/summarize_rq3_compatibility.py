import os
"""Summarize compatibility diagnostics in installed PinTrace environments."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from summarize_mr_constraint_profiles import DISPLAY_NAMES


def summarize_compatibility(root):
    root = Path(root)
    analysis, tables = root / "analysis", root / "tables"
    sources = [analysis / "response_flags.csv", analysis / "paired_cluster_bootstrap.csv"]
    with sources[0].open(encoding="utf-8-sig", newline="") as handle:
        responses = [r for r in csv.DictReader(handle) if r["track"] == "d2"]
    with sources[1].open(encoding="utf-8-sig", newline="") as handle:
        contrasts = list(csv.DictReader(handle))
    assert len(responses) == 20_000
    models = list(DISPLAY_NAMES) + ["All observations"]
    records = []
    for model in models:
        for mode, condition in [("requirements.txt", "MR"), ("inline", "VA")]:
            selected = [r for r in responses if r["mode"] == mode and (model == "All observations" or r["model"] == model)]
            assert len(selected) == (10000 if model == "All observations" else 1000)
            installed = [r for r in selected if r["installation"] == "True"]
            diagnosed = [r for r in installed if r["static"] != "True"]
            assert all(r["static_executed"] == "True" and r["static_status"] == "diagnostic" for r in diagnosed)
            classes = Counter(r["compatibility_layer"] for r in diagnosed)
            assert set(classes) <= {"import_or_initialization", "api_mismatch_candidate", "code_or_syntax"}
            library = classes["import_or_initialization"] + classes["api_mismatch_candidate"]
            records.append(dict(model=model, condition=condition, N=len(selected), installed=len(installed),
                                static_pass=len(installed)-len(diagnosed), diagnostic_responses=len(diagnosed),
                                diagnostic_pct_among_installed=100*len(diagnosed)/len(installed),
                                import_candidates=classes["import_or_initialization"], api_candidates=classes["api_mismatch_candidate"],
                                other_code_diagnostics=classes["code_or_syntax"], library_candidates=library,
                                library_pct_among_installed=100*library/len(installed),
                                library_pct_among_diagnosed=100*library/len(diagnosed)))
    assert all(r["library_pct_among_installed"] >= 20 for r in records if r["model"] != "All observations")
    totals = {r["condition"]: r for r in records if r["model"] == "All observations"}
    for condition in ["MR", "VA"]:
        for key in ["N", "installed", "static_pass", "diagnostic_responses", "import_candidates", "api_candidates", "other_code_diagnostics", "library_candidates"]:
            assert totals[condition][key] == sum(r[key] for r in records if r["condition"] == condition and r["model"] != "All observations")
    with (analysis / "rq3_installed_compatibility.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    def markdown(name, headers, data, note):
        contents = note+"\n\n| "+" | ".join(headers)+" |\n| "+" | ".join(["---"]*len(headers))+" |\n"
        contents += "".join("| "+" | ".join(map(str, row))+" |\n" for row in data)
        (tables / f"{name}.md").write_text(contents, encoding="utf-8")

    stage_rows = []
    for metric, label in [("input", "Entered joint resolution"), ("lock", "Complete lock"),
                          ("installation", "Environment installed"), ("static", "Installation and static validation")]:
        row = next(r for r in contrasts if r["population"] == "PinTrace complete" and r["metric"] == metric and r["model"] == "All observations")
        mr = sum(r[metric] == "True" and r["mode"] == "requirements.txt" for r in responses)
        va = sum(r[metric] == "True" and r["mode"] == "inline" for r in responses)
        assert abs(float(row["MR_pct"])-mr/100) < 1e-9 and abs(float(row["VA_pct"])-va/100) < 1e-9
        stage_rows.append([label, f"{mr:,} ({mr/100:.2f}%)", f"{va:,} ({va/100:.2f}%)"]+
                          [f"{float(row[k]):.2f}" for k in ["delta_VA_minus_MR_pp", "delta_ci_low", "delta_ci_high"]])
    markdown("rq3_outcomes", ["Stage", "MR", "VA", "VA minus MR (pp)", "95% lower", "95% upper"], stage_rows,
             "Complete PinTrace; 10,000 responses per condition. All stages retain the full denominator. Confidence limits refer to paired VA-minus-MR differences.")
    rows = []
    for model in models:
        values = {r["condition"]: r for r in records if r["model"] == model}
        row = [DISPLAY_NAMES.get(model, model)]
        for condition in ["MR", "VA"]:
            r = values[condition]
            row += [f"{r['installed']:,}",
                    f"{r['diagnostic_responses']:,} ({r['diagnostic_pct_among_installed']:.2f})",
                    f"{r['library_candidates']:,} ({r['library_pct_among_installed']:.2f})"]
        rows.append(row)
    markdown("rq3_installed_diagnostics", ["Model", "MR installed", "MR any n (%)", "MR library n (%)", "VA installed", "VA any n (%)", "VA library n (%)"], rows,
             "Complete PinTrace. Any counts responses with a static diagnostic. Library is the subset assigned to import-diagnostic or API-diagnostic candidate categories. Both diagnostic rates use installed responses within each model and condition. All observations sums counts before division.")

    stage_tex = r"""\begin{table}[htbp]
\centering\footnotesize
\caption{\revised{Environment construction and compatibility assessment in complete \myso.
Cells give response counts and percentages using $N_q=10,000$ per condition at every stage.
The final row records a pass in $H_i[\mathsf{static}]$.
Differences $\widehat{\Delta}_{\theta}$ are VA minus MR in percentage points, with task-clustered 95\% confidence limits.}}
\label{tab:rq3_outcomes}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabularx}{\linewidth}{@{}>{\hsize=1.3\hsize\linewidth=\hsize\raggedright\arraybackslash}X>{\hsize=.85\hsize\linewidth=\hsize\centering\arraybackslash}X>{\hsize=.85\hsize\linewidth=\hsize\centering\arraybackslash}Xrrr@{}}
\toprule
\revised{\textbf{Stage}} & \revised{\textbf{MR}} & \revised{\textbf{VA}} & \revised{$\widehat{\Delta}_{\theta}$} & \revised{\textbf{Lower}} & \revised{\textbf{Upper}} \\
\midrule
"""
    stage_tex += "".join(" & ".join(row).replace("%", r"\%")+r" \\"+"\n" for row in stage_rows)
    stage_tex += "\\bottomrule\n\\end{tabularx}\n\\end{table}\n"
    (tables / "rq3_outcomes.tex").write_text(stage_tex, encoding="utf-8")

    diagnosed_tex = r"""\begin{table}[htbp]
\centering\footnotesize
\caption{\revised{Static diagnostics in installed \myso environments.
Installed counts responses with successful installation.
Any counts responses with a static diagnostic, while Library gives the subset classified as import-diagnostic or API-diagnostic candidates.
Both diagnostic columns report $n$ (\%) using Installed as the denominator.
All observations sums counts across models before division.}}
\label{tab:rq3_installed_diagnostics}
\setlength{\tabcolsep}{2.4pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xrrrrrr@{}}
\toprule
\revised{\textbf{Model}} & \multicolumn{3}{c}{\revised{\textbf{MR}}} & \multicolumn{3}{c}{\revised{\textbf{VA}}} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
& \revised{\textbf{Installed}} & \revised{\textbf{Any}} & \revised{\textbf{Library}} & \revised{\textbf{Installed}} & \revised{\textbf{Any}} & \revised{\textbf{Library}} \\
\midrule
"""
    for row in rows:
        if row[0] == "All observations":
            diagnosed_tex += "\\midrule\n"
        diagnosed_tex += " & ".join(row).replace("%", r"\%")+r" \\"+"\n"
    diagnosed_tex += "\\bottomrule\n\\end{tabularx}\n\\end{table}\n"
    (tables / "rq3_installed_diagnostics.tex").write_text(diagnosed_tex, encoding="utf-8")

    supporting = []
    for pop, metric in [("PinTrace clean", "static"), ("BigCodeBench", "dynamic")]:
        r = next(r for r in contrasts if r["population"] == pop and r["metric"] == metric and r["model"] == "All observations")
        supporting.append([pop, metric, r["observations_per_condition"]]+[f"{float(r[k]):.2f}" for k in ["MR_pct", "VA_pct", "delta_VA_minus_MR_pp", "delta_ci_low", "delta_ci_high"]])
    markdown("rq3_supporting_outcomes", ["Population", "Outcome", "N per condition", "MR (%)", "VA (%)", "VA minus MR (pp)", "95% lower", "95% upper"], supporting,
             "Additional response populations, with their original denominators. BigCodeBench test outcomes are reported in the task-execution analysis.")
    static_row = next(r for r in contrasts if r["population"] == "PinTrace complete" and r["metric"] == "static" and r["model"] == "All observations")
    manifest = dict(sources_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    installed_records=records, stages=stage_rows,
                    paired_static_counts={k: int(static_row[k]) for k in ["both_success", "MR_only", "VA_only"]},
                    supporting_outcomes=supporting,
                    unit="saved response", diagnostic_scope="Any includes all recorded static-diagnostic responses; Library includes import or API diagnostic candidates")
    (tables / "rq3_compatibility_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = summarize_compatibility(Path(os.environ.get("PINTRACE_WORKSPACE", Path(__file__).resolve().parents[1])))
    print(f"Reproduced {len(result['installed_records'])} model-condition rows and four end-to-end stages.")
