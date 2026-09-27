from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

from .experiments import config_fingerprint

REQUIRED_METHODS = {"baseline", "fixed_update_budget_matched_baseline", "fixed_incremental",
                    "dynamic_all_classes", "dynamic_incremental"}


def _name(value):
    return str(value).replace("_", r"\_")


def _method(value):
    return {"baseline": "Ordinary baseline",
            "fixed_update_budget_matched_baseline": "Fixed update-budget control",
            "fixed_incremental": "Fixed incremental",
            "dynamic_all_classes": "Dynamic all-classes",
            "dynamic_incremental": "Dynamic incremental"}[value]


def _comparison(value):
    first, second = value.split(" - ")
    return f"{_method(first)} minus {_method(second)}"


def _family(value):
    return {"primary": "Primary", "exploratory_minority": "Minority exploratory"}[value]


def _dataset(value):
    return {"dry_bean": "Dry Bean", "steel_plates": "Steel Plates Faults"}.get(value, value)


def validate_submission_metadata(metadata):
    required = ("author", "student_number", "institution", "module_code", "repository_url")
    missing = [key for key in required if not metadata.get(key) or "PLACEHOLDER" in metadata[key]]
    if missing:
        raise ValueError(f"Submission metadata still contains placeholders: {', '.join(missing)}")
    if not re.fullmatch(r"https://[^\s]+", metadata["repository_url"]):
        raise ValueError("repository_url must be a usable HTTPS URL")
    return f"{metadata['student_number']}RW{metadata['module_code']}assignment3.pdf"


def validate_result_provenance(dataset_results, manifests, config, boundary_statuses):
    if set(dataset_results) != set(manifests):
        raise ValueError("Every dataset summary requires a matching manifest")
    expected = config_fingerprint(config)
    for dataset, manifest in manifests.items():
        if manifest.get("configuration_fingerprint") != expected:
            raise ValueError(f"Stale or mismatched results for {dataset}")
        if set(manifest.get("methods", [])) != REQUIRED_METHODS:
            raise ValueError(f"Incorrect method set for {dataset}")
    unresolved = [row["dataset"] for row in boundary_statuses if not row["resolved"]]
    if unresolved:
        raise ValueError("Architecture search remains at its upper boundary for: " + ", ".join(unresolved))


def export_generated_results(dataset_results, path="report/canonical_results.tex"):
    """Export numerical tables only; report structure remains manually authored."""
    lines = [r"% Generated from canonical notebook outputs. Do not edit manually."]
    for dataset, result in dataset_results.items():
        main, _, stats, efficiency = result
        label = dataset.replace("_", "-")
        macro = main[main.metric == "macro_f1"].set_index("method")
        minority = main[main.metric.isin(["minority_macro_f1", "minority_recall"])]
        lines += [r"\begin{table}[t]", rf"\caption{{Overall test performance: {_dataset(dataset)}}}",
            rf"\label{{tab:{label}-overall}}", r"\centering", r"\begin{tabular}{lcc}",
            r"\toprule Method & Macro-F1 & Fold std. dev.\\", r"\midrule"]
        for method, row in macro.iterrows():
            lines.append(rf"{_name(_method(method))} & {row['mean']:.4f} & {row['between_fold_std']:.4f}\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", r"\begin{table}[t]",
            rf"\caption{{Minority performance: {_dataset(dataset)}}}",
            rf"\label{{tab:{label}-minority}}", r"\centering", r"\begin{tabular}{lcc}",
            r"\toprule Method & Macro-F1 & Recall\\", r"\midrule"]
        for method in macro.index:
            part = minority[minority.method == method].set_index("metric")
            lines.append(rf"{_name(_method(method))} & {part.loc['minority_macro_f1','mean']:.4f} & "
                         rf"{part.loc['minority_recall','mean']:.4f}\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", r"\begin{table*}[t]",
            rf"\caption{{Paired fold-level comparisons: {_dataset(dataset)}}}",
            rf"\label{{tab:{label}-statistics}}", r"\centering\small", r"\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrr}",
            r"\toprule Family & Comparison & Mean diff. & 95\% interval & $d_z$ & Raw $p$ & Holm $p$ & $n$\\", r"\midrule"]
        for row in stats.itertuples():
            comparison = _name(_comparison(row.comparison))
            lines.append(rf"{_name(_family(row.family))} & {comparison} & {row.mean_difference:+.4f} & "
                         rf"[{row.ci95_low:.4f}, {row.ci95_high:.4f}] & {row.paired_effect_dz:.3f} & "
                         rf"{row.p_value:.4f} & {row.p_holm:.4f} & {row.n_folds}\\")
        lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", r"\begin{table*}[t]",
            rf"\caption{{Training cost: {_dataset(dataset)}}}",
            rf"\label{{tab:{label}-efficiency}}", r"\centering\small", r"\resizebox{\textwidth}{!}{\begin{tabular}{lrrrrrr}",
            r"\toprule Method & Mean updates & Mean examples processed & Mean final hidden & Accepted & Rejected & Mean time (s)\\", r"\midrule"]
        for row in efficiency.itertuples():
            lines.append(rf"{_name(_method(row.method))} & {row.mean_updates:.0f} & {row.mean_examples:.0f} & "
                         rf"{row.mean_final_hidden_units:.1f} & {row.mean_accepted_probes:.1f} & "
                         rf"{row.mean_rejected_probes:.1f} & {row.mean_elapsed_seconds:.1f}\\")
        lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    grouped = {kind: [] for kind in ("overall", "minority", "statistics", "efficiency")}
    block = []
    for line in lines[1:]:
        if line.startswith(r"\begin{table"):
            block = [line]
        else:
            block.append(line)
        if line.startswith(r"\end{table"):
            label = next(item for item in block if item.startswith(r"\label{tab:"))
            kind = next(kind for kind in grouped if label.endswith(f"-{kind}}}"))
            grouped[kind].extend(block)
            block = []
    for kind, contents in grouped.items():
        (path.parent / f"canonical_{kind}.tex").write_text("\n".join(contents) + "\n")


def load_manifest(path):
    return json.loads(Path(path).read_text())


def compile_report(report_dir="report", metadata=None, output_dir=None):
    filename = validate_submission_metadata(metadata or {})
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                   cwd=report_dir, check=True)
    destination = Path(output_dir or report_dir) / filename
    destination.write_bytes((Path(report_dir) / "report.pdf").read_bytes())
    return destination
