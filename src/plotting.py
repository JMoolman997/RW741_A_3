from __future__ import annotations

from pathlib import Path
from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t
from sklearn.metrics import confusion_matrix

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": .22,
    "legend.frameon": False,
})

METHOD_LABELS = {
    "baseline": "Ordinary baseline",
    "fixed_update_budget_matched_baseline": "Fixed update control",
    "fixed_incremental": "Fixed incremental",
    "dynamic_all_classes": "Dynamic all-classes",
    "dynamic_incremental": "Dynamic incremental",
}
METHOD_ORDER = list(METHOD_LABELS)
METHOD_COLORS = dict(zip(METHOD_ORDER, sns.color_palette("colorblind", len(METHOD_ORDER))))
COMPARISONS = (("fixed_incremental", "fixed_update_budget_matched_baseline", "Fixed comparison"),
               ("dynamic_incremental", "dynamic_all_classes", "Dynamic comparison"))
DATASET_LABELS = {"dry_bean": "Dry Bean", "steel_plates": "Steel Plates Faults"}


def _save(fig, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_class_distribution(y, dataset_name, path="report/figures/class_distribution.pdf"):
    counts = y.value_counts().sort_values()
    counts.index = [str(value).replace("_", " ") for value in counts.index]
    ratio = counts.max() / counts.min()
    colors = ["#CC6677" if i < 2 else "#4477AA" for i in range(len(counts))]
    fig, ax = plt.subplots(figsize=(5.2, 3.4)); counts.plot.barh(ax=ax, color=colors)
    ax.set(xlabel="Instances", ylabel="Class", title=f"{dataset_name} class distribution")
    ax.text(.98, .03, f"Imbalance ratio {ratio:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8)
    ax.set_xlim(0, counts.max() * 1.27)
    ax.grid(axis="y", visible=False)
    for patch, value in zip(ax.patches, counts):
        ax.text(value, patch.get_y() + patch.get_height()/2,
                f" {value:,} ({value / counts.sum():.1%})", va="center", fontsize=8)
    _save(fig, path); return fig


def plot_architecture_search(summary, selected, path="report/figures/baseline_architecture.pdf"):
    folds = sorted(summary.outer_fold.unique())
    fig, axes = plt.subplots(1, len(folds), figsize=(8.0, 2.5), sharey=True)
    for ax, fold in zip(np.atleast_1d(axes), folds):
        group = summary[summary.outer_fold == fold]
        group = group.sort_values("hidden_units")
        ax.plot(group.hidden_units, group["mean"], marker="o", markersize=2.5, linewidth=1,
                color="#4477AA", label=f"Fold {fold} ($h={selected[(group.dataset.iloc[0], int(fold))]}$)")
        chosen = selected[(group.dataset.iloc[0], int(fold))]
        point = group[group.hidden_units == chosen]
        boundary = chosen == group.hidden_units.max()
        ax.scatter(point.hidden_units, point["mean"], marker="^" if boundary else "s",
                   color="#CC6677" if boundary else "black", s=38, zorder=4)
        ax.set_title(f"Fold {fold}\nselected {chosen}" + (" (boundary)" if boundary else ""), fontsize=8)
        ax.set_xscale("symlog", linthresh=1, base=2)
        ax.set_xticks(group.hidden_units[::2], [str(v) for v in group.hidden_units[::2]], rotation=60)
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("Hidden units", fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("Validation Macro-F1")
    _save(fig, path); return fig


def plot_method_comparison(methods, path="report/figures/method_comparison.pdf"):
    comparisons = [("fixed_incremental", "fixed_update_budget_matched_baseline"),
                   ("dynamic_incremental", "dynamic_all_classes")]
    rows = []
    for first, second in comparisons:
        a = pd.DataFrame(methods[first]).groupby("outer_fold", as_index=False).macro_f1.mean()
        b = pd.DataFrame(methods[second]).groupby("outer_fold", as_index=False).macro_f1.mean()
        paired = a.merge(b, on="outer_fold", suffixes=("_first", "_second"), validate="one_to_one")
        for row in paired.itertuples():
            label = "Fixed incremental $-$ update control" if first == "fixed_incremental" else \
                    "Dynamic incremental $-$ all-classes"
            rows.append({"comparison": label, "outer_fold": row.outer_fold,
                         "difference": row.macro_f1_first-row.macro_f1_second})
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    sns.stripplot(data=data, x="comparison", y="difference", hue="outer_fold",
                  palette="colorblind", size=6, ax=ax)
    grouped = data.groupby("comparison", sort=False).difference
    means = grouped.mean()
    half_width = grouped.std() * t.ppf(.975, grouped.count() - 1) / np.sqrt(grouped.count())
    ax.errorbar(range(len(means)), means, yerr=half_width, fmt="D", markersize=6,
                color="black", capsize=5, linewidth=1.4, label="Mean and 95% CI", zorder=5)
    ax.axhline(0, color="black", linewidth=.9); ax.set(ylabel=r"Paired fold-mean $\Delta$ Macro-F1",
        xlabel="", title="Primary curriculum comparisons"); ax.tick_params(axis="x", rotation=8)
    ax.legend(title="Outer fold", ncol=3, fontsize=7, title_fontsize=7)
    _save(fig, path); return fig


def plot_primary_effects(all_methods, path="report/figures/primary_effects.pdf"):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.1))
    for ax, (dataset, methods) in zip(axes, all_methods.items()):
        ax.axvline(0, color="black", linewidth=1)
        extent = 0
        for row, (first, second, label) in enumerate(COMPARISONS):
            first_fold = pd.DataFrame(methods[first]).groupby("outer_fold").macro_f1.mean()
            second_fold = pd.DataFrame(methods[second]).groupby("outer_fold").macro_f1.mean()
            differences = first_fold.subtract(second_fold).dropna().to_numpy()
            mean = differences.mean()
            half_width = t.ppf(.975, len(differences)-1) * differences.std(ddof=1) / np.sqrt(len(differences))
            extent = max(extent, np.max(np.abs(differences)), abs(mean)+half_width)
            y = 1-row
            ax.scatter(differences, y + np.linspace(-.10, .10, len(differences)), s=18,
                       color=METHOD_COLORS[first], alpha=.75, zorder=3)
            ax.errorbar(mean, y, xerr=half_width, fmt="D", markersize=6,
                        color="black", capsize=4, linewidth=1.5, zorder=4)
        ax.set_yticks([1, 0], ["Fixed − update control", "Dynamic − all-classes"])
        ax.set_ylim(-.35, 1.35)
        ax.set_xlim(-extent*1.12, extent*1.12)
        ax.set_title(DATASET_LABELS[dataset], fontsize=10)
        ax.set_xlabel("Paired fold mean difference in test Macro-F1")
    _save(fig, path); return fig


def plot_per_class_effects(all_methods, class_names, class_order, path="report/figures/per_class_effects.pdf"):
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2))
    for row, (dataset, methods) in enumerate(all_methods.items()):
        names = class_names[dataset]
        order = class_order[dataset]
        for col, (first, second, label) in enumerate(COMPARISONS):
            ax = axes[row, col]
            ax.axvline(0, color="black", linewidth=1)
            def fold_values(method):
                frame = pd.DataFrame(methods[method])[["outer_fold", "f1"]].copy()
                frame["f1"] = frame.f1.map(lambda value: literal_eval(value) if isinstance(value, str) else value)
                expanded = pd.DataFrame(frame.f1.tolist(), columns=names)
                expanded["outer_fold"] = frame.outer_fold.to_numpy()
                return expanded.groupby("outer_fold")[names].mean()
            differences = fold_values(first).subtract(fold_values(second))[order]
            for y, name in enumerate(order):
                values = differences[name].to_numpy()
                ax.scatter(values, y + np.linspace(-.12, .12, len(values)), s=10,
                           alpha=.42, color=METHOD_COLORS[first])
                ax.scatter(values.mean(), y, s=38, marker="D", color=METHOD_COLORS[first], zorder=3)
            ax.set_yticks(range(len(order)), [name.replace("_", " ") for name in order])
            ax.invert_yaxis()
            ax.set_title(f"{DATASET_LABELS[dataset]}: {label}", fontsize=9)
            ax.set_xlabel("Paired fold mean ΔF1")
    _save(fig, path); return fig


def plot_performance_cost(efficiency, path="report/figures/performance_cost.pdf"):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    for ax, (dataset, frame) in zip(axes, efficiency.items()):
        frame = frame.set_index("method").loc[METHOD_ORDER]
        for method, result in frame.iterrows():
            x = np.log10(result.mean_updates)
            ax.scatter(x, result.macro_f1, s=65, color=METHOD_COLORS[method],
                       marker="s" if method.startswith("fixed") else "o", zorder=3)
            ax.annotate(f"{METHOD_LABELS[method]} ({result.mean_final_hidden_units:.1f} h)",
                        (x, result.macro_f1), xytext=(4, 4),
                        textcoords="offset points", fontsize=7)
        values = np.log10(frame.mean_updates)
        ax.set_xlim(values.min()-.08, values.max()+.35)
        ax.set_title(DATASET_LABELS[dataset])
        ax.set(xlabel=r"$\log_{10}$(mean gradient updates)", ylabel="Mean test Macro-F1")
    _save(fig, path); return fig


def plot_dynamic_capacity_performance(methods, path="report/figures/dynamic_capacity_performance.pdf"):
    frames = []
    for method in ("dynamic_all_classes", "dynamic_incremental"):
        frame = pd.DataFrame(methods[method])[["final_hidden_units", "macro_f1"]].copy()
        frame["method"] = METHOD_LABELS[method]
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    rng = np.random.default_rng(42)
    for method in ("dynamic_all_classes", "dynamic_incremental"):
        part = data[data.method == METHOD_LABELS[method]]
        x = part.final_hidden_units.to_numpy() + rng.uniform(-.12, .12, len(part))
        ax.scatter(x, part.macro_f1, color=METHOD_COLORS[method], alpha=.65,
                   marker="o" if method == "dynamic_all_classes" else "s",
                   s=24, label=METHOD_LABELS[method])
        ax.scatter(part.final_hidden_units.mean(), part.macro_f1.mean(),
                   color=METHOD_COLORS[method], marker="D", s=90, edgecolor="black", zorder=4)
    ax.legend(fontsize=8)
    ax.set_xticks(range(int(data.final_hidden_units.max())+1))
    ax.set(xlabel="Final hidden units", ylabel="Test Macro-F1",
           title="Dynamic capacity and test performance")
    _save(fig, path); return fig


def plot_per_class(per_class, path="report/figures/per_class_f1.pdf", ylim=(0, 1.03)):
    data = per_class[per_class.metric == "f1"].copy()
    data["class"] = data["class"].astype(str).str.replace("_", " ", regex=False)
    methods = [m for m in METHOD_ORDER if m in set(data.method)]
    classes = list(data["class"].unique())
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    offsets = np.linspace(-.24, .24, len(methods))
    for offset, method in zip(offsets, methods):
        part = data[data.method == method].set_index("class").loc[classes]
        x = np.arange(len(classes)) + offset
        ax.errorbar(x, part["mean"], yerr=part["between_fold_std"], marker="o", markersize=3.5,
                    linestyle="none", capsize=2, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    ax.set_xticks(np.arange(len(classes)), classes, rotation=24, ha="right")
    ax.set(ylabel=r"Mean test F1 $\pm$ fold SD", xlabel="Class",
           title="Per-class test performance across outer folds", ylim=ylim)
    ax.legend(fontsize=7, ncol=2)
    _save(fig, path); return fig


def plot_minority_performance(main, path="report/figures/minority_performance.pdf"):
    data = main[main.metric.isin(["minority_macro_f1", "minority_recall"])].copy()
    labels = {"minority_macro_f1": "Minority Macro-F1", "minority_recall": "Minority recall"}
    data["metric"] = data.metric.map(labels)
    data["method_label"] = data.method.map(METHOD_LABELS)
    order = [METHOD_LABELS[m] for m in METHOD_ORDER if m in set(data.method)]
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    offsets = {"Minority Macro-F1": -.13, "Minority recall": .13}
    colors = {"Minority Macro-F1": "#4477AA", "Minority recall": "#CC6677"}
    for metric in labels.values():
        part = data[data.metric == metric].set_index("method_label").loc[order]
        ax.errorbar(np.arange(len(order)) + offsets[metric], part["mean"],
                    yerr=part["between_fold_std"], fmt="o", capsize=3,
                    color=colors[metric], label=metric)
    ax.set_xticks(np.arange(len(order)), order, rotation=20, ha="right")
    values = data["mean"] - data["between_fold_std"]
    ax.set(xlabel="", ylabel="Mean test score ± fold SD",
           title="Performance on the two minority classes",
           ylim=(max(0, np.floor(values.min() * 20) / 20), 1.01))
    ax.legend(fontsize=8, title="", loc="lower left")
    _save(fig, path); return fig


def plot_sensitivity(frame, path="report/figures/capacity_sensitivity.pdf"):
    summary = frame.groupby(["capacity_probe_epochs", "capacity_min_gain"], as_index=False)[
        "final_validation_macro_f1"].mean()
    table = summary.pivot(index="capacity_probe_epochs", columns="capacity_min_gain",
                          values="final_validation_macro_f1")
    canonical = table.loc[15, .001]
    change = table - canonical
    limit = max(abs(change.min().min()), abs(change.max().max()))
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    sns.heatmap(change, annot=table, fmt=".4f", cmap="viridis",
                vmin=-limit, vmax=limit, ax=ax, cbar_kws={"label": "Δ validation Macro-F1"})
    col = list(table.columns).index(.001)
    row = list(table.index).index(15)
    ax.add_patch(plt.Rectangle((col, row), 1, 1, fill=False, edgecolor="black", linewidth=2))
    ax.set(xlabel="Minimum accepted gain", ylabel="Probe epochs",
           title="Dynamic incremental capacity sensitivity")
    _save(fig, path); return fig


def plot_incremental(stages, progression_path="report/figures/incremental_progression.pdf",
                     growth_path="report/figures/network_growth.pdf"):
    accepted = stages.sort_values("attempt").groupby(
        ["dataset", "method", "outer_fold", "seed", "stage"], as_index=False).tail(1)
    curriculum = accepted[accepted.method.isin(["fixed_incremental", "dynamic_incremental"])].copy()
    curriculum["active_count"] = curriculum.stage + 1
    fold = curriculum.groupby(["method", "outer_fold", "active_count"], as_index=False).val_macro_f1.mean()
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for method in ("fixed_incremental", "dynamic_incremental"):
        part = fold[fold.method == method]
        for _, group in part.groupby("outer_fold"):
            ax.plot(group.active_count, group.val_macro_f1, color=METHOD_COLORS[method], alpha=.25, linewidth=.9)
        mean = part.groupby("active_count").val_macro_f1.mean()
        ax.plot(mean.index, mean, marker="o", color=METHOD_COLORS[method], linewidth=2,
                label=METHOD_LABELS[method])
    ax.set(xlabel="Active classes", ylabel="Validation Macro-F1",
           title="Validation performance during class introduction")
    ax.set_xticks(sorted(fold.active_count.unique()))
    ax.legend(fontsize=8)
    _save(fig, progression_path)
    dynamic = accepted[accepted.method == "dynamic_incremental"].copy()
    reference = accepted[accepted.method == "dynamic_all_classes"]
    fig2, ax2 = plt.subplots(figsize=(5.2, 3.5))
    rng = np.random.default_rng(42)
    for stage, group in dynamic.groupby("stage"):
        ax2.scatter(stage + 1 + rng.uniform(-.13, .13, len(group)), group.hidden_units,
                    color=METHOD_COLORS["dynamic_incremental"], alpha=.4, s=15)
    mean = dynamic.groupby("stage").hidden_units.mean()
    ax2.plot(mean.index + 1, mean, marker="D", color=METHOD_COLORS["dynamic_incremental"],
             label="Dynamic incremental mean")
    ax2.axhline(reference.hidden_units.mean(), linestyle="--",
                color=METHOD_COLORS["dynamic_all_classes"],
                label="Dynamic all-classes mean (all classes from start)")
    ax2.set(xlabel="Active classes", ylabel="Hidden units", title="Dynamic capacity growth")
    ax2.set_xticks(sorted(dynamic.stage.unique()+1))
    ax2.set_yticks(range(int(dynamic.hidden_units.max())+1))
    ax2.legend(fontsize=8)
    _save(fig2, growth_path); return fig, fig2


def plot_confusions(methods, class_names, root="report/figures"):
    for method, filename, title in [
        ("baseline", "confusion_baseline.pdf", "Baseline confusion matrix"),
        ("dynamic_incremental", "confusion_incremental.pdf", "Dynamic incremental confusion matrix"),
        ("fixed_incremental", "confusion_fixed.pdf", "Fixed incremental confusion matrix")]:
        matrices = [confusion_matrix(r["y_true"], r["y_pred"], labels=np.arange(len(class_names)),
                                     normalize="true") for r in methods[method]]
        cm, cm_std = np.mean(matrices, axis=0), np.std(matrices, axis=0, ddof=1)
        Path(root).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(Path(root)/f"{method}_confusion_mean.csv")
        pd.DataFrame(cm_std, index=class_names, columns=class_names).to_csv(Path(root)/f"{method}_confusion_std.csv")
        fig, ax = plt.subplots(figsize=(7, 6)); sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set(xlabel="Predicted", ylabel="True", title=title); _save(fig, Path(root)/filename)


def export_summaries(main, per_class, stages, stats, efficiency, root="results/summary"):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    main.to_csv(root/"main_results.csv", index=False)
    per_class.to_csv(root/"per_class_results.csv", index=False)
    stages.to_csv(root/"stage_results.csv", index=False)
    stats.to_csv(root/"statistical_results.csv", index=False)
    efficiency.to_csv(root/"efficiency_results.csv", index=False)
