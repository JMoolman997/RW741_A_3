from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_rel

from .data import class_order
from .incremental import run_incremental
from .metrics import classification_metrics
from .model import FeedForwardClassifier
from .training import cross_entropy, train_model


@dataclass
class ExperimentConfig:
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    architecture_sizes: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    learning_rate: float = .01
    batch_size: int = 128
    max_epochs: int = 150
    patience: int = 10
    min_delta: float = .001
    l2_coefficient: float = 1e-4
    activation: str = "relu"
    max_hidden_neurons: int = 512
    capacity_probe_epochs: int = 15
    capacity_min_gain: float = .001
    sensitivity_probe_epochs: tuple[int, ...] = (10, 15, 25)
    sensitivity_min_gains: tuple[float, ...] = (.0005, .001, .002)
    sensitivity_seed: int = 1


def _jsonable(row):
    return {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}


def architecture_search(folds, config):
    rows = []
    for fold in folds:
        for hidden in config.architecture_sizes:
            for seed in config.seeds:
                model = FeedForwardClassifier(fold.X_train.shape[1], len(fold.class_names), hidden,
                                              seed, config.activation)
                model, hist, stop = train_model(
                    model, fold.X_train, fold.y_train, fold.X_val, fold.y_val,
                    seed=seed, learning_rate=config.learning_rate, batch_size=config.batch_size,
                    max_epochs=config.max_epochs, patience=config.patience, min_delta=config.min_delta,
                    l2_coefficient=config.l2_coefficient)
                vm = classification_metrics(fold.y_val, model.predict(fold.X_val),
                                            np.arange(len(fold.class_names)))
                rows.append({"dataset": fold.dataset, "outer_fold": fold.outer_fold,
                    "hidden_units": hidden, "seed": seed, "epochs": len(hist), "stop_reason": stop,
                    "train_loss": hist[-1]["train_loss"], "val_loss": hist[-1]["val_loss"],
                    "train_macro_f1": hist[-1]["train_macro_f1"], "val_macro_f1": vm["macro_f1"]})
    raw = pd.DataFrame(rows)
    summary = raw.groupby(["dataset", "outer_fold", "hidden_units"])["val_macro_f1"].agg(
        mean="mean", std="std").reset_index()
    selected = {}
    for (dataset, outer_fold), group in summary.groupby(["dataset", "outer_fold"]):
        best = group["mean"].max()
        selected[(dataset, int(outer_fold))] = int(
            group.loc[group["mean"] >= best - .001, "hidden_units"].min())
    return raw, summary, selected


def architecture_boundary_status(selected, config, dataset, threshold=3):
    maximum = max(config.architecture_sizes)
    count = sum(size == maximum for (name, _), size in selected.items() if name == dataset)
    return {"dataset": dataset, "largest_candidate": maximum,
            "folds_selecting_largest": count, "threshold": threshold,
            "resolved": dataset != "steel_plates" or count < threshold}


def _minority_metrics(metric, order, count=2):
    idx = order[:count]
    return {"minority_macro_f1": float(np.mean([metric["f1"][i] for i in idx])),
            "minority_recall": float(np.mean([metric["recall"][i] for i in idx]))}


def run_baseline_experiment(fold, config, hidden_units, update_budgets=None):
    rows = []
    labels, order = np.arange(len(fold.class_names)), class_order(fold.y_train)
    for seed in config.seeds:
        started = perf_counter()
        model = FeedForwardClassifier(fold.X_train.shape[1], len(labels), hidden_units, seed, config.activation)
        budget = None if update_budgets is None else update_budgets[seed]
        batches = int(np.ceil(len(fold.y_train) / config.batch_size))
        budget_epochs = config.max_epochs if budget is None else int(np.ceil(budget / batches)) + 1
        model, hist, stop = train_model(
            model, fold.X_train, fold.y_train, fold.X_val, fold.y_val,
            seed=seed, learning_rate=config.learning_rate, batch_size=config.batch_size,
            max_epochs=max(config.max_epochs, budget_epochs), patience=config.patience,
            min_delta=config.min_delta, l2_coefficient=config.l2_coefficient,
            max_updates=budget, early_stopping=budget is None)
        metric = classification_metrics(fold.y_test, model.predict(fold.X_test), labels)
        metric.update(_minority_metrics(metric, order))
        metric.update(dataset=fold.dataset, outer_fold=fold.outer_fold, seed=seed,
            selected_hidden_units=hidden_units, epochs=len(hist), final_hidden_units=hidden_units,
            stop_reason=stop, total_updates=hist[-1]["updates"],
            total_examples=hist[-1]["examples_processed"], accepted_probes=0, rejected_probes=0,
            run_status="complete", failure_reason=None,
            elapsed_seconds=perf_counter()-started,
            train_loss=cross_entropy(model, fold.X_train, fold.y_train),
            val_loss=cross_entropy(model, fold.X_val, fold.y_val),
            test_loss=cross_entropy(model, fold.X_test, fold.y_test),
            y_true=fold.y_test.tolist(), y_pred=model.predict(fold.X_test).tolist())
        rows.append(metric)
    return rows


def run_dataset_experiments(folds, config, selected_sizes, result_dir):
    result_dir = Path(result_dir); (result_dir / "raw").mkdir(parents=True, exist_ok=True)
    methods = {name: [] for name in ("baseline", "fixed_update_budget_matched_baseline",
        "dynamic_all_classes", "dynamic_incremental", "fixed_incremental")}
    stages, orders = [], {}
    for fold in folds:
        selected = selected_sizes[(fold.dataset, fold.outer_fold)]
        order = class_order(fold.y_train); orders[fold.outer_fold] = order
        fixed_budgets = {}
        for seed in config.seeds:
            dynamic, dynamic_stages, _ = run_incremental(
                fold, order, config, seed, 0, True, max_hidden_units=selected,
                initial_classes="minority_pair")
            fixed, fixed_stages, _ = run_incremental(
                fold, order, config, seed, selected, False, max_hidden_units=selected,
                initial_classes="minority_pair")
            all_classes, all_stages, _ = run_incremental(
                fold, order, config, seed, 0, True, max_hidden_units=selected,
                initial_classes="all")
            for row in (dynamic, fixed):
                row.update(dataset=fold.dataset, outer_fold=fold.outer_fold,
                           selected_hidden_units=selected)
                if row["run_status"] == "complete":
                    row.update(_minority_metrics(row, order))
            methods["dynamic_incremental"].append(dynamic)
            methods["fixed_incremental"].append(fixed)
            all_classes.update(dataset=fold.dataset, outer_fold=fold.outer_fold,
                               selected_hidden_units=selected)
            if all_classes["run_status"] == "complete":
                all_classes.update(_minority_metrics(all_classes, order))
            methods["dynamic_all_classes"].append(all_classes)
            fixed_budgets[seed] = fixed["total_updates"]
            stages += [dict(method="dynamic_incremental", dataset=fold.dataset,
                            outer_fold=fold.outer_fold, **row) for row in dynamic_stages]
            stages += [dict(method="fixed_incremental", dataset=fold.dataset,
                            outer_fold=fold.outer_fold, **row) for row in fixed_stages]
            stages += [dict(method="dynamic_all_classes", dataset=fold.dataset,
                            outer_fold=fold.outer_fold, **row) for row in all_stages]
        methods["baseline"] += run_baseline_experiment(fold, config, selected)
        methods["fixed_update_budget_matched_baseline"] += run_baseline_experiment(
            fold, config, selected, fixed_budgets)
    for name, rows in methods.items():
        pd.DataFrame([_jsonable(row) for row in rows]).to_csv(result_dir/"raw"/f"{name}.csv", index=False)
    stage_frame = pd.DataFrame(stages)
    stage_frame.to_csv(result_dir/"raw"/"stage_results.csv", index=False)
    pd.DataFrame([{"outer_fold": fold, "minority_rank": rank + 1,
                   "class_index": int(label), "class_name": folds[0].class_names[int(label)]}
                  for fold, order in orders.items() for rank, label in enumerate(order[:2])]).to_csv(
                      result_dir/"raw"/"minority_classes.csv", index=False)
    write_result_manifest(result_dir, config, methods, len(folds))
    return methods, stage_frame, orders


def run_capacity_sensitivity(folds, config, selected_sizes, result_dir):
    """Validation-only 3x3 probe sensitivity study using one declared seed."""
    rows = []
    for fold in folds:
        order = class_order(fold.y_train)
        maximum = selected_sizes[(fold.dataset, fold.outer_fold)]
        for epochs in config.sensitivity_probe_epochs:
            for gain in config.sensitivity_min_gains:
                variant = replace(config, capacity_probe_epochs=epochs, capacity_min_gain=gain)
                result, stages, _ = run_incremental(
                    fold, order, variant, config.sensitivity_seed, 0, True,
                    max_hidden_units=maximum, initial_classes="minority_pair",
                    evaluate_test=False)
                last = stages[-1] if stages else {}
                rows.append({"dataset": fold.dataset, "outer_fold": fold.outer_fold,
                    "seed": config.sensitivity_seed, "capacity_probe_epochs": epochs,
                    "capacity_min_gain": gain, "run_status": result["run_status"],
                    "failure_reason": result["failure_reason"],
                    "final_validation_macro_f1": last.get("val_macro_f1", np.nan),
                    "final_hidden_units": result["final_hidden_units"],
                    "total_updates": result["total_updates"],
                    "accepted_probes": result["accepted_probes"],
                    "rejected_probes": result["rejected_probes"]})
    frame = pd.DataFrame(rows)
    root = Path(result_dir) / "sensitivity"
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "capacity_sensitivity.csv", index=False)
    sensitivity_manifest = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_fingerprint": config_fingerprint(config), "validation_only": True,
        "seed": config.sensitivity_seed,
        "capacity_probe_epochs": list(config.sensitivity_probe_epochs),
        "capacity_min_gains": list(config.sensitivity_min_gains)}
    (root / "manifest.json").write_text(json.dumps(sensitivity_manifest, indent=2))
    return frame


def config_fingerprint(config):
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def write_result_manifest(result_dir, config, methods, outer_folds):
    manifest = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": asdict(config), "configuration_fingerprint": config_fingerprint(config),
        "methods": list(methods), "outer_folds": outer_folds,
        "seeds": list(config.seeds), "canonical": True}
    Path(result_dir, "raw", "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def summarise_results(methods, class_names):
    main, per_class, efficiency = [], [], []
    for method, rows in methods.items():
        frame = pd.DataFrame(rows)
        if "run_status" not in frame or not frame.run_status.eq("complete").all():
            raise ValueError(f"Cannot summarise incomplete runs for {method}")
        for metric in ("macro_f1", "minority_macro_f1", "minority_recall", "accuracy", "balanced_accuracy"):
            fold_seed = frame.pivot(index="outer_fold", columns="seed", values=metric)
            fold_means = fold_seed.mean(axis=1)
            main.append({"method": method, "metric": metric, "mean": fold_means.mean(),
                "between_fold_std": fold_means.std(ddof=1),
                "mean_within_fold_std": fold_seed.std(axis=1, ddof=1).mean()})
        for i, name in enumerate(class_names):
            for metric in ("precision", "recall", "f1"):
                frame["value"] = frame[metric].map(lambda values: values[i])
                fold_seed = frame.pivot(index="outer_fold", columns="seed", values="value")
                fold_means = fold_seed.mean(axis=1)
                per_class.append({"method": method, "class": name, "metric": metric,
                    "mean": fold_means.mean(), "between_fold_std": fold_means.std(ddof=1),
                    "mean_within_fold_std": fold_seed.std(axis=1, ddof=1).mean()})
        efficiency.append({"method": method, "macro_f1": frame.macro_f1.mean(),
            "minority_macro_f1": frame.minority_macro_f1.mean(),
            "mean_updates": frame.total_updates.mean(), "mean_examples": frame.total_examples.mean(),
            "mean_final_hidden_units": frame.final_hidden_units.mean(),
            "mean_accepted_probes": frame.accepted_probes.mean(),
            "mean_rejected_probes": frame.rejected_probes.mean(),
            "completed_runs": int(frame.run_status.eq("complete").sum()),
            "mean_elapsed_seconds": frame.elapsed_seconds.mean()})

    comparisons = [("fixed_incremental", "fixed_update_budget_matched_baseline"),
                   ("dynamic_incremental", "dynamic_all_classes")]
    stats = []
    for metric in ("macro_f1", "minority_macro_f1"):
        fold_means = {name: pd.DataFrame(rows).groupby("outer_fold")[metric].mean()
                      for name, rows in methods.items()}
        for first, second in comparisons:
            if not fold_means[first].index.equals(fold_means[second].index):
                raise ValueError(f"Missing paired outer folds for {first} and {second}")
            first_values, second_values = fold_means[first].to_numpy(), fold_means[second].to_numpy()
            differences = first_values - second_values
            n = len(differences); mean = differences.mean(); sd = differences.std(ddof=1)
            se = sd / np.sqrt(n); critical = t.ppf(.975, df=n-1)
            test = ttest_rel(first_values, second_values)
            stats.append({"family": "primary" if metric == "macro_f1" else "exploratory_minority",
                "metric": metric, "comparison": f"{first} - {second}",
                "mean_difference": mean, "ci95_low": mean-critical*se, "ci95_high": mean+critical*se,
                "paired_effect_dz": mean/sd if sd else np.nan, "t_statistic": test.statistic,
                "p_value": test.pvalue, "n_folds": n})
    stats = pd.DataFrame(stats); stats["p_holm"] = np.nan
    for family, group in stats.groupby("family"):
        ordered = group.sort_values("p_value")
        stats.loc[ordered.index, "p_holm"] = holm_adjust(ordered.p_value.to_numpy())
    return pd.DataFrame(main), pd.DataFrame(per_class), stats, pd.DataFrame(efficiency)


def holm_adjust(sorted_p_values):
    values = np.asarray(sorted_p_values, dtype=float)
    return np.minimum(np.maximum.accumulate(
        [(len(values)-i)*p for i, p in enumerate(values)]), 1.0)
