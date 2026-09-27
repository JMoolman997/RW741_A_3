from __future__ import annotations

import json
from time import perf_counter

import numpy as np

from .metrics import classification_metrics
from .model import FeedForwardClassifier
from .training import cross_entropy, train_model


def _subset(X, y, active):
    mask = np.isin(y, active)
    local = {label: i for i, label in enumerate(active)}
    return X[mask], np.array([local[v] for v in y[mask]], dtype=int)


def run_incremental(split, class_order, config, seed, hidden_units=0, dynamic=True,
                    max_hidden_units=None, initial_classes="minority_pair",
                    evaluate_test=True):
    """Run the shared growth protocol with a curriculum or all classes available.

    ``initial_classes`` is deliberately explicit: ``minority_pair`` implements the
    assignment curriculum, while ``all`` is its capacity-policy-matched control.
    With ``evaluate_test=False`` the outer test partition is never accessed, which
    is required for validation-only sensitivity analysis.
    """
    if initial_classes not in {"minority_pair", "all"}:
        raise ValueError("initial_classes must be 'minority_pair' or 'all'")
    active = list(class_order if initial_classes == "all" else class_order[:2])
    model = FeedForwardClassifier(split.X_train.shape[1], len(active), hidden_units, seed,
                                  config.activation)
    stages, optimizer_state = [], None
    total_epochs = 0
    total_updates = total_examples = accepted_probes = rejected_probes = 0
    max_hidden_units = config.max_hidden_neurons if max_hidden_units is None else max_hidden_units
    started = perf_counter()
    stage_count = 1 if initial_classes == "all" else len(class_order) - 1
    failure_reason = None
    for stage_number in range(1, stage_count + 1):
        introduced = active[-1] if stage_number > 1 else None
        attempts = 0
        while True:
            attempts += 1
            Xt, yt = _subset(split.X_train, split.y_train, active)
            Xv, yv = _subset(split.X_val, split.y_val, active)
            model, history, reason, optimizer_state = train_model(
                model, Xt, yt, Xv, yv, seed=seed + stage_number + attempts,
                learning_rate=config.learning_rate, batch_size=config.batch_size,
                max_epochs=config.max_epochs, patience=config.patience,
                min_delta=config.min_delta, l2_coefficient=config.l2_coefficient,
                optimizer_state=optimizer_state, return_state=True)
            total_updates += history[-1]["updates"]
            total_examples += history[-1]["examples_processed"]
            total_epochs += len(history)
            train_metrics = classification_metrics(yt, model.predict(Xt), np.arange(len(active)))
            val_metrics = classification_metrics(yv, model.predict(Xv), np.arange(len(active)))
            train_loss, val_loss = cross_entropy(model, Xt, yt), cross_entropy(model, Xv, yv)
            grew = False
            probe_gain = None
            if dynamic and reason == "stagnation" and model.hidden_units < max_hidden_units:
                candidate = model.copy(); candidate.add_hidden_neuron()
                candidate, probe_history, _, candidate_state = train_model(
                    candidate, Xt, yt, Xv, yv, seed=seed + 1000 + stage_number + attempts,
                    learning_rate=config.learning_rate, batch_size=config.batch_size,
                    max_epochs=config.capacity_probe_epochs, patience=config.patience,
                    min_delta=config.min_delta, l2_coefficient=config.l2_coefficient,
                    optimizer_state=None, return_state=True, early_stopping=False)
                candidate_metrics = classification_metrics(
                    yv, candidate.predict(Xv), np.arange(len(active)))
                probe_gain = candidate_metrics["macro_f1"] - val_metrics["macro_f1"]
                total_updates += probe_history[-1]["updates"]
                total_examples += probe_history[-1]["examples_processed"]
                total_epochs += len(probe_history)
                if probe_gain >= config.capacity_min_gain:
                    model, optimizer_state, val_metrics = candidate, candidate_state, candidate_metrics
                    train_metrics = classification_metrics(yt, model.predict(Xt), np.arange(len(active)))
                    train_loss, val_loss = cross_entropy(model, Xt, yt), cross_entropy(model, Xv, yv)
                    grew = True
                    accepted_probes += 1
                else:
                    rejected_probes += 1
            stages.append({
                "seed": seed, "stage": stage_number, "attempt": attempts,
                "active_classes": json.dumps([split.class_names[x] for x in active]),
                "introduced_class": None if introduced is None else split.class_names[introduced],
                "class_counts": json.dumps({split.class_names[x]: int((split.y_train == x).sum()) for x in active}),
                "hidden_units": model.hidden_units, "epochs": len(history),
                "stop_reason": reason, "capacity_increased": grew,
                "capacity_probe_gain": probe_gain,
                "train_loss": train_loss, "val_loss": val_loss,
                "train_macro_f1": train_metrics["macro_f1"], "val_macro_f1": val_metrics["macro_f1"],
                "accuracy": val_metrics["accuracy"], "balanced_accuracy": val_metrics["balanced_accuracy"],
                "precision": json.dumps(val_metrics["precision"]), "recall": json.dumps(val_metrics["recall"]),
                "f1": json.dumps(val_metrics["f1"]),
            })
            if reason == "max_epochs":
                failure_reason = (
                    f"{split.dataset} fold {split.outer_fold} seed {seed} stage "
                    f"{stage_number} reached max_epochs before stagnation"
                )
                break
            if grew:
                continue
            break
        if failure_reason:
            break
        if len(active) < len(class_order):
            active.append(class_order[len(active)])
            model.expand_output(len(active))
            optimizer_state = None

    complete = failure_reason is None and len(active) == len(class_order)
    if evaluate_test and complete:
        local_test = np.array([active.index(v) for v in split.y_test])
        test_pred_local = model.predict(split.X_test)
        test_pred = np.array([active[v] for v in test_pred_local])
        result = classification_metrics(split.y_test, test_pred, np.arange(len(split.class_names)))
        result.update(y_true=split.y_test.tolist(), y_pred=test_pred.tolist(),
                      test_loss=cross_entropy(model, split.X_test, local_test))
    else:
        result = {}
    result.update(seed=seed, epochs=total_epochs,
                  total_updates=total_updates, total_examples=total_examples,
                  accepted_probes=accepted_probes, rejected_probes=rejected_probes,
                  elapsed_seconds=perf_counter()-started,
                  hidden_units=model.hidden_units, final_hidden_units=model.hidden_units,
                  run_status="complete" if complete else "failed",
                  failure_reason=failure_reason, initial_classes=initial_classes)
    return result, stages, model
