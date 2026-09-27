from __future__ import annotations

import random
import numpy as np
from .metrics import classification_metrics


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)


def cross_entropy(model, X, y):
    p = model.forward(X)
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


def objective_loss(model, X, y, l2_coefficient=0.0):
    penalty = .5 * l2_coefficient * sum(
        np.sum(value ** 2) for key, value in model.params.items() if key.startswith("W")
    )
    return cross_entropy(model, X, y) + float(penalty)


def _aligned_state(model, state):
    """Restore Adam state only when the parameter shapes are unchanged."""
    if state is None:
        return ({k: np.zeros_like(x) for k, x in model.params.items()},
                {k: np.zeros_like(x) for k, x in model.params.items()}, 0)
    old_m, old_v, step = state
    if set(old_m) != set(model.params) or any(old_m[k].shape != model.params[k].shape for k in model.params):
        raise ValueError("Adam state does not match the current architecture; reset it after expansion")
    return ({k: old_m[k].copy() for k in model.params},
            {k: old_v[k].copy() for k in model.params}, step)


def train_model(model, X_train, y_train, X_val, y_val, *, seed=1, learning_rate=.01,
                batch_size=128, max_epochs=150, patience=10, min_delta=.001,
                l2_coefficient=1e-4, optimizer_state=None, return_state=False,
                max_updates=None, early_stopping=True):
    """Train with Adam; checkpoint and stop using validation Macro-F1 only."""
    set_seed(seed); rng = np.random.default_rng(seed)
    m, v, step = _aligned_state(model, optimizer_state)
    start_step, examples_processed = step, 0
    best, best_state = model.copy(), None
    best_score, meaningful_best, stale, history = -np.inf, -np.inf, 0, []
    labels = np.arange(model.output_dim)
    for epoch in range(1, max_epochs + 1):
        for batch in np.array_split(rng.permutation(len(y_train)), max(1, int(np.ceil(len(y_train)/batch_size)))):
            if max_updates is not None and step - start_step >= max_updates:
                break
            _, grads = model.loss_and_gradients(X_train[batch], y_train[batch], l2_coefficient)
            step += 1
            examples_processed += len(batch)
            for key, grad in grads.items():
                m[key] = .9*m[key] + .1*grad; v[key] = .999*v[key] + .001*grad*grad
                mh, vh = m[key]/(1-.9**step), v[key]/(1-.999**step)
                model.params[key] -= learning_rate * mh/(np.sqrt(vh)+1e-8)
        train_pred, val_pred = model.predict(X_train), model.predict(X_val)
        train_f1 = classification_metrics(y_train, train_pred, labels)["macro_f1"]
        val_f1 = classification_metrics(y_val, val_pred, labels)["macro_f1"]
        row = {"epoch": epoch, "train_loss": cross_entropy(model,X_train,y_train),
               "train_objective": objective_loss(model, X_train, y_train, l2_coefficient),
               "val_loss": cross_entropy(model,X_val,y_val), "train_macro_f1": train_f1,
               "val_macro_f1": val_f1, "updates": step-start_step,
               "examples_processed": examples_processed}
        history.append(row)
        if val_f1 > best_score:
            best, best_score = model.copy(), val_f1
            best_state = ({k: x.copy() for k, x in m.items()},
                          {k: x.copy() for k, x in v.items()}, step)
        if val_f1 > meaningful_best + min_delta:
            meaningful_best, stale = val_f1, 0
        else:
            stale += 1
        budget_done = max_updates is not None and step - start_step >= max_updates
        if budget_done or (early_stopping and stale >= patience):
            break
    reason = "update_budget" if max_updates is not None and step-start_step >= max_updates else (
             "stagnation" if early_stopping and stale >= patience else "max_epochs")
    result = (best, history, reason)
    return result + (best_state,) if return_state else result
