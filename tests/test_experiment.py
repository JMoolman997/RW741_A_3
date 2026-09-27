import json
import numpy as np
import pandas as pd
import pytest

from src.data import DatasetBundle, class_order, decode_one_hot_targets, load_dry_bean, make_nested_folds
from src.model import FeedForwardClassifier
from src.training import objective_loss, train_model
from src.experiments import ExperimentConfig, architecture_boundary_status, config_fingerprint, holm_adjust
from src.incremental import run_incremental
from src.plotting import plot_architecture_search, plot_class_distribution
from src.reporting import validate_result_provenance, validate_submission_metadata


def synthetic_bundle():
    rng = np.random.default_rng(7)
    y = np.repeat(["a", "b", "c"], 30)
    X = pd.DataFrame(rng.normal(size=(90, 4)), columns=list("wxyz"))
    return DatasetBundle("synthetic", X, pd.Series(y), ["a", "b", "c"], "test", 0)


def test_nested_outer_folds_are_disjoint_and_complete():
    bundle = synthetic_bundle(); folds = make_nested_folds(bundle, 5, 11)
    all_test = np.concatenate([fold.test_indices for fold in folds])
    assert len(np.unique(all_test)) == len(bundle.y) == len(all_test)
    for fold in folds:
        assert not set(fold.test_indices) & set(fold.train_indices)
        assert not set(fold.test_indices) & set(fold.val_indices)
        assert np.allclose(fold.X_train.mean(0), 0, atol=1e-10)


def test_one_hot_targets_require_exactly_one_class():
    valid = pd.DataFrame({"a": [1, 0], "b": [0, 1]})
    assert decode_one_hot_targets(valid).tolist() == ["a", "b"]
    with pytest.raises(ValueError):
        decode_one_hot_targets(pd.DataFrame({"a": [1], "b": [1]}))


def test_output_and_hidden_expansion_preserve_function():
    model = FeedForwardClassifier(4, 2, 0, seed=2); X = np.arange(20).reshape(5, 4) / 10
    before = model.forward(X); old_w = model.params["W"].copy()
    model.add_hidden_neuron()
    assert np.array_equal(model.params["W"], old_w)
    assert np.allclose(model.forward(X), before)
    old = {k: v.copy() for k, v in model.params.items()}; model.expand_output(3)
    assert np.array_equal(model.params["W"][:, :2], old["W"])
    assert np.array_equal(model.params["W2"][:, :2], old["W2"])


def test_adam_state_must_reset_after_expansion():
    rng = np.random.default_rng(4); X = rng.normal(size=(60, 4)); y = (X[:, 0] > 0).astype(int)
    model = FeedForwardClassifier(4, 2, 0, seed=4)
    model, _, _, state = train_model(model, X[:40], y[:40], X[40:], y[40:], max_epochs=2,
                                     return_state=True)
    model.add_hidden_neuron()
    with pytest.raises(ValueError):
        train_model(model, X[:40], y[:40], X[40:], y[40:], max_epochs=1,
                    optimizer_state=state, return_state=True)
    _, _, _, reset = train_model(model, X[:40], y[:40], X[40:], y[40:], max_epochs=1,
                                 optimizer_state=None, return_state=True)
    assert reset[2] > 0


def test_l2_objective_matches_returned_loss():
    model = FeedForwardClassifier(3, 2, 2, seed=5)
    X = np.ones((4, 3)); y = np.array([0, 1, 0, 1]); coefficient = .02
    loss, _ = model.loss_and_gradients(X, y, coefficient)
    assert np.isclose(loss, objective_loss(model, X, y, coefficient))


def test_invalid_activation_fails():
    with pytest.raises(ValueError):
        FeedForwardClassifier(2, 2, activation="sigmoid")


def test_update_budget_is_exact():
    rng = np.random.default_rng(8); X = rng.normal(size=(80, 3)); y = (X[:, 0] > 0).astype(int)
    model = FeedForwardClassifier(3, 2, seed=8)
    _, history, reason = train_model(model, X[:60], y[:60], X[60:], y[60:],
                                     batch_size=8, max_epochs=20, max_updates=11,
                                     early_stopping=False)
    assert reason == "update_budget" and history[-1]["updates"] == 11


def test_dynamic_all_classes_starts_with_every_class_and_zero_capacity():
    fold = make_nested_folds(synthetic_bundle(), 5, 11)[0]
    config = ExperimentConfig(seeds=(1,), architecture_sizes=(0, 1), max_epochs=1,
                              patience=1, capacity_probe_epochs=1,
                              max_hidden_neurons=1)
    result, stages, _ = run_incremental(
        fold, class_order(fold.y_train), config, seed=1, hidden_units=0,
        dynamic=True, max_hidden_units=1, initial_classes="all", evaluate_test=False)
    assert len(json.loads(stages[0]["active_classes"])) == 3
    assert result["initial_classes"] == "all"
    assert result["total_updates"] > 0 and result["total_examples"] > 0


def test_architecture_boundary_guard_only_blocks_steel():
    config = ExperimentConfig(architecture_sizes=(0, 1, 2))
    steel = {("steel_plates", fold): 2 for fold in range(1, 6)}
    bean = {("dry_bean", fold): 2 for fold in range(1, 6)}
    assert not architecture_boundary_status(steel, config, "steel_plates")["resolved"]
    assert architecture_boundary_status(bean, config, "dry_bean")["resolved"]


def test_holm_adjustment_is_monotone_and_bounded():
    adjusted = holm_adjust([.01, .03, .2])
    assert np.allclose(adjusted, [.03, .06, .2])
    assert np.all(np.diff(adjusted) >= 0) and np.all(adjusted <= 1)


def test_submission_metadata_rejects_placeholders_and_non_https_url():
    with pytest.raises(ValueError, match="placeholders"):
        validate_submission_metadata({"author": "PLACEHOLDER AUTHOR"})
    metadata = {"author": "A", "student_number": "1", "institution": "U",
                "module_code": "741", "repository_url": "ssh://example/repo"}
    with pytest.raises(ValueError, match="HTTPS"):
        validate_submission_metadata(metadata)
    metadata["repository_url"] = "https://example/repo"
    assert validate_submission_metadata(metadata) == "1RW741assignment3.pdf"


def test_provenance_rejects_stale_configuration_and_unresolved_boundary():
    config = ExperimentConfig(seeds=(1,))
    manifest = {"configuration_fingerprint": "stale",
                "methods": ["baseline"]}
    with pytest.raises(ValueError, match="Stale"):
        validate_result_provenance({"steel_plates": object()}, {"steel_plates": manifest},
                                   config, [{"dataset": "steel_plates", "resolved": True}])
    manifest = {"configuration_fingerprint": config_fingerprint(config),
                "methods": ["baseline", "fixed_update_budget_matched_baseline",
                            "fixed_incremental", "dynamic_all_classes", "dynamic_incremental"]}
    with pytest.raises(ValueError, match="upper boundary"):
        validate_result_provenance({"steel_plates": object()}, {"steel_plates": manifest},
                                   config, [{"dataset": "steel_plates", "resolved": False}])


def test_plot_titles_and_fold_specific_architecture_markers(tmp_path):
    fig = plot_class_distribution(pd.Series(["a", "a", "b"]), "Steel Plates Faults",
                                  tmp_path / "classes.pdf")
    assert fig.axes[0].get_title() == "Steel Plates Faults class distribution"
    summary = pd.DataFrame({"dataset": ["steel_plates"] * 4, "outer_fold": [1, 1, 2, 2],
                            "hidden_units": [0, 2, 0, 2], "mean": [.4, .5, .45, .55],
                            "std": [.01] * 4})
    selected = {("steel_plates", 1): 2, ("steel_plates", 2): 2}
    fig = plot_architecture_search(summary, selected, tmp_path / "architecture.pdf")
    assert len(fig.axes) == 2
    assert fig.axes[0].lines[0].get_label() == "Fold 1 ($h=2$)"
    assert fig.axes[1].lines[0].get_label() == "Fold 2 ($h=2$)"
    assert all("boundary" in ax.get_title() for ax in fig.axes)


def test_dry_bean_duplicates_removed_and_order_is_stable():
    bundle = load_dry_bean()
    assert bundle.X.shape == (13543, 16) and bundle.y.nunique() == 7
    y = np.array([0]*4 + [1]*2 + [2]*3)
    assert class_order(y) == [1, 2, 0]
