"""Build the canonical two-dataset experiment notebook."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
m, c = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
m(r"""# Two-Dataset Incremental Class Learning

This notebook is the sole experiment entry point. It evaluates minority-to-majority incremental learning on UCI Dry Bean and Steel Plates Faults. Five-fold outer cross-validation with a fold-specific inner training/validation split keeps test observations untouched during architecture selection."""),
m("## Configuration and reproducibility\n\nFive optimisation seeds quantify stochastic training variation inside each of five disjoint outer test folds."),
c("""from pathlib import Path
import numpy as np
import pandas as pd
import json
from IPython.display import display, Markdown
from src.data import load_dry_bean, load_steel_plates, inspect_dataset, make_nested_folds, class_order
from src.experiments import (ExperimentConfig, architecture_search, architecture_boundary_status,
                             run_dataset_experiments, run_capacity_sensitivity, summarise_results,
                             config_fingerprint)
from src.plotting import (plot_class_distribution, plot_architecture_search,
                          plot_incremental, plot_confusions, plot_sensitivity,
                          plot_dynamic_capacity_performance,
                          plot_primary_effects, plot_per_class_effects, plot_performance_cost,
                          export_summaries)
from src.reporting import (export_generated_results, compile_report, load_manifest,
                           validate_result_provenance)

CONFIG = ExperimentConfig(seeds=(1, 2, 3, 4, 5))
SUBMISSION = {'author': 'PLACEHOLDER AUTHOR', 'student_number': 'PLACEHOLDER STUDENT NUMBER',
              'institution': 'PLACEHOLDER INSTITUTION', 'module_code': 'PLACEHOLDER MODULE',
              'repository_url': 'PLACEHOLDER HTTPS REPOSITORY URL'}
DATASETS = [load_dry_bean('data'), load_steel_plates('data')]
FOLDS = {d.name: make_nested_folds(d, n_splits=5, seed=42) for d in DATASETS}
CONFIG"""),
m(r"""## Dataset inspection

Both data sets contain seven naturally imbalanced classes and numerical features. Exact duplicates are removed before fold construction. The imbalance ratio is $IR=n_{\max}/n_{\min}$. Steel Plates provides a second domain where rare-class difficulty need not resemble the readily separable BOMBAY class."""),
c("""for dataset in DATASETS:
    info = inspect_dataset(dataset)
    display(Markdown(f'### {dataset.name}'))
    display(pd.DataFrame({'count': info['class_counts'], 'proportion': info['class_proportions']}))
    display({k:v for k,v in info.items() if k not in ('class_counts','class_proportions')})
    display_name = {'dry_bean': 'Dry Bean', 'steel_plates': 'Steel Plates Faults'}[dataset.name]
    plot_class_distribution(dataset.y, display_name,
                            f'report/figures/{dataset.name}/class_distribution.pdf')"""),
m(r"""## Outer folds, inner split, and preprocessing

Each outer fold reserves 20% for final testing. Inner training and validation occupy approximately 65% and 15%. The scaler uses inner-training statistics only: $z_j=(x_j-\mu_j)/\sigma_j$. Architecture selection, stopping, growth, and class order never access the corresponding outer test fold."""),
c("""for dataset in DATASETS:
    folds = FOLDS[dataset.name]
    assert len(set(np.concatenate([f.test_indices for f in folds]))) == len(dataset.y)
    assert sum(len(f.test_indices) for f in folds) == len(dataset.y)
    print(dataset.name, [(len(f.y_train), len(f.y_val), len(f.y_test)) for f in folds])"""),
m(r"""## Model, metrics, and capacity decision

The residual hidden contribution preserves the learned function because every new outgoing hidden weight starts at zero. Adam state persists only while parameter shapes remain unchanged and resets after expansion. Macro-F1 is primary. Minority F1 averages the two least frequent inner-training classes. Underfitting is operationally identified when a function-preserving one-neuron probe improves validation Macro-F1 by at least $\delta_c=0.001$ after stagnation."""),
m(r"""## Fold-specific architecture selection

Selection is repeated inside every outer training partition so the test fold cannot influence capacity. Let $F^*$ be the best seed-mean validation Macro-F1. The selected capacity is the smallest $h$ satisfying $F(h)\geq F^*-0.001$. Candidates extend through 512; a boundary diagnostic blocks final reporting if at least three folds still select the largest candidate."""),
c("""ARCHITECTURE = {}
SELECTED = {}
BOUNDARY_STATUS = []
for dataset in DATASETS:
    raw, summary, selected = architecture_search(FOLDS[dataset.name], CONFIG)
    ARCHITECTURE[dataset.name] = (raw, summary)
    SELECTED.update(selected)
    root = Path('results')/dataset.name
    (root/'raw').mkdir(parents=True, exist_ok=True); (root/'summary').mkdir(parents=True, exist_ok=True)
    raw.to_csv(root/'raw'/'architecture_search.csv', index=False)
    summary.to_csv(root/'summary'/'architecture_search.csv', index=False)
    display(pd.DataFrame({'outer_fold': range(1,6), 'selected_hidden_units': [selected[(dataset.name,i)] for i in range(1,6)]}))
    BOUNDARY_STATUS.append(architecture_boundary_status(selected, CONFIG, dataset.name))
    plot_architecture_search(summary, selected,
        f'report/figures/{dataset.name}/baseline_architecture.pdf')"""),
m(r"""## Validation-only growth sensitivity

The $3\times3$ probe grid uses one declared seed across all five inner training/validation splits. It never evaluates outer test folds and is stored separately from canonical results."""),
c("""SENSITIVITY = {}
for dataset in DATASETS:
    frame = run_capacity_sensitivity(FOLDS[dataset.name], CONFIG, SELECTED, Path('results')/dataset.name)
    SENSITIVITY[dataset.name] = frame
    plot_sensitivity(frame, f'report/figures/{dataset.name}/capacity_sensitivity.pdf')"""),
m("""## Final controlled experiments

The fixed update-budget control matches gradient-update counts, not exact computation. Dynamic all-classes starts with zero hidden units and shares the growth policy with dynamic incremental learning; only class availability differs. Runs that exhaust `max_epochs` are recorded as failed and block summarisation."""),
c("""ALL_METHODS, ALL_STAGES, SUMMARIES = {}, {}, {}
for dataset in DATASETS:
    root = Path('results')/dataset.name
    methods, stages, orders = run_dataset_experiments(FOLDS[dataset.name], CONFIG, SELECTED, root)
    main, per_class, stats, efficiency = summarise_results(methods, dataset.class_names)
    export_summaries(main, per_class, stages, stats, efficiency, root/'summary')
    ALL_METHODS[dataset.name], ALL_STAGES[dataset.name] = methods, stages
    SUMMARIES[dataset.name] = (main, per_class, stats, efficiency)
    display(Markdown(f'### {dataset.name}')); display(main, efficiency, stats)"""),
m("## Incremental progression, capacity, and compute\n\nOne-neuron probes follow the assignment specification. Efficiency tables expose the resulting update, example, capacity, and wall-clock costs."),
c("""for dataset in DATASETS:
    name=dataset.name; figroot=f'report/figures/{name}'
    plot_incremental(ALL_STAGES[name], f'{figroot}/incremental_progression.pdf', f'{figroot}/network_growth.pdf')
    plot_dynamic_capacity_performance(ALL_METHODS[name], f'{figroot}/dynamic_capacity_performance.pdf')
plot_primary_effects(ALL_METHODS, 'report/figures/primary_effects.pdf')
plot_per_class_effects(ALL_METHODS,
    {d.name: d.class_names for d in DATASETS},
    {d.name: list(d.y.value_counts().sort_values().index) for d in DATASETS},
    'report/figures/per_class_effects.pdf')
plot_performance_cost({name: result[3] for name, result in SUMMARIES.items()},
    'report/figures/performance_cost.pdf')"""),
m("## Per-class and confusion analysis\n\nPer-run row-normalised confusion matrices are averaged so every fold-seed run has equal influence. Cell-wise standard deviations are exported beside each figure."),
c("""for dataset in DATASETS:
    name=dataset.name; main, per_class, _, _=SUMMARIES[name]; figroot=f'report/figures/{name}'
    plot_confusions(ALL_METHODS[name], dataset.class_names, figroot)"""),
m("## Cross-dataset interpretation\n\nRaw observations and probabilities are not pooled across data sets. Effect direction, uncertainty, minority performance, and computational cost support the cross-domain interpretation."),
c("""combined = Path('results/combined'); combined.mkdir(parents=True, exist_ok=True)
pd.concat([v[0].assign(dataset=k) for k,v in SUMMARIES.items()]).to_csv(combined/'main_results.csv', index=False)
pd.concat([v[2].assign(dataset=k) for k,v in SUMMARIES.items()]).to_csv(combined/'statistical_results.csv', index=False)
pd.concat([v[3].assign(dataset=k) for k,v in SUMMARIES.items()]).to_csv(combined/'efficiency_results.csv', index=False)
(combined/'manifest.json').write_text(json.dumps({
    'configuration_fingerprint': config_fingerprint(CONFIG),
    'datasets': [d.name for d in DATASETS]}, indent=2))
display(pd.read_csv(combined/'main_results.csv'), pd.read_csv(combined/'efficiency_results.csv'))"""),
m("""## Report export and submission guard

Tables are generated from completed outputs, while substantive interpretation remains manually authored. Fingerprints, method membership, architecture boundaries, failures, identity fields, and the HTTPS repository link are validated before producing a submission-named PDF."""),
c("""MANIFESTS = {d.name: load_manifest(Path('results')/d.name/'raw'/'manifest.json') for d in DATASETS}
validate_result_provenance(SUMMARIES, MANIFESTS, CONFIG, BOUNDARY_STATUS)
export_generated_results(SUMMARIES, 'report/canonical_results.tex')
output = compile_report('report', SUBMISSION)
print(f'Generated {output}')""")]

nb.metadata.kernelspec = {"display_name":"Python 3","language":"python","name":"python3"}
nb.metadata.language_info = {"name":"python","version":"3"}
nbf.write(nb, "analysis.ipynb")
