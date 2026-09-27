# Incremental Class Learning — Dry Bean and Steel Plates

The canonical and only experiment interface is `analysis.ipynb`. It evaluates UCI Dry Bean and Steel Plates Faults with five-fold outer cross-validation and a fold-specific inner training/validation split, fold-specific architecture selection, five optimisation seeds, a fixed update-budget-matched control, and a dynamic all-classes control.

## Run

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute analysis.ipynb --inplace
```

The first connected run downloads either missing dataset through `ucimlrepo`; cached CSV files are used thereafter. The notebook regenerates dataset-specific tables beneath `results/`, figures beneath `report/figures/`, generated LaTeX results, and `report/report.pdf`.

The full configuration is computationally expensive because architecture search is nested independently within five outer folds for two datasets. For a smoke run, reduce the seed tuple, architecture candidates, and maximum epochs in the configuration cell; restore the committed configuration for final evidence.

Run the canonical notebook from a clean kernel with:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute analysis.ipynb \
  --output analysis.ipynb --ExecutePreprocessor.timeout=7200
```

The notebook generates numerical report content from completed results and compiles the PDF automatically. It refuses to create a submission PDF when runs fail, configurations are stale, architecture search remains unresolved, or submission metadata still contains placeholders. Populate `SUBMISSION` in the notebook before the final clean run. Do not submit the currently committed PDF: it predates the dynamic all-classes control.

Implementation details live in `src/`; no command-line experiment drivers are required.
