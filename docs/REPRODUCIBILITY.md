# Reproducibility Guide

## Environment

The repository was validated with Python 3.13.5. From the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[full]"
```

The `full` extra installs visualization, notebook, and Bayesian-SV dependencies.
Exact tested versions are also recorded in `requirements.txt`.

## Fast verification

The tests use synthetic data and do not access private datasets or overwrite
thesis artifacts:

```bash
python -m unittest discover -s tests -t .
```

They cover package imports, path configuration, publication-lag masking,
quarterly aggregation, indicator-selection smoothing, forecast metrics, and
no-look-ahead feature construction for XGBoost and the factor-augmented MLP.

## Research workflow

The notebooks document the analysis in order:

1. `01_data_understanding.ipynb`
2. `02_data_preparation.ipynb`
3. `03_elastic_net_selection.ipynb`
4. `04_pls_selection_and_stability.ipynb`
5. `05_build_model_inputs.ipynb`
6. `06_dfm_nowcasting.ipynb`
7. `07_xgboost_nowcasting.ipynb`

For scripted full reruns after the private paths are configured:

```bash
python scripts/pipelines/orchestrators/01_selection.py
python scripts/pipelines/orchestrators/02_dfm.py
python scripts/pipelines/orchestrators/03_dfm_suite.py
python scripts/pipelines/orchestrators/05_xgb.py
python -m german_gdp_nowcasting.models.mlp.mlp_utils
python scripts/pipelines/orchestrators/09_finalize.py
```

These commands are computationally expensive. They refit expanding-window
models over the full 2011–2025 evaluation sample and write generated artifacts
under `outputs/`, which is intentionally ignored by Git.

## Real-time safeguards

The empirical design protects against look-ahead bias by:

- using expanding training windows;
- evaluating against first-release GDP;
- masking each monthly indicator according to its publication lag;
- filling only the ragged edge visible at each forecast origin;
- tuning models before the evaluation window or within the available history;
- applying the same information set and evaluation dates across model classes.

## Scope of automated verification

The fast suite validates deterministic building blocks and interface contracts.
It does not rerun the 60-quarter backtests, Bayesian MCMC, or the full figure
pipeline. Those results require the licensed data and substantially more
compute.
