# Reproducibility

## Environment

Validated with Python 3.13.5. From the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[full]"
```

The `full` extra installs visualisation, notebook, and NumPyro (SV) dependencies. Exact versions are in `requirements.txt`.

## Fast verification

The tests use synthetic data. They do not read private files or overwrite thesis artefacts:

```bash
python -m unittest discover -s tests -t .
```

They cover package imports, path configuration, publication-lag masking, the release-block freeze used in the post-COVID counterfactual, quarterly aggregation, selection smoothing, RMSFE / Diebold–Mariano helpers, the Hansen–Lunde–Nason model confidence set, and no-look-ahead features for XGBoost and the factor-augmented MLP.

## Notebooks

The notebooks document the analysis in order. They load saved CSVs when those files exist.

1. `01_data_understanding.ipynb`
2. `02_data_preparation.ipynb`
3. `03_elastic_net_selection.ipynb`
4. `04_pls_selection_and_stability.ipynb`
5. `05_build_model_inputs.ipynb`
6. `06_dfm_nowcasting.ipynb`
7. `07_xgboost_nowcasting.ipynb`

Part I selects on completed-quarter aggregates. Publication lags and the ragged edge are applied in Part II, not by dropping series from the selection matrices.

## Scripted reruns

After the private paths are set (see [DATA.md](DATA.md)) and notebooks 01–02 have written the prepared panel:

```bash
python scripts/pipelines/orchestrators/01_selection.py
python scripts/pipelines/orchestrators/02_dfm.py
python scripts/pipelines/orchestrators/03_dfm_suite.py
python scripts/pipelines/orchestrators/05_xgb.py
python -m german_gdp_nowcasting.models.mlp.mlp_utils
python scripts/pipelines/orchestrators/09_finalize.py
```

These commands refit expanding-window models over 2011–2025 and write to `outputs/`, which Git ignores.
The DFM suite includes the ifoCAST, stochastic-volatility, block-balanced,
equal-weight-combination, and TVP benchmarks required by the final tables.

The post-COVID DFM-EN release-block attribution reuses the saved observed path and refits only the two hybrid information sets (hard activity frozen; non-hard complement frozen):

```bash
python scripts/pipelines/dfm/run_release_block_counterfactual.py
```

It writes the four-state forecast paths, quarter-level and mean Shapley loss decompositions, and the PDF figure under `outputs/nowcasting/`. The experiment is a fitted-model accounting identity for 2022Q1–2025Q4 ($N=16$). It does not show that official hard releases are generally harmful; bootstrap intervals for the hard-block contribution include zero.

## Real-time protocol

- Expanding training windows; the target quarter never enters estimation.
- First-release GDP, never a revised vintage, as the scoring target.
- Per-series publication-lag mask at each monthly origin.
- Univariate AR fill only on the ragged edge visible at that origin.
- The same origin grid and information set across model classes.

Predictor revisions are not simulated. The panel is a static historical extract.

## What the tests do not cover

The fast suite does not rerun the 60-quarter backtests, the SV sampler, or the figure pipeline. Those require the licensed workbook and substantially more compute.
