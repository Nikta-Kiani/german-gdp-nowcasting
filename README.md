# Real-Time German GDP Nowcasting

### Data-driven indicator selection versus expert curation

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-20%20passing-brightgreen)](#testing)
[![Research](https://img.shields.io/badge/status-thesis%20complete-blue)](#research-design)
[![Data](https://img.shields.io/badge/data-licensed%20%7C%20not%20included-lightgrey)](docs/DATA.md)

> Can a transparent, real-time statistical selection rule match the indicator
> choices of professional forecasters when nowcasting German GDP?

This repository contains the econometric and machine-learning pipeline developed
for a master's thesis on German quarter-on-quarter GDP nowcasting. It compares
time-varying data-driven indicator selection with the fixed expert-curated
ifoCAST panel under one common pseudo-real-time evaluation design.

## Executive summary

German GDP is released only after the reference quarter has ended. Nowcasting
uses earlier monthly information—production, orders, trade, turnover, surveys,
and financial indicators—to estimate current-quarter growth.

The project screens **585 monthly indicators from 1991–2025** at **180 monthly
forecast origins from 2011–2025**. Every model uses expanding training windows,
first-release GDP targets, publication-lag masking, and a ragged-edge treatment
that prevents future information from leaking into historical forecasts.

The main result is not that one selection method always wins. Data-driven
selection matches expert curation on average, while their errors differ across
economic regimes. Combining structurally different forecasts is more robust
than searching for one universally best model.

## Research questions

1. Which monthly indicators carry stable, method-robust predictive information
   for German GDP, and how does that set change across economic regimes?
2. Does real-time statistical selection improve nowcast accuracy relative to
   univariate benchmarks and an expert-curated indicator set?
3. Can stochastic volatility deliver calibrated prediction intervals through
   the extreme GDP movements around COVID-19?

## Headline findings

- Targeted dynamic factor models achieve roughly **0.84–0.95 percentage points
  RMSFE**, compared with **2.41 percentage points** for the expanding AR(1)
  benchmark over the full evaluation period.
- Elastic Net and block-balanced data-driven panels slightly outperform the
  fixed expert panel in point estimates, but the differences are not
  statistically distinguishable.
- An equal-weight combination of expert and data-driven factor models is the
  strongest specification, at approximately **0.70 percentage points RMSFE**.
- Forecast rankings are regime-dependent: factor models gain most during the
  pandemic, while simpler break-robust autoregressive methods perform better in
  the low-growth period after 2022.
- Bayesian stochastic-volatility intervals cover **52 of 60 quarters (86.7%)**
  at a nominal 90% level and widen appropriately during the pandemic.

## Methods

The pipeline combines:

- Elastic Net, fixed-*k* targeted predictors, partial least squares, and
  frequency-smoothed selection;
- mixed-frequency dynamic factor models with Mariano–Murasawa aggregation;
- publication-lag masking and autoregressive ragged-edge filling;
- stochastic-volatility and time-varying-parameter DFM extensions;
- XGBoost with SHAP-guided feature pruning;
- a factor-augmented multilayer perceptron;
- AR(1), random-walk, rolling-window, and expert-set benchmarks;
- RMSFE, noise-to-signal ratios, Diebold–Mariano tests,
  Mincer–Zarnowitz regressions, and interval-coverage diagnostics.

## Repository structure

```text
german-gdp-nowcasting/
├── notebooks/                 # Seven ordered research notebooks
├── src/german_gdp_nowcasting/
│   ├── config/                # Portable path configuration
│   ├── selection/             # Indicator-selection methods
│   ├── models/
│   │   ├── dfm/               # Factor, SV, TVP, and ragged-edge models
│   │   ├── xgboost/           # Gradient-boosting benchmark
│   │   └── mlp/               # Factor-augmented neural network
│   └── visualization/         # Publication-quality plotting
├── scripts/
│   ├── pipelines/             # Reproducible pipeline entry points
│   └── experiments/           # Approved robustness analysis
├── tests/                     # Fast synthetic regression tests
├── docs/                      # Data and reproducibility guidance
├── pyproject.toml             # Package and dependency metadata
└── requirements.txt           # Exact tested dependency versions
```

The Streamlit dashboard and LaTeX thesis are deliberately maintained separately.
Private data, generated outputs, local environments, and development artifacts
are excluded from this repository.

## Quick start

The project was validated with Python 3.13.5.

```bash
git clone <repository-url>
cd german-gdp-nowcasting

python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[full]"
```

To open the narrative analysis:

```bash
jupyter lab notebooks/
```

## Data configuration

The licensed source data are not distributed. Configure private paths before
running the empirical pipeline:

```bash
export GERMAN_GDP_NOWCASTING_DATASET_XLSX="/private/path/ifoCAST_DATA.xlsx"
export GERMAN_GDP_NOWCASTING_DATA_DIR="/private/path/data"
export GERMAN_GDP_NOWCASTING_OUTPUTS_DIR="/private/path/outputs"
```

See [Data Access and Privacy](docs/DATA.md) for expected files and optional
review-workbook configuration.

## Running the pipeline

After preparing the data with notebooks 01–02, the main scripted sequence is:

```bash
python scripts/pipelines/orchestrators/01_selection.py
python scripts/pipelines/orchestrators/02_dfm.py
python scripts/pipelines/orchestrators/03_dfm_suite.py
python scripts/pipelines/orchestrators/05_xgb.py
python -m german_gdp_nowcasting.models.mlp.mlp_utils
python scripts/pipelines/orchestrators/09_finalize.py
```

Full reruns are computationally expensive and write only to the ignored
`outputs/` directory. See the
[Reproducibility Guide](docs/REPRODUCIBILITY.md) before running them.

## Testing

The test suite uses synthetic data and does not access or overwrite thesis data:

```bash
python -m unittest discover -s tests -t .
```

The current suite contains 20 tests covering imports, path configuration,
publication lags, quarterly aggregation, selection smoothing, forecast metrics,
and no-look-ahead feature construction.

## Reproducibility and limitations

- The public code and synthetic tests are reproducible without private data.
- Exact thesis estimates require the licensed ifo/Macrobond source workbook.
- The predictor panel is a static historical extract rather than a complete
  vintage-by-vintage indicator database.
- The evaluation contains 60 quarters, including only 16 post-COVID quarters.
- Full expanding-window backtests and Bayesian MCMC require substantial compute.

## Citation

If this repository supports your research, please cite the accompanying master's
thesis:

*Nowcasting and Indicator Selection in a Data-Rich Environment: An Application
to German GDP Growth.*
