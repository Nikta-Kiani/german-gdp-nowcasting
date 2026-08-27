# Data Access and Privacy

The modelling panel used from Part I onward has 585 monthly series after
deduplication and the discontinued-series filter, spanning January 1991 to
December 2025. The licensed source workbook is not distributed.

The source workbook combines ifo and Macrobond series with official German and
European macroeconomic releases. These data may be subject to institutional or
commercial redistribution restrictions. Derived panels, metadata exports, and
model outputs are excluded for the same reason.

## Expected inputs

The preparation notebooks expect:

- a source workbook named `ifoCAST_DATA.xlsx`;
- sheets containing the raw monthly panel, transformation metadata, and
  first-release GDP vintages;
- optional review workbooks used to document publication lags and approved
  indicator changes.

Later pipeline stages consume files produced by the preparation notebooks:

```text
data/
├── panel/
│   ├── data_df.csv
│   └── data_transformed.csv
├── metadata/
│   ├── data_dict_catalog.csv
│   ├── data_dict_enriched.csv
│   └── pub_lag_map.csv
└── qa/
    ├── deduplication_decisions.csv
    ├── near_duplicate_pairs.csv
    └── stationarity_report.csv
```

All of these paths are ignored by Git.

## Configure private paths

The package defaults to local `data/` and `outputs/` directories. Private files
can remain elsewhere by setting environment variables:

```bash
export GERMAN_GDP_NOWCASTING_DATASET_XLSX="/private/path/ifoCAST_DATA.xlsx"
export GERMAN_GDP_NOWCASTING_DATA_DIR="/private/path/data"
export GERMAN_GDP_NOWCASTING_OUTPUTS_DIR="/private/path/outputs"
```

Optional review files can be configured independently:

```bash
export GERMAN_GDP_NOWCASTING_SUPERVISOR_KEPT_XLSX="/private/path/indicators_kept.xlsx"
export GERMAN_GDP_NOWCASTING_SUPERVISOR_DROPPED_XLSX="/private/path/indicators_dropped.xlsx"
```

## What can be reproduced without the private data?

The complete package can be imported, and the automated test suite runs on
small synthetic panels. Reproducing the thesis estimates, tables, and figures
requires access to the original licensed data.
