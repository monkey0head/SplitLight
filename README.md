![Teaser](streamlit_ui/pictures/teaser.png)

![PyData](https://img.shields.io/badge/data-Pandas-EE4C2C)
[![Hydra](https://img.shields.io/badge/config-Hydra-ADD8E6)](https://github.com/facebookresearch/hydra)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-%238a91faff)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE) 
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-ff69b4)](https://www.python.org/)

# SplitLight: Explore your RecSys dataset and split

SplitLight is a lightweight framework for auditing recommender-system datasets and evaluating splitting results. Its main goal is to help you produce trustworthy splits and justify split choices via transparent, data-driven diagnostics. 
SplitLight could be used in Jupyter/Python scripts for comprehensive analysis and offers easy-to-use Streamlit UI for interactive exploration, health checks, and side-by-side comparisons. 


## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH="$(pwd):$PYTHONPATH"
export SEQ_SPLITS_DATA_PATH=$(pwd)/data
```
- Requirements file: `requirements.txt`
- Your datasets live under `data/` (see layout below).

## Data layout

SplitLight expects each dataset under `data/<DatasetName>/` with either a `raw.csv` (original schema) or `preprocessed.csv` (standard schema).

- `raw.csv` (optional): original column names are defined in `runs/configs/dataset/<DatasetName>.yaml`
- `preprocessed.csv`: standardized columns: `user_id`, `item_id`, `timestamp` (seconds)
- After splitting, a per-split subfolder contains: `train.csv`, `validation_input.csv`, `validation_target.csv`, `test_input.csv`, `test_target.csv`

Example:

```bash
data/
  Beauty/
    raw.csv                # optional
    preprocessed.csv
    leave-one-out/         # example split folder
      train.csv
      validation_input.csv
      validation_target.csv
      test_input.csv
      test_target.csv
  Diginetica/
    preprocessed.csv
    GTS-q09-val_by_time-target_last/
      train.csv
      validation_input.csv
      validation_target.csv
      test_input.csv
      test_target.csv
```

## Streamlit UI

Launch the app for interactive dataset and split audits.

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
export SEQ_SPLITS_DATA_PATH=$(pwd)/data
streamlit run SplitLight.py
```

For better experience, zoom out the page to adjust to your screen size.  

What you can explore:
- Core and temporal statistics per subset and vs. reference
- Interactions distribution over time
- Repeated consumption patterns (non-unique and consecutive repeats)
- Temporal leakage: shared interactions, overlap, and “leakage from future”
- Cold-start exposure of users and items
- Compare splits side-by-side and analyze time-gap deltas between input and target

## What SplitLight checks

| Category                | Description                                                                                                                                                                                          |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dataset and Subsets** | Analyze raw and preprocessed data in terms of core and temporal statistics and compare. Identify repeated consumption patterns. Visualize interactions distribution over time.                       |
| **Subsets and Splits**  | Analyze splitted data in terms of core and temporal statistics and compare subsets with full data. Identify and visualize presence of data leakage. Quantify and visualize user and item cold start. |
| **Compare splits**      | Compare different splits in terms of core and temporal statistics. Identify distribution shifts for target subset.                                                                 |


> You can also run these checks manually using functions from the [`src/stats`](src/stats)  module for custom analyses or integration into your own pipelines (see [`demo notebook`](demo.ipynb)).

## Streamlit Summary Page
![Summary](streamlit_ui/pictures/summary.png)
The Summary page in the Streamlit UI provides a high-level overview of dataset and split health. It aggregates key diagnostics into a single dashboard, helping you quickly identify quality issues and distribution imbalances.

### What it provides
- Instant snapshot of dataset quality and split integrity  
- Compact visualization of core, temporal, and leakage statistics  
- Color-coded signals to highlight potential issues at a glance 

Each metric is assigned a health status based on configurable thresholds:
- 🟢 **Good** — within expected bounds  
- 🟡 **Need Attention** — mild irregularity detected  
- 🔴 **Warning** — potential data issue or leakage risk 

### Configuration
Thresholds and color rules for the Summary view can be customized in  
[`streamlit_ui/config/summary.yml`](streamlit_ui/config/summary.yml).

## Configuration

- UI thresholds and labels: `streamlit_ui/config/summary.yml`
- Dataset schemas: `runs/configs/dataset/*.yaml`

## CLI Utilities
These CLI tools are provided to illustrate a complete pipeline for preprocessing and creating splits.

### Preprocess

Standardize and clean your raw interaction logs.

```bash
export SEQ_SPLITS_DATA_PATH=$(pwd)/data
python runs/preprocess.py +dataset=Beauty
```

- Config: `runs/configs/preprocess.yaml`
- Dataset column mapping: `runs/configs/dataset/<DatasetName>.yaml`
- Output: `data/<DatasetName>/preprocessed.csv`

### Split

Split your dataset using `Leave-One-Out (LOO)` or `Global Time Split (GTS)` strategies.
See [`src/splits.py`](src/splits.py) for implementation details.

```bash
# Leave-one-out (LOO)
python runs/split.py split_type=leave-one-out

# Global time split (GTS)
python runs/split.py \
  dataset=Beauty \
  split_type=global_timesplit \
  split_params.quantile=0.9 \
  split_params.validation_type=by_time \
  split_params.target_type=last
```

- Common options:
  - `dataset=<Name>`: must match a YAML in `runs/configs/dataset/`
  - `remove_cold_users=true|false`
  - `remove_cold_items=true|false`
- GTS options:
  - `split_params.quantile` (required) — global time threshold
  - `split_params.validation_type` — `by_time` | `by_user` | `last_train_item`
  - `split_params.validation_size` — number of users for `by_user`
  - `split_params.validation_quantile` — time for `by_time`
  - `split_params.target_type` — `all` | `first` | `last` | `random`

## FAQ

- Q: Can I use Parquet files?  
  A: Yes, `.csv` and `.parquet` are available. In the UI home page, choose `.parquet` or both.
- Q: Do I need `raw.csv`?  
  A: No. You can provide only `preprocessed.csv` in the standard schema.
- Q: What time unit is `timestamp`?  
  A: Seconds since epoch.

## Cite

If you use SplitLight in research or production, please cite this repository.
