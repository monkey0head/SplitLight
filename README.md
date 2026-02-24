![Teaser](streamlit_ui/pictures/splitlight.png)


# 🌟 SplitLight: Explore Your RecSys Dataset and Split

<a href="https://arxiv.org/abs/2602.19339"><img src="https://img.shields.io/badge/arXiv-2602.19339-b31b1b.svg" height=22.5><a>
[![Cite](https://img.shields.io/badge/Cite-BibTeX-EE4C2C)](https://github.com/monkey0head/SplitLight?tab=readme-ov-file#-citation)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff69b4)](https://streamlit.io/)
[![Hydra](https://img.shields.io/badge/Config-Hydra-%238a91faff)](https://github.com/facebookresearch/hydra)
[![License](https://img.shields.io/badge/License-MIT-ADD8E6)](./LICENSE) 
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-green)](https://www.python.org/)

**SplitLight** is a lightweight framework for auditing recommender-system datasets and evaluating splitting results. Its main goal is to help you produce **trustworthy data preprocessing and splits** and justify split choices via transparent, data-driven diagnostics.
**SplitLight** can be used in Jupyter/Python scripts for comprehensive analysis and offers an easy-to-use Streamlit UI for interactive exploration, health checks, and side-by-side comparisons.

### Why SplitLight?

- **Trustworthy evaluation** — Poor or inconsistent train/validation/test splits lead to overoptimistic metrics and non-reproducible research. SplitLight helps you detect leakage, cold-start issues, and distribution shifts before training.
- **Transparent diagnostics** — Instead of treating the split as a black box, you get concrete stats: shared interactions, temporal overlap, leaked targets, cold user/item shares, and temporal deltas between input and target.
- **Flexible workflow** — Use the Streamlit app for ad-hoc audits, or call `src/stats` and `src/splits` from your own pipelines and notebooks (see the [demo notebook](demo.ipynb)).


<div align="center">
    <a >
        <img src="/home/jovyan/gusak/SplitLight/streamlit_ui/pictures/pipeline.png" width="100%">
    </a>
    <p>
        <i> SplitLight in a data-preparation pipeline. From the raw dataset to split subsets, SplitLight audits data, flags problems, and enables side-by-side comparison of alternative splits to justify the selected evaluation protocol.</i>
    </p>
</div>

> [!NOTE]
> See short [**video walkthrough**](https://drive.google.com/file/d/15ZSKai7dYXBVmcqPIuV4qsM9bbsRxNY4/view) of SplitLight motivation and usage.

## Quick Start

```bash
pip install -r requirements.txt
export PYTHONPATH="$(pwd):$PYTHONPATH"
export SEQ_SPLITS_DATA_PATH=$(pwd)/data
```
- Requirements file: `requirements.txt`
- Your datasets live under `data/` (see layout below).

Install the requirements and set the environment variables. Then, run the Streamlit as described [here](#streamlit-ui) to get the data overview or start jupyter notebook and explore the data and splits in depth (see the [demo notebook](demo.ipynb)).

## Data Layout

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

## What SplitLight Checks

| Category                | Description                                                                                                                                                                                          |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dataset and Subsets** | Analyze raw and preprocessed data in terms of core and temporal statistics and compare. Identify repeated consumption patterns. Visualize interactions distribution over time.                       |
| **Subsets and Splits**  | Analyze split data in terms of core and temporal statistics and compare subsets with full data. Identify and visualize presence of data leakage. Quantify and visualize user and item cold start. |
| **Compare splits**      | Compare different splits in terms of core and temporal statistics. Identify distribution shifts for target subset.                                                                 |


> You can also run these checks manually using functions from the [`src/stats`](src/stats)  module for custom analyses or integration into your own pipelines (see [`demo notebook`](demo.ipynb)).

## Streamlit Summary Page
![Summary](streamlit_ui/pictures/summary.png)
The Summary page in the Streamlit UI provides a high-level overview of dataset and split health. It aggregates key diagnostics into a single dashboard, helping you quickly identify quality issues and distribution imbalances.

### What It Provides
- Instant snapshot of dataset quality and split integrity  
- Compact visualization of core, temporal, and leakage statistics  
- Color-coded signals to highlight potential issues at a glance 

Each metric is assigned a health status based on configurable thresholds:
- 🟢 **Good** — within expected bounds  
- 🟡 **Need Attention** — mild irregularity detected  
- 🔴 **Warning** — potential data issue or leakage risk 


<div align="center"> <video src="https://github.com/user-attachments/assets/f2114376-311c-474b-a511-2ffe17fc59f1" controls style="max-width: 90%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"> Your browser does not support the video tag. <a href="https://github.com/user-attachments/assets/f2114376-311c-474b-a511-2ffe17fc59f1">Watch the demo video</a>. </video> <p style="font-size: 12px; color: #666; margin-top: 8px;"> ▶ Click to play the short SplitLight's Summary Dashboard showcase. </p> </div>

### Configuration
Thresholds and color rules for the Summary view can be customized in  
[`streamlit_ui/config/summary.yml`](streamlit_ui/config/summary.yml).


## Project Structure (Key Parts)

- **`src/stats/`** — Core diagnostics: `base` (core/temporal stats), `leaks`, `cold`, `duplicates`, `temporal`, `plots`. Use these in scripts or notebooks for custom analyses.
- **`streamlit_ui/pages/`** — Streamlit pages for load, Summary, core/temporal stats, repeated consumption, leakage, cold start, and split comparison.
- **`runs/`** — CLI entrypoints and Hydra configs: `preprocess.py`, `split.py`, `train_rs.py`; configs under `runs/configs/` (dataset, split, preprocess, train_rs, model).

## FAQ

- **Q: Can I use Parquet files?**  
  A: Yes. Both `.csv` and `.parquet` are supported. On the UI home page, choose the file format (e.g. `.parquet` or both).
- **Q: Do I need `raw.csv`?**  
  A: No. You can provide only `preprocessed.csv` in the standard schema (`user_id`, `item_id`, `timestamp`). `raw.csv` is optional when you want to run the preprocessing pipeline from raw logs.
- **Q: What time unit is `timestamp`?**  
  A: Seconds since epoch (Unix time). The preprocess step and all stats assume this; convert your timestamps before use if needed.
- **Q: I only have raw interaction logs. How do I start?**  
  A: (1) Add a dataset config under `runs/configs/dataset/<Name>.yaml` mapping your columns to `user_id`, `item_id`, `timestamp`. (2) Put `raw.csv` (or raw data) under `data/<DatasetName>/`. (3) Run your own preprocessing script or use example `python runs/preprocess.py +dataset=<DatasetName>` to get `preprocessed.csv`. (4) Run your split script or use example `python runs/split.py` to create a split, then open the Streamlit app or jupyter notebook (see [demo notebook](demo.ipynb)) to audit dataset and split.
- **Q: How do I use SplitLight in my own Python code?**  
  A: Use the **stats** API: import functions from `src.stats` (e.g. `leaks.get_leaks`, `cold.share_of_cold`, `base.base_stats) and call them on your DataFrames. See the [demo notebook](demo.ipynb) for examples.
- **Q: Why should I care about split quality?**  
  A: The split defines what you are actually evaluating. Leaky or inconsistent splits lead to overestimated metrics and results that don’t transfer to real deployment. SplitLight helps you document and justify your split choice and catch issues early.

## CLI Utilities For Experimenting
These CLI tools are provided to illustrate a complete pipeline for preprocessing and splitting datasets. The results of the preprocessing and splitting could be audited using the SplitLight. To train a sequential model on the split data and evaluate, how different data preprocessing and splitting strategies affect the model performance, use the example `python runs/train_rs.py`.

See [runs/README.md](runs/README.md) for more detailed explanation on CLI tools and experimental setup for splitting results in `/data` dir. 

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
python runs/split.py split_type=leave-one-out split_params.remove_cold_items=True

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

- Config: `runs/configs/split.yaml`
- Output: splits are saved under `data/<DatasetName>/<split_name>/`

### Train Recommender Model on Selected Data Split 

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
export SEQ_SPLITS_DATA_PATH=$(pwd)/data
python runs/train_rs.py dataset=Beauty split_name=leave-one-out
```
- Config: `runs/configs/train_rs.yaml`


## Contributing
We welcome and appreciate all forms of contributions to make SplitLight better! If you have ideas to improve SplitLight, please feel free to submit a Pull Request.

## Citation
If you use SplitLight in research or production, please consider citing our paper:

```bib
@misc{splitlight2026,
      title={SplitLight: An Exploratory Toolkit for Recommender Systems Datasets and Splits}, 
      author={Anna Volodkevich and Dmitry Anikin and Danil Gusak and Anton Klenitskiy and Evgeny Frolov and Alexey Vasilev},
      year={2026},
      eprint={2602.19339},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

---
<div align="center">
We welcome contributions from the community! 🤝
</div>
