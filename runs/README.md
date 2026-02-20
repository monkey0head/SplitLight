# Reproducing Experiments

SplitLight uses **Hydra-based configs** and command-line overrides to run
preprocessing, data splitting, and model training.
All experiments in the paper can be reproduced by varying a small number of
parameters at each stage.

1. [Preprocessing](#1-preprocessing)
2. [Splitting](#2-splitting)
3. [How to run model training](#3-model-training)

## 1. Preprocessing

Preprocessing constructs user interaction sequences, applies standard
filtering (e.g. 5-core filtering) and prepares data for splitting.

Example:

```bash
python runs/preprocess.py \
  +dataset=Movielens-1m
```

### Preprocessing parameters

The table below lists all available preprocessing parameters that should be used under `prep_params` key.

| Parameter             | Default | Description                                                                            |
| --------------------- | ------- | -------------------------------------------------------------------------------------- |
| `item_min_count`      | `5`     | Minimum number of interactions per item (k-core filtering).                            |
| `seq_min_len`         | `5`     | Minimum user sequence length after filtering.                                          |
| `core`                | `True`  | Enables iterative k-core filtering on users and items.                                 |
| `encoding`            | `True`  | Encodes users and items into contiguous integer IDs.                                   |
| `drop_consec_repeats` | `True`  | Removes consecutive duplicate items in user sequences. Set to `False` to keep repeats. |
| `filter_by_relevance` | `False` | Optional relevance-based filtering (not used).                                         |
| `save_to_disk`        | `True`  | Saves preprocessed data to disk.                                                       |
| `shuffle_collision`   | `False` | Randomly shuffles interactions with identical timestamps.                              |                                           |
| `users_sample`        | `null`  | Optional subsampling of users (integer or fraction of all users).                                              |
| `seed`                | `42`    | Random seed (used for timestamp shuffling and user sampling). 

All preprocessing options are specified in the [dedicated config](runs/config/preprocess.yaml)

Examples:

```bash
# Shuffle equal timestamps
python runs/preprocess.py \
  dataset=Movielens-1m \
  prep_params.shuffle_collision=True \
  tag=shuffle
```

The optional `tag` argument is used to distinguish derived datasets on disk and
must be reused at the splitting stage.

## 2. Splitting

Splitting is performed **after preprocessing**.

Got it — we’ll describe `split_type` separately, in prose, without including it in the table:

---

Splitting is performed **after preprocessing**.

* **`split_type`**: This is a **top-level configuration key** that specifies the overall splitting strategy. Available options are:

  * `global_timesplit` – splits data based on a global time quantile, separating older interactions for training and newer ones for testing.
  * `leave-one-out` – reserves the last interaction of each user as the test set, with the rest used for training.

All split-related parameters are defined under the `split_params` key.

| Parameter             | Default            | Description                                                                    |
| --------------------- | ------------------ | ------------------------------------------------------------------------------ |
| `quantile`            | `0.9`              | Global time quantile used to separate train/test (e.g. 0.8, 0.9, 0.95, 0.975). |
| `validation_quantile` | `${quantile}`      | Quantile used to form validation after the test split.                         |
| `validation_type`     | `by_user`          | Validation strategy: `by_user`, `by_time`, `last_train_item`.                  |
| `validation_size`     | `100`              | Number of validation users (only for `by_user`).               |
| `remove_cold_users`   | `False`            | Removes users unseen in training.                                 |
| `remove_cold_items`   | `True`             | Removes items unseen in training. Set to `False` to keep cold items.           |
| `target_type`         | `first`            | Prediction target for GTS: `first`, `last`, or `all`.                          |
| `tag`                 | `null`             | Identifies preprocessing variant to split (e.g. `shuffle`).     |

Complete data splitting parameters are defined in the [split config](runs/config/split.yaml)


### Example: Leave-One-Out (LOO)

```bash
python runs/split.py \
  dataset=Movielens-1m \
  split_type=leave-one-out
```

### Example: Global Time Split (GTS)

Last-item prediction with time-based validation:

```bash
python runs/split.py \
  dataset=Movielens-1m \
  split_type=global_timesplit \
  split_params.validation_type=by_time \
  split_params.target_type=last
```

### Example: Using alternative preprocessing variants

To run splits on modified preprocessing outputs
(e.g. shuffled timestamps), pass the same `tag`:

```bash
python runs/split.py \
  dataset=Zvuk \
  split_type=leave-one-out \
  tag=shuffle
```

## 3. Model Training

Training is performed with `runs/train_rs.py`.
Each run specifies several parameters:

| Parameter                     | Default                                | Description                                 |
| ----------------------------- | -------------------------------------- | ------------------------------------------- |
| **General / Experiment**      |                                        |                                             |
| `dataset`                     | `Beauty`                                 | Dataset name                                |
| `model`                       | `SASRec`                                 | Model name:    `SASRec`, `RNN`, `BERT4Rec`                               |
| `split_name`                  | `leave-one-out`                          | Name of data split directory                          |
| `random_state`                | `42`                                     | Seed for reproducibility                    |
| `metrics_save_dir`            | `results`                                | Directory to save metrics                   |
| `load_if_possible`            | `True`                                   | Load existing checkpoints if available      |
| 

Full model and training parameters are defined in the [training config](runs/config/train_rs.yaml)

Example: training SASRec with a single split and seed:

```bash
python runs/train_rs.py -m \
  dataset=Movielens-1m \
  model=SASRec \
  split_name=leave-one-out \
  random_state=0
```

### Multiple splits and random seeds

Hydra supports comma-separated values:

```bash
python runs/train_rs.py -m \
  dataset=Movielens-1m \
  model=SASRec \
  split_name=\
leave-one-out,\
leave-one-out-no_cold_items,\
GTS-q0.9-val_by_time-target_last \
  random_state=0,1,2,3,4
```

Each `(split_name, random_state)` combination is trained independently.