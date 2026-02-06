"""
Preprocessing pipeline
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import OmegaConf, open_dict

import pandas as pd

from src.preprocess.pipeline import preprocess


@hydra.main(config_path="configs", config_name="preprocess")
def main(config):
    data_format = "csv"
    print(OmegaConf.to_yaml(config, resolve=True))
    data_path = os.environ["SEQ_SPLITS_DATA_PATH"]

    # os.chdir('../..')

    data = pd.read_csv(
        os.path.join(data_path, config.dataset.name, "raw.csv")
    )

    with open_dict(config):
        save_to_disk = config.prep_params.pop("save_to_disk")

    preprocess(
        data=data,
        **config.prep_params,
        **config.dataset.column_name,
        path_to_save=os.path.join(
            data_path, config.dataset.name, f"preprocessed{config.tag}.csv",
        )
        if save_to_disk
        else None,
    )


if __name__ == "__main__":
    main()
