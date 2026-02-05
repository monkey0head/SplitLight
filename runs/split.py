"""Make split to train, validation and test."""

import os
import sys

import hydra
import pandas as pd
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.splits import LeaveOneOutSplitter, GlobalTimeSplitter
from src.stats.base import base_stats


@hydra.main(version_base=None, config_path="configs", config_name="split")
def main(config):
    print(OmegaConf.to_yaml(config, resolve=True))

    data = pd.read_csv(os.path.join(config.data_path, config.dataset.name, "preprocessed.csv"))

    if config.split_type == "leave-one-out":
        splitter = LeaveOneOutSplitter()
        dir_name = os.path.join(
            config.splitted_data_path, config.dataset.name, config.split_type
        )
        train, validation_input, validation_target, test_input, test_target = (
            splitter.split(data)
        )

    elif config.split_type == "global_timesplit":
        splitter = GlobalTimeSplitter(**config.split_params)

        if config.split_params.quantile is not None:
            q = "q0" + str(config.split_params.quantile)[2:]
        else:
            raise ValueError(
                "'global_timesplit' split must be run with parameter 'quantile'"
            )

        dir_name = os.path.join(
            config.splitted_data_path,
            config.dataset.name,
            "GTS" + "-" + q +"-val_" + config.split_params.validation_type + "-target_" + config.split_params.target_type,
        )

        train, validation_input, validation_target, test_input, test_target = (
            splitter.split(data)
        )

    else:
        raise ValueError("Wrong type of splitter.")

    print(f'train\n', base_stats(train))
    print(f'validation_input\n', base_stats(validation_input))
    print(f'validation_target\n', base_stats(validation_target))
    print(f'test_input\n', base_stats(test_input))
    print(f'test_target\n', base_stats(test_target))

    if config.save_results:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        train.to_csv(os.path.join(dir_name, "train.csv"), index=False)
        validation_input.to_csv(
            os.path.join(dir_name, "validation_input.csv"), index=False
        )
        validation_target.to_csv(
            os.path.join(dir_name, "validation_target.csv"), index=False
        )
        test_input.to_csv(os.path.join(dir_name, "test_input.csv"), index=False)
        test_target.to_csv(os.path.join(dir_name, "test_target.csv"), index=False)


if __name__ == "__main__":
    main()
