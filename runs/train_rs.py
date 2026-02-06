"""
Run experiment.
"""

import os
import time

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from clearml import Task
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (DeviceStatsMonitor, EarlyStopping, ModelCheckpoint,
                                         ModelSummary, TQDMProgressBar)
from torch.utils.data import DataLoader

from rs_src.datasets import (CausalLMDataset,
                          CausalLMPredictionDataset,
                          MaskedLMDataset,
                          MaskedLMPredictionDataset,
                          PaddingCollateFn)
from rs_src.models import RNN, SASRec
# from rs_src.models import RNN, BERT4Rec, SASRec
from rs_src.modules import SeqRec
from rs_src.metrics import Evaluator
from rs_src.postprocess import preds2recs

from run_utils import reset_peak_memory_stats, log_peak_memory_stats, fix_seeds

@hydra.main(version_base=None, config_path="configs", config_name="train_rs")
def main(config):

    print(OmegaConf.to_yaml(config))

    if hasattr(config, 'cuda_visible_devices'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)

    reset_peak_memory_stats()

    if hasattr(config, 'clearml_project_name') and config.clearml_project_name is not None:
        task = Task.init(project_name=config.clearml_project_name, task_name=config.clearml_task_name,
                        reuse_last_task_id=False)
        task.connect(OmegaConf.to_container(config))
    else:
        task = None

    # fix seeds only after task init as clearml task also 
    # initiates random seed and results may be different without it
    fix_seeds(config.random_state)

    # TO DO: move to passing train, test/validation input and targets dataframes, do not split and concat inside run script
    train, validation_input, validation_target, test_input, test_target, item_count = prepare_data(config)
    validation_full = pd.concat([validation_input, validation_target], axis=0)
    
    train_loader, eval_loader = create_dataloaders(train, validation_full, config)

    model = create_model(config, item_count=item_count)
    trainer, seqrec_module = training(model, train_loader, eval_loader, config, task)

    if config.calc_val_metrics:
        recs_validation = predict(trainer, seqrec_module, validation_input, config, task, prefix='val')
        evaluate(recs_validation, validation_target, train, task, config, prefix='val')

    metrics_test = None
    if config.calc_test_metrics:
        recs_test = predict(trainer, seqrec_module, test_input, config, task, prefix='test')
        metrics_test = evaluate(recs_test, test_target, train, task, config, prefix='test')

    log_peak_memory_stats(task)
    
    if task is not None:
        task.close()
        
    if metrics_test is not None:
        metrics_df = pd.Series(metrics_test).to_frame().T
        metrics_df['random_state'] = config.random_state

        save_dir = os.path.join(
            config.save_dir,
            config.dataset.name,
            config.split_name,
            config.model.model_class,
        )

        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_dir, "test_metrics.csv")

        if os.path.exists(save_path):
            metrics_df.to_csv(save_path, mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(save_path, index=False)

        return metrics_test[config.optimize_metric]
    return None
    

def prepare_data(config):
    data_task = None
    if hasattr(config, 'clearml_data_project_name') and config.clearml_data_project_name is not None:
        data_task = Task.get_task(project_name=config.clearml_data_project_name,
                                    task_name=config.clearml_data_task_name)
    else:
        seq_rec_data_path = os.getenv("SEQ_SPLITS_DATA_PATH")
        data_path = os.path.join(
            seq_rec_data_path, config.dataset.name, config.split_name)

    subsets = {}
    max_item_id = 0
    for subset in ['train', 'validation_input', 'validation_target', 'test_input', 'test_target']:
        if data_task is not None:
            subset_path = data_task.artifacts[subset].get_local_copy()
        else:
            subset_path = os.path.join(data_path, f'{subset}.csv')
        subsets[subset] = pd.read_csv(subset_path)
        print(f'{subset} shape', subsets[subset].shape)
        # index 1 is used for masking value (SASRec); index 0,1 for BERT4Rec
        if config.model.model_class in ['SASRec', 'RNN']:
            subsets[subset].item_id += 1
        if config.model.model_class == 'BERT4Rec':
            subsets[subset].item_id += 2
        max_item_id = max(max_item_id, subsets[subset]['item_id'].max())
    
    return subsets['train'], subsets['validation_input'], subsets['validation_target'], subsets['test_input'], subsets['test_target'], max_item_id + 1


def create_dataloaders(train, validation, config):
    validation_size = config.dataloader.validation_size
    validation_users = validation.user_id.unique()
    if validation_size and (validation_size < len (validation_users)):
        validation_users = np.random.choice(validation_users, size=validation_size, replace=False)
        validation = validation[validation.user_id.isin(validation_users)]

    if config.model.model_class in ['SASRec', 'RNN']:
        train_dataset = CausalLMDataset(train, **config.model.dataset_params)
        eval_dataset = CausalLMPredictionDataset(
            validation, max_length=config.model.dataset_params.max_length, validation_mode=True)
    elif config.model.model_class == 'BERT4Rec':
        train_dataset = MaskedLMDataset(train, **config.model.dataset_params)
        eval_dataset = MaskedLMPredictionDataset(
            validation, max_length=config.model.dataset_params.max_length, validation_mode=True)
    else:
        raise ValueError(f"Unknown model_class: {config.model.model_class}. Supported: SASRec, RNN, BERT4Rec.")

    train_loader = DataLoader(train_dataset, batch_size=config.dataloader.batch_size,
                              shuffle=True, num_workers=config.dataloader.num_workers,
                              collate_fn=PaddingCollateFn(left_padding=False))
    eval_loader = DataLoader(eval_dataset, batch_size=config.dataloader.test_batch_size,
                             shuffle=False, num_workers=config.dataloader.num_workers,
                             collate_fn=PaddingCollateFn(left_padding=False))

    return train_loader, eval_loader


def get_checkpoint_name(config):
    model_params = "_".join([f"{key}_{value}" for key, value in config.model.model_params.items()])
    return (
        f"{config.model.model_class}_" +
        f"{model_params}_" +
        f"bs_{config.dataloader.batch_size}_" +
        f"seed_{config.random_state}"
    ).translate(str.maketrans('', '', ".,/"))


def get_model_path(config, create_dir=True):
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 
        config.dataset.name, config.split_name)
    if create_dir and not os.path.exists(model_path):
        os.makedirs(model_path)
    return model_path


def create_model(config, item_count):
    if config.model.model_class == 'SASRec':
        model = SASRec(item_num=item_count, **config.model.model_params)
    # elif config.model.model_class == 'BERT4Rec':
    #     model = BERT4Rec(vocab_size=item_count,
    #                      bert_config=config.model.model_params)
    elif config.model.model_class == 'RNN':
        model = RNN(vocab_size=item_count,
                    rnn_config=config.model.model_params)
    else:
        raise ValueError(f"Unknown model_class: {config.model.model_class}. Supported: SASRec, RNN, BERT4Rec.")
    return model


def training(model, train_loader, eval_loader, config, task=None):
    model_path = get_model_path(config)
    checkpoint_name = get_checkpoint_name(config)

    seqrec_module = SeqRec(model, **config.seqrec_module)
    early_stopping = EarlyStopping(monitor="val_ndcg", mode="max",
                                   patience=config.patience, verbose=False)
    model_summary = ModelSummary(max_depth=4)
    checkpoint = ModelCheckpoint(
        dirpath=model_path, 
        filename='_' + checkpoint_name,
        save_top_k=1, monitor="val_ndcg",
        mode="max", save_weights_only=True)
    progress_bar = TQDMProgressBar(refresh_rate=20)
    device_stats = DeviceStatsMonitor()
    callbacks=[early_stopping, model_summary, checkpoint, progress_bar, device_stats]

    trainer = pl.Trainer(callbacks=callbacks, enable_checkpointing=True,
                         **config['trainer_params'])

    start_time = time.time()
   
    try:
        trainer.fit(model=seqrec_module,
                    train_dataloaders=train_loader,
                    val_dataloaders=eval_loader)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            if os.path.exists(checkpoint.best_model_path):
                os.remove(checkpoint.best_model_path)
                print(f"Removed checkpoint due to CUDA OOM error: {checkpoint.best_model_path}")
        raise
    finally:
        if not getattr(trainer, "interrupted", False):
            seqrec_module.load_state_dict(torch.load(checkpoint.best_model_path)['state_dict'])
            num_best_epochs = torch.load(checkpoint.best_model_path)['epoch']
            print(f"Model best epoch: {num_best_epochs}")
            if task is not None:
                task.get_logger().report_single_value('num_best_epochs', num_best_epochs)
            print(f"Loaded checkpoint from: {checkpoint.best_model_path}")
            os.rename(checkpoint.best_model_path, 
            os.path.join(model_path, checkpoint_name + "_" + str(num_best_epochs)+ "_epochs_" + str(time.time()) + ".ckpt")) 
        else:
            print("Detected interruption of training. Removed checkpoint.")
            os.remove(checkpoint.best_model_path)

    training_time = time.time() - start_time
    print('Training time', training_time)

    if task is not None:
        task.get_logger().report_single_value('training_time', training_time)

    return trainer, seqrec_module


def predict(trainer, seqrec_module, data, config, task=None, prefix='test'):
    start_time = time.time()
    if config.model.model_class in ['SASRec', 'RNN']:
        predict_dataset = CausalLMPredictionDataset(data, max_length=config.model.dataset_params.max_length)
    elif config.model.model_class == 'BERT4Rec':
        predict_dataset = MaskedLMPredictionDataset(data, max_length=config.model.dataset_params.max_length)
    else:
        raise ValueError(f"Model class {config.model.model_class} not supported. Supported: SASRec, RNN, BERT4Rec.")

    predict_loader = DataLoader(
        predict_dataset, shuffle=False,
        collate_fn=PaddingCollateFn(left_padding=False),
        batch_size=config.dataloader.test_batch_size,
        num_workers=config.dataloader.num_workers)

    seqrec_module.predict_top_k = max(config.evaluator.top_k)
    preds = trainer.predict(model=seqrec_module, dataloaders=predict_loader)

    recs = preds2recs(preds)
    print('recs shape', recs.shape)
    predict_time = time.time() - start_time
    print(f"{prefix} prediction time", predict_time)

    if task is not None:
        task.get_logger().report_single_value(f"{prefix}_predict_time", predict_time)
        if config[f"save_{prefix}_predictions"]:
            task.upload_artifact(f"{prefix}_pred.csv", recs)

    return recs


def evaluate(recs, test, train, task, config, prefix='test'):
    start_time = time.time()
    evaluator = Evaluator(metrics=list(config.evaluator.metrics),
                            top_k=list(config.evaluator.top_k),
                            polars_metrics=config.evaluator.polars_metrics)

    metrics = evaluator.compute_metrics(test, recs, train)
    metrics_dct = {prefix + '_' + key: value for key, value in metrics.items()}
    print('metrics\n', metrics_dct)

    if task:
        clearml_logger = task.get_logger()
        for key, value in metrics_dct.items():
            clearml_logger.report_single_value(key, value)
        metrics = pd.Series(metrics_dct).to_frame().reset_index()
        metrics.columns = ['metric_name', 'metric_value']

        clearml_logger.report_table(title=f'{prefix}_metrics', series='dataframe',
                                    table_plot=metrics)
        task.upload_artifact(f'{prefix}_metrics', metrics)
    
    eval_time = time.time() - start_time
    print(f"{prefix} evaluation time", eval_time)

    if task is not None:
        task.get_logger().report_single_value(f"{prefix}_eval_time", eval_time)
    
    return metrics_dct

if __name__ == "__main__":

    main()
