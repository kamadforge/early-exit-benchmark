import os
from pathlib import Path

from common import get_default_args
from datasets_config import (DATASET_TO_NUM_CLASSES,
                             DATASET_TO_SEQUENCE_LENGTH,
                             MODEL_TO_TOKENIZER_NAME)

DISPLAY_BASE = "BERT-base"

DS_TO_BASE_ARGS = {
    "rte": {"lr": 2e-5, "batch_size": 16, "epochs": 5},
    "mrpc": {"lr": 2e-5, "batch_size": 16, "epochs": 5},
    "sst2": {"lr": 1e-5, "batch_size": 16, "epochs": 3},
    "qqp": {"lr": 5e-5, "batch_size": 32, "epochs": 3},
    "qnli": {"lr": 1e-5, "batch_size": 16, "epochs": 3},
    "20newsgroups": {"lr": 5e-5, "batch_size": 32, "epochs": 5, "ee_batch_size": 64},
}


def get_bert_args(dataset):
    SEQ_LEN = DATASET_TO_SEQUENCE_LENGTH[dataset]
    NUM_CLASSES = DATASET_TO_NUM_CLASSES[dataset]
    return {"num_classes": NUM_CLASSES, "max_seq_length": SEQ_LEN}


def get_dataset_args(dataset):
    return {
        "tokenizer_name": MODEL_TO_TOKENIZER_NAME["bert_base"],
        "max_seq_length": DATASET_TO_SEQUENCE_LENGTH[dataset],
    }


def get_bert_newsgroups_args():
    common_args = get_default_args()
    common_args.runs_dir = Path(os.environ["RUNS_DIR"])

    dataset = "20newsgroups"
    common_args.dataset = dataset
    common_args.dataset_args = get_dataset_args(dataset=dataset)
    common_args.use_wandb = True
    common_args.mixup_alpha = None
    common_args.cutmix_alpha = None
    common_args.loss_type = "ce"
    common_args.loss_args = {}
    common_args.loss_args.label_smoothing = 0.0
    common_args.optimizer_class = "adam"
    common_args.optimizer_args = {}
    common_args.optimizer_args.weight_decay = 0.01

    common_args.scheduler_class = "cosine_warm_restarts"
    common_args.scheduler_args = {}
    common_args.scheduler_args.T_0 = 100
    common_args.scheduler_args.eta_min = 0

    common_args.early_stopping = True
    common_args.early_stopping_patience = 10
    common_args.early_stopping_delta = 0.0
    common_args.early_stopping_metric_type = "acc"

    return common_args
