import os
from pathlib import Path

from omegaconf import DictConfig
from submitit import AutoExecutor

from common import get_default_args
from scripts.regimes.configuration.executor import get_executor
from scripts.regimes.nlp.config import get_bert_args, get_dataset_args
from scripts.regimes.regimes_training import RegimeTrainer, train_regimes


class BertNewsgropupsTrainer(RegimeTrainer):
    def get_executor(self, job_name: str) -> AutoExecutor:
        executor = get_executor(slurm_job_name=job_name, slurm_partition='rtx4090_batch',
                                slurm_qos='batch', timeout_min=60 * 24 * 1)
        return executor

    def get_common_args(self) -> DictConfig:
        common_args = get_default_args()
        common_args.runs_dir = Path(os.environ["RUNS_DIR"])

        common_args.dataset = "20newsgroups"
        common_args.dataset_args = get_dataset_args(common_args.dataset)
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
        common_args.early_stopping_patience = 3
        common_args.early_stopping_delta = 0.0
        common_args.early_stopping_metric_type = "acc"

        return common_args

    def setup_base_args(self, base_model_args: DictConfig):
        base_model_args.model_class = "bert_base"
        base_model_args.model_args = get_bert_args(base_model_args.dataset)
        base_model_args.model_args.hidden_dropout_prob = 0.0
        base_model_args.model_args.attention_probs_dropout_prob = 0.0
        base_model_args.batch_size = 32
        base_model_args.eval_points = 20

    def setup_ee_args(self, config: DictConfig):
        config.model_args = {}
        config.model_args.head_type = "vit_standard_head"
        config.model_args.place_at = [2 * i + 1 for i in range(11)]
        config.batch_size = 16
        config.eval_points = 20

    def setup_gpf_args(self, gpf_model_args):
        gpf_model_args.model_args.head_dim = 768
        gpf_model_args.model_args.state_dropout = 0.0

    def name(self):
        return f"brt_nws"


def main():
    trainer = BertNewsgropupsTrainer()
    lr = 5e-5
    exp_ids = [1, 2, 3]
    train_regimes(trainer, lr, exp_ids, max_epochs_amount_per_phase=1000, train_base=True,
                  train_sdn=True, train_gpf=False, train_pbee=False)


if __name__ == "__main__":
    main()
