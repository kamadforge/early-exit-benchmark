import os
from pathlib import Path

from omegaconf import DictConfig
from submitit import AutoExecutor

from common import get_default_args
from scripts.regimes.configuration.executor import get_executor
from scripts.regimes.regimes_training import RegimeTrainer, train_regimes


class Resnet34Cifar100Trainer(RegimeTrainer):
    def get_executor(self, job_name: str) -> AutoExecutor:
        executor = get_executor(slurm_job_name=job_name, slurm_partition='rtx4090_batch',
                                slurm_qos='batch', timeout_min=60 * 24 * 1)
        return executor

    def get_common_args(self) -> DictConfig:
        common_args = get_default_args()

        common_args.runs_dir = Path(os.environ["RUNS_DIR"])
        common_args.dataset = "cifar100"

        common_args.train_val_split_test_size = 0.1
        common_args.train_val_split_seed = 42

        common_args.use_wandb = True
        common_args.batch_size = 128
        common_args.eval_points = 50

        common_args.loss_type = "ce"
        common_args.loss_args = {}
        common_args.loss_args.label_smoothing = 0.0

        common_args.optimizer_class = "adam"
        common_args.optimizer_args = {}
        common_args.optimizer_args.weight_decay = 0.0

        common_args.scheduler_class = "cosine_warm_restarts"
        common_args.scheduler_args = {}
        common_args.scheduler_args.T_0 = 1000
        common_args.scheduler_args.eta_min = 0.0

        common_args.early_stopping = True
        common_args.early_stopping_patience = 50
        common_args.early_stopping_delta = 0.0
        common_args.early_stopping_metric_type = 'acc'

        common_args.cutmix_alpha = 1.0
        common_args.mixup_alpha = 0.8
        common_args.mixup_smoothing = 0.1

        return common_args

    def setup_base_args(self, base_model_args: DictConfig):
        base_model_args.model_class = "resnet34"
        base_model_args.model_args = {'standard_first_conv': False}
        base_model_args.model_args.num_classes = 100

    def setup_ee_args(self, sdn_model_args: DictConfig):
        sdn_model_args.model_args = {}
        sdn_model_args.model_args.head_type = "pooled_head"
        sdn_model_args.model_args.place_at = [2, 4, 6, 8, 10, 12, 14]
        sdn_model_args.aggregate_loss = 'arith_mean'
        sdn_model_args.is_gradient_equalibrum = False

    def setup_gpf_args(self, gpf_model_args):
        gpf_model_args.model_args.head_dim = 256
        gpf_model_args.model_args.state_dropout = 0.0

    def name(self):
        return "rescif"


def main():
    trainer = Resnet34Cifar100Trainer()
    lr = 5e-4
    exp_ids = [1, 2, 3]
    train_regimes(trainer, lr, exp_ids, max_epochs_amount_per_phase=1000, train_base=True,
                  train_sdn=True, train_gpf=False, train_pbee=False)


if __name__ == "__main__":
    main()
