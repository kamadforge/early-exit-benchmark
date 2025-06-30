import os
from pathlib import Path

from omegaconf import DictConfig
from submitit import AutoExecutor

from common import get_default_args
from scripts.regimes.configuration.executor import get_executor
from scripts.regimes.regimes_training import RegimeTrainer, train_regimes


class MSDNetCifar100Trainer(RegimeTrainer):

    def get_executor(self, job_name: str) -> AutoExecutor:
        executor = get_executor(slurm_job_name=job_name, slurm_partition='rtx4090_batch',
                                slurm_qos='batch', timeout_min=60 * 24 * 1)
        return executor

    def get_common_args(self) -> DictConfig:
        common_args = get_default_args()
        common_args.runs_dir = Path(os.environ['RUNS_DIR'])
        common_args.dataset = 'cifar100'
        common_args.dataset_args = {}
        common_args.train_val_split_test_size = 0.05
        common_args.train_val_split_seed = 42

        common_args.use_wandb = True

        common_args.batch_size = 512
        common_args.eval_points = 40

        common_args.loss_type = 'ce'
        common_args.loss_args = {}
        common_args.loss_args.label_smoothing = 0.0

        common_args.optimizer_class = 'adam'
        common_args.optimizer_args = {}
        common_args.optimizer_args.weight_decay = 0.0

        common_args.scheduler_class = 'cosine_warm_restarts'
        common_args.scheduler_args = {}
        common_args.scheduler_args.T_0 = 1500
        common_args.scheduler_args.eta_min = 1e-6

        common_args.early_stopping = True
        common_args.early_stopping_patience = 50
        common_args.early_stopping_delta = 0.0
        common_args.early_stopping_metric_type = 'acc'

        common_args.mixup_alpha = 0.8
        common_args.mixup_smoothing = 0.1
        common_args.cutmix_alpha = 1.0

        common_args.is_rank_train = False
        common_args.is_rank_test = False
        common_args.is_mi_train = False
        common_args.is_mi_test = False

        return common_args

    def setup_base_args(self, base_model_args: DictConfig):
        base_model_args.model_class = 'msdnet'
        base_model_args.model_args = {}
        base_model_args.model_args.num_classes = 200
        base_model_args.model_args.num_blocks = 7
        base_model_args.model_args.step = 2
        base_model_args.model_args.stepmode = 'even'
        base_model_args.model_args.num_channels = 16
        base_model_args.model_args.variant = 'cifar'
        base_model_args.model_args.growth_rate = 16
        base_model_args.model_args.gr_factors = [1, 2, 4]
        base_model_args.model_args.bn_factors = [1, 2, 4]
        base_model_args.model_args.base = 4
        base_model_args.model_args.prune = 'max'
        base_model_args.model_args.bottleneck = True
        base_model_args.model_args.reduction = 0.5

        base_model_args.init_fun = 'default'

    def setup_ee_args(self, sdn_model_args: DictConfig):
        sdn_model_args.gather_stats = False
        sdn_model_args.model_args = {}
        sdn_model_args.model_args.head_type = 'ranet_head'
        sdn_model_args.model_args.place_at = [i for i in range(0, 18)]
        sdn_model_args.aggregate_loss = 'arith_mean'
        sdn_model_args.gradient_equilibrium = False

    def name(self):
        return "msdcif"


def main():
    trainer = MSDNetCifar100Trainer()
    lr = 1e-3
    exp_ids = [1, 2, 3]
    train_regimes(trainer, lr, exp_ids, max_epochs_amount_per_phase=1000, train_base=True,
                  train_sdn=True, train_gpf=False, train_pbee=False)


if __name__ == "__main__":
    main()
