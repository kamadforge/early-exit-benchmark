import os
from pathlib import Path

from omegaconf import DictConfig
from submitit import AutoExecutor

from common import get_default_args
from scripts.regimes.configuration.executor import get_executor
from scripts.regimes.regimes_training import RegimeChoice, RegimeTrainer, train_regimes


class ViTTinyCifar100Trainer(RegimeTrainer):
    def get_executor(self, job_name: str) -> AutoExecutor:
        executor = get_executor(slurm_job_name=job_name, slurm_partition='rtx4090_batch',
                                slurm_qos='batch', timeout_min=60 * 24 * 1)
        return executor

    def get_common_args(self) -> DictConfig:
        common_args = get_default_args()

        common_args.runs_dir = Path(os.environ["RUNS_DIR"])
        common_args.dataset = "cifar100"
        common_args.dataset_args = {}
        common_args.dataset_args.variant = "trivial_augment"
        common_args.dataset_args.image_size = 32

        common_args.use_wandb = True
        common_args.batch_size = 256
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
        common_args.early_stopping_patience = 30
        common_args.early_stopping_delta = 0.0
        common_args.early_stopping_metric_type = 'acc'

        common_args.cutmix_alpha = 1.0
        common_args.mixup_alpha = 0.8
        common_args.mixup_smoothing = 0.1

        return common_args

    def setup_base_args(self, base_model_args: DictConfig):
        base_model_args.model_class = "vit"
        base_model_args.model_args = {}
        base_model_args.model_args.num_classes = 100
        base_model_args.model_args.input_channels = 3
        base_model_args.model_args.image_size = 32
        base_model_args.model_args.patch_size = 4
        base_model_args.model_args.hidden_dim = 256
        base_model_args.model_args.mlp_dim = 256
        base_model_args.model_args.num_layers = 7
        base_model_args.model_args.num_heads = 8
        base_model_args.model_args.dropout = 0.0
        base_model_args.model_args.attention_dropout = 0.0

    def setup_ee_args(self, sdn_model_args: DictConfig):
        sdn_model_args.model_args = {}
        sdn_model_args.model_args.head_type = "vit_standard_head"
        sdn_model_args.model_args.place_at = [0, 1, 2, 3, 4, 5, 6]

    def setup_gpf_args(self, gpf_model_args):
        gpf_model_args.model_args.head_dim = 512
        gpf_model_args.model_args.state_dropout = 0.0

    def name(self):
        return "vitcif"


def main():
    trainer = ViTTinyCifar100Trainer()
    lr = 5e-4
    exp_ids = [1, 2, 3]
    train_regimes(trainer, lr, exp_ids, max_epochs_amount_per_phase=1000, train_base=True,
                  train_sdn=True, train_gpf=False, train_pbee=True)

    # wait until base models are trained to avoid training them twice
    # chosen_regimes = RegimeChoice(disjoint=True, joint=True, mixed=True, alternating=False, branch_wise=False, late_first=False, separate=False)
    # train_regimes(trainer, lr, exp_ids, max_epochs_amount_per_phase=1000, train_base=True,
    #               train_sdn=False, train_gpf=True, train_pbee=False, chosen_regimes=chosen_regimes)


if __name__ == "__main__":
    main()
