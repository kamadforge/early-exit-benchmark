import os
from pathlib import Path

from omegaconf import DictConfig
from submitit import AutoExecutor

from common import get_default_args
from scripts.regimes.configuration.executor import get_executor
from scripts.regimes.regimes_training import (RegimeChoice, RegimeTrainer,
                                              train_regimes)


class ViTTinyImagenetTrainer(RegimeTrainer):
    def __init__(self, n_gpus: int, use_memfs: bool):
        super().__init__()
        self.n_gpus = n_gpus
        self.use_memfs = use_memfs

    def get_executor(self, job_name: str) -> AutoExecutor:
        executor = get_executor(slurm_job_name=job_name, timeout_min=60 *
                                24 * 2, slurm_gpus_per_task=self.n_gpus, slurm_mem_per_gpu="125G")
        if self.use_memfs:
            executor.update_parameters(
                slurm_additional_parameters={"constraint": "memfs"},
            )
        return executor

    def get_common_args(self) -> DictConfig:
        common_args = get_default_args()

        common_args.runs_dir = Path(os.environ["RUNS_DIR"])
        common_args.dataset = "imagenet"
        common_args.dataset_args = {}
        common_args.dataset_args.variant = "trivial_augment"
        common_args.train_val_split_test_size = 0.1
        common_args.train_val_split_seed = 42
        common_args.eval_points = 100
        common_args.use_wandb = True

        common_args.mixup_alpha = 0.8
        common_args.cutmix_alpha = 1.0
        common_args.mixup_smoothing = 0.1

        common_args.batch_size = 512 * self.n_gpus
        common_args.loss_type = "ce"
        common_args.loss_args = {}
        common_args.loss_args.label_smoothing = 0.0

        common_args.optimizer_class = "adam"
        common_args.optimizer_args = {}
        common_args.optimizer_args.weight_decay = 0.0

        common_args.scheduler_class = "cosine_warm_restarts"
        common_args.scheduler_args = {}
        common_args.scheduler_args.T_0 = 10000
        common_args.scheduler_args.eta_min = 0.0

        common_args.early_stopping = True
        common_args.early_stopping_patience = 25
        common_args.early_stopping_delta = 0.0
        common_args.early_stopping_metric_type = "acc"

        return common_args

    def setup_base_args(self, base_model_args: DictConfig):
        base_model_args.model_class = "vit"
        base_model_args.model_args = {}
        base_model_args.model_args.num_classes = 1000

        base_model_args.model_args.image_size = 224
        base_model_args.model_args.patch_size = 16
        base_model_args.model_args.hidden_dim = 192
        base_model_args.model_args.mlp_dim = 768
        base_model_args.model_args.num_layers = 12
        base_model_args.model_args.num_heads = 3

    def setup_ee_args(self, sdn_model_args):
        sdn_model_args.model_args = {}
        sdn_model_args.model_args.head_type = "vit_standard_head"
        sdn_model_args.model_args.place_at = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def name(self):
        return "vittIMK"

def main():
    trainer = ViTTinyImagenetTrainer(n_gpus=4, use_memfs=True)
    lr = 5e-4
    exp_ids = [1, 2, 3]
    chosen_regimes = RegimeChoice(disjoint=True, joint=True, mixed=True,
                                  alternating=False, branch_wise=False, late_first=False, separate=False)
    train_regimes(trainer, lr, exp_ids, max_epochs_amount_per_phase=3000, train_base=True,
                  train_sdn=True, train_gpf=False, train_pbee=False, chosen_regimes=chosen_regimes)

if __name__ == "__main__":
    main()
