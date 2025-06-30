import os
from copy import deepcopy
from pathlib import Path
from datetime import datetime

import submitit

from common import get_default_args
from methods.early_exit import train as train_ee
from train import train
from utils import generate_run_name, submit_job
from visualize.cost_vs_plot import get_default_args as get_default_cost_plot_args
from visualize.cost_vs_plot import main as cost_vs_plot


# command:
# python -m scripts.sdn_example


def main():
    # ════════════════════════ submitit setup ════════════════════════ #

    job_name = "effbench"

    account = "plgeetraining-gpu-a100"
    account = None

    qos = "normal"

    partition = "plgrid-gpu-a100"
    # partition = 'rtx3080'
    # partition = 'batch'

    # timeout = 60 * 24 * 7
    timeout = 60 * 24 * 2
    # timeout = 10
    # timeout = 60

    gpus_per_task = 1
    cpus_per_gpu = 8
    # cpus_per_gpu = 4
    mem_per_gpu = "16G"
    # mem_per_gpu = "6G"

    executor = submitit.AutoExecutor(folder=os.environ["LOGS_DIR"])
    executor.update_parameters(
        stderr_to_stdout=True,
        timeout_min=timeout,
        slurm_job_name=job_name,
        slurm_account=account,
        slurm_qos=qos,
        slurm_partition=partition,
        slurm_ntasks_per_node=1,
        slurm_gpus_per_task=gpus_per_task,
        slurm_cpus_per_gpu=cpus_per_gpu,
        slurm_mem_per_gpu=mem_per_gpu,
    )

    # ════════════════════════ common experiment settings ════════════════════════ #

    common_args = get_default_args()

    exp_ids = [1,2,3]
    # exp_ids = [1,2,3,4,5]

    common_args.runs_dir = Path(os.environ["RUNS_DIR"])
    common_args.dataset = "cifar100"
    common_args.train_val_split_test_size = 0.1
    common_args.train_val_split_seed = 42
    # common_args.init_fun = 'effnets'
    common_args.eval_points = 50
    common_args.use_wandb = True
    common_args.mixup_alpha = None
    common_args.batch_size = 125
    # common_args.phase1 = 3000
    common_args.epochs = 3000  # == 0 indicates that phase 1 lasts 0 epochs

    common_args.loss_type = "ce"
    common_args.loss_args = {}
    common_args.loss_args.label_smoothing = 0.0

    common_args.optimizer_class = "sgd"
    common_args.optimizer_args = {}
    common_args.optimizer_args.lr = 8e-2
    common_args.optimizer_args.weight_decay = 0.0

    # common_args.scheduler_class = 'cosine'
    # common_args.scheduler_args = {'eta_min': 5e-3}
    common_args.scheduler_class = None
    common_args.scheduler_args = {}

    common_args.early_stopping = True
    common_args.early_stopping_patience = 50
    common_args.early_stopping_delta = 0.0
    common_args.early_stopping_metric_type = 'acc'
    
    common_args.is_rank_train = False
    common_args.is_rank_test = False
    common_args.rank_multiplier = 50
    common_args.is_mi_train = False
    common_args.is_mi_test = False
    common_args.mi_multiplier = 50
    common_args.MAX_REPR_SIZE = 2000

    # ════════════════════════ base model settings ════════════════════════ #

    base_model_args = deepcopy(common_args)

    base_model_args.model_class = "resnet34"
    base_model_args.model_args = {'standard_first_conv': False}
    # base_model_args.model_class = "resnet50"
    # base_model_args.model_args = {'standard_first_conv': False}
    base_model_args.model_args.num_classes = 100

    # ════════════════════════ SDN model settings ════════════════════════ #

    sdn_model_args = deepcopy(common_args)

    sdn_model_args.batch_size = 125
    sdn_model_args.phase2 = 3000  # == 0 indicates that phase 2 lasts 0 epochs / epoch_common
    sdn_model_args.phase3 = 0  # == 0 indicates that phase 3 lasts 0 epochs
    sdn_model_args.epochs = sdn_model_args.phase2 + sdn_model_args.phase3
    sdn_model_args.epochs_common = sdn_model_args.phase2
    sdn_model_args.with_backbone = True if sdn_model_args.phase2 != 0 else False
    sdn_model_args.with_phase3 = True if sdn_model_args.phase3 != 0 else False
    sdn_model_args.local_description = f'+-+'
    sdn_model_args.gather_stats = False

    sdn_model_args.model_class = "sdn"
    sdn_model_args.model_args = {}
    sdn_model_args.model_args.head_type = "pooled_head"
    sdn_model_args.model_args.place_at = [4, 6, 8, 10, 12, 14]
    sdn_model_args.aggregate_loss = 'arith_mean'

    sdn_model_args.optimizer_args = {}
    sdn_model_args.optimizer_args.lr = 8e-2
    # sdn_model_args.optimizer_args.momentum = 0.9
    # sdn_model_args.optimizer_args.nesterov = True
    sdn_model_args.optimizer_args.weight_decay = 0.0
    
    sdn_model_args.gradient_equilibrium = False

    # ════════════════════════ train base models ════════════════════════ #rwszy ro

    jobs = []
    base_jobs = {}
    exp_names = []
    display_names = []

    for exp_id in exp_ids:
        args = deepcopy(base_model_args)
        args.exp_id = exp_id

        job = submit_job(executor, train, args, num_gpus=gpus_per_task)
        jobs.append(job)
        base_jobs[exp_id] = job

    base_model_exp_name = generate_run_name(args)[0]
    exp_names.append(base_model_exp_name)
    display_names.append(f"ResNet34")

    # ════════════════════════ train SDN models ════════════════════════ #

    for exp_id in exp_ids:
        args = deepcopy(sdn_model_args)
        args.exp_id = exp_id
        args.base_on = base_model_exp_name  # path to the base model will be inferred from the experiment name

        # this makes the sdn job wait for the base model to finish
        dependency_str = f"afterany:{base_jobs[exp_id].job_id}"

        executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
        job = submit_job(executor, train_ee, args, num_gpus=gpus_per_task)
        jobs.append(job)

    sdn_model_exp_name = generate_run_name(args)[0]
    exp_names.append(sdn_model_exp_name)
    display_names.append(f"SDN")

    # ═════════════════════════════════════════════════════════ #

    print(f"Exp names: {exp_names}")
    print(f"Display names: {display_names}")
    print(f"SLURM JIDs: {[job.job_id for job in jobs]}")

    # ════════════════════════ plot cost vs acc ════════════════════════ #

    plot_args = get_default_cost_plot_args()
    out_dir_name = f"ee_{common_args.dataset}"
    output_dir = Path(os.environ["RESULTS_DIR"]) / out_dir_name / datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    plot_args.output_dir = output_dir
    plot_args.runs_dir = common_args.runs_dir
    plot_args.exp_names = exp_names
    plot_args.exp_ids = exp_ids  # TODO: problem z rysowaniem plotu
    plot_args.display_names = display_names
    plot_args.output_name = "cost_vs"
    plot_args.mode = "acc"
    plot_args.use_wandb = False
    dependency_str = f'afterany:{":".join(job.job_id for job in jobs)}'  # wait for all jobs to finish before plotting
    executor.update_parameters(stderr_to_stdout=True,
                               timeout_min=timeout,
                               slurm_job_name=job_name,
                               slurm_account=account,
                               slurm_qos=qos,
                               slurm_partition=partition,
                               slurm_ntasks_per_node=1,
                               slurm_gpus_per_task=1,
                               slurm_cpus_per_gpu=cpus_per_gpu,
                               slurm_mem_per_gpu=mem_per_gpu,
                               slurm_additional_parameters={"dependency": dependency_str})
    submit_job(executor, cost_vs_plot, plot_args)


if __name__ == "__main__":
    main()
