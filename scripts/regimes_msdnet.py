import os
from copy import deepcopy
from pathlib import Path

import submitit

from common import get_default_args
from methods.early_exit import train as train_ee
from train import train
from utils import generate_run_name, submit_job
from visualize.cost_vs_plot import get_default_args as get_default_cost_plot_args
from visualize.cost_vs_plot import main as cost_vs_plot
from visualize.cost_vs_table import main as cost_vs_table


def main():
    # ════════════════════════ submitit setup ════════════════════════ #

    job_name = 'effbench'

    # account = 'plgccbench-gpu-a100'
    account = None
    # qos = 'big'
    qos = 'normal'
    # qos = 'batch'
    # qos = 'preemptible'
    # partition = 'plgrid-gpu-a100'
    partition = 'dgxa100'
    # partition = 'rtx4090'
    # partition = 'rtx4090_batch'
    # partition = 'batch'

    # timeout = 60 * 24 * 7
    timeout = 60 * 24 * 2
    # timeout = 60 * 24 * 1

    gpus_per_task = 1
    # gpus_per_task = 2
    # cpus_per_gpu = 5
    cpus_per_gpu = 10
    # cpus_per_gpu = 20
    # cpus_per_gpu = 16
    # mem_per_gpu = '64G'

    gpu_type = ''
    # gpu_type = 'ampere:'

    executor = submitit.AutoExecutor(folder=os.environ['LOGS_DIR'])
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
        # slurm_mem_per_gpu=mem_per_gpu,
    )

    # ════════════════════════ common experiment settings ════════════════════════ #

    common_args = get_default_args()
    # exp_ids = [2, 3, 4]
    # exp_ids = [1, 2, 3]
    exp_ids = [1]
    common_args.runs_dir = Path(os.environ['RUNS_DIR'])
    # common_args.dataset = 'tinyimagenet'
    common_args.dataset = 'cifar100'
    common_args.dataset_args = {}
    common_args.train_val_split_test_size = 0.05
    common_args.train_val_split_seed = 42

    common_args.use_wandb = True

    # common_args.batch_size = 256
    common_args.batch_size = 512
    common_args.epochs = 0
    common_args.eval_points = 40

    common_args.loss_type = 'ce'
    common_args.loss_args = {}
    common_args.loss_args.label_smoothing = 0.0

    # common_args.optimizer_class = "sgd"
    # common_args.optimizer_args = {}
    # common_args.optimizer_args.lr = 8e-2
    # common_args.optimizer_args.weight_decay = 0.0
    common_args.optimizer_class = 'adam'
    common_args.optimizer_args = {}
    common_args.optimizer_args.lr = 1e-3
    common_args.optimizer_args.weight_decay = 0.0
    # not tuned, but 1e-4 was used in the original MSDNet paper for cifar100
    # common_args.optimizer_args.weight_decay = 1e-4

    common_args.scheduler_class = 'cosine_warm_restarts'
    common_args.scheduler_args = {}
    # common_args.scheduler_args.T_0 = 1000
    common_args.scheduler_args.T_0 = 1500
    common_args.scheduler_args.eta_min = 1e-6

    # common_args.scheduler_class = None
    # common_args.scheduler_args = {}

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

    # ═══════════════════════════════════════════════════════════════════ #

    place_at = [i for i in range(0, 18)]

    # ═══════════════════════════════════════════════════════════════════ #

    jobs = []
    run_to_job_map = {}
    exp_names = []
    display_names = []

    # for regime in ['disjoint', 'joint', 'mixed']:
    # for regime in ['disjoint']:
    for regime in ['joint']:
        if regime == 'disjoint':
            epochs_backbone, epochs_heads_only, epochs_everything = 1000, 1000, 0
        elif regime == 'joint':
            # epochs_backbone, epochs_heads_only, epochs_everything = 0, 0, 2000
            epochs_backbone, epochs_heads_only, epochs_everything = 0, 0, 1
        elif regime == 'mixed':
            epochs_backbone, epochs_heads_only, epochs_everything = 1000.0, 0, 1000

        # ════════════════════════ base model settings ════════════════════════ #

        base_model_args = deepcopy(common_args)
        # pretrained but reset weights
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

        base_model_args.epochs = epochs_backbone
        base_model_args.init_fun = 'default'

        # ════════════════════════ train base models ════════════════════════ #

        for exp_id in exp_ids:
            args = deepcopy(base_model_args)
            args.exp_id = exp_id
            executor.update_parameters(slurm_additional_parameters={})
            exp_name, run_name = generate_run_name(args)
            if run_name not in run_to_job_map:
                job = submit_job(executor, train, args, num_gpus=gpus_per_task, gpu_type=gpu_type)
                jobs.append(job)
                run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        display_names.append(f'MSDNet')
        base_exp_name = exp_name

        # ════════════════════════ SDN model settings ════════════════════════ #

        sdn_model_args = deepcopy(common_args)
        phase2 = epochs_everything
        phase3 = epochs_heads_only
        sdn_model_args.epochs = phase2 + phase3
        sdn_model_args.epochs_common = phase2
        sdn_model_args.with_backbone = True if phase2 != 0 else False
        sdn_model_args.with_phase3 = True if phase3 != 0 else False
        sdn_model_args.local_description = f'something_something'
        sdn_model_args.gather_stats = False

        sdn_model_args.model_class = 'sdn'
        sdn_model_args.model_args = {}
        sdn_model_args.model_args.head_type = 'ranet_head'
        sdn_model_args.model_args.place_at = place_at
        sdn_model_args.aggregate_loss = 'arith_mean'

        sdn_model_args.optimizer_class = 'adam'
        # sdn_model_args.optimizer_args = {}
        # sdn_model_args.optimizer_args.lr = 0.001
        # sdn_model_args.optimizer_args.weight_decay = 0.0

        sdn_model_args.gradient_equilibrium = False

        # ════════════════════════ train SDN models ════════════════════════ #

        for exp_id in exp_ids:
            args = deepcopy(sdn_model_args)
            args.exp_id = exp_id
            args.base_on = base_exp_name
            base_run_name = f'{base_exp_name}_{exp_id}'
            dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
            executor.update_parameters(slurm_additional_parameters={'dependency': dependency_str})
            job = submit_job(executor, train_ee, args, num_gpus=gpus_per_task, gpu_type=gpu_type)
            jobs.append(job)
            exp_name, run_name = generate_run_name(args)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        display_names.append(f'{regime}')

    # ═════════════════════════════════════════════════════════ #

    print(f'Exp names: {exp_names}')
    print(f'Display names: {display_names}')
    print(f'SLURM JIDs: {[job.job_id for job in jobs]}')

    # ════════════════════════ plot cost vs acc ════════════════════════ #

    plot_args = get_default_cost_plot_args()
    out_dir_name = f'regimes_{base_model_args.model_class}_t_{common_args.dataset}'
    output_dir = Path(os.environ['RESULTS_DIR']) / out_dir_name
    plot_args.output_dir = output_dir
    plot_args.runs_dir = common_args.runs_dir
    plot_args.exp_names = exp_names
    plot_args.exp_ids = exp_ids
    plot_args.display_names = display_names
    plot_args.output_name = 'cost_vs'
    plot_args.mode = 'acc'
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
                               # slurm_mem_per_gpu=mem_per_gpu,
                               slurm_additional_parameters={'dependency': dependency_str})
    submit_job(executor, cost_vs_plot, plot_args, gpu_type=gpu_type)

    # ════════════════════════ generate a table ════════════════════════ #

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
                               # slurm_mem_per_gpu=mem_per_gpu,
                               slurm_additional_parameters={'dependency': dependency_str})
    submit_job(executor, cost_vs_table, plot_args, gpu_type=gpu_type)


if __name__ == '__main__':
    main()
