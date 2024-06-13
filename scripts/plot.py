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


def main():
    # ════════════════════════ submitit setup ════════════════════════ #

    job_name = "effbench"

    account = "plgeetraining-gpu-a100"
    qos = "normal"
    partition = "plgrid-gpu-a100"
    timeout = 60 // 2

    gpus_per_task = 1
    cpus_per_gpu = 4
    mem_per_gpu = "64G"

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

    exp_ids = [1, 2, 3]

    common_args.runs_dir = Path(os.environ["RUNS_DIR"])
    common_args.dataset = "cifar100"
    

    # ════════════════════════ plot cost vs acc ════════════════════════ #
    
    exp_names = ['cifar100_tv_vit_b_16_adapt_head_A7F47VTU',  'cifar100_sdn_OUKWIUPB', 'cifar100_sdn_4DDIILT6', 'cifar100_sdn_OH7IR7OZ']
    display_names = ['ViT-B-IN1k pretrained', '-+-', '++-', '+-+']

    plot_args = get_default_cost_plot_args()
    out_dir_name = f"ee_{common_args.dataset}"
    output_dir = Path(os.environ["RESULTS_DIR"]) / out_dir_name / "vit-b-pretrained"
    plot_args.output_dir = output_dir
    plot_args.runs_dir = common_args.runs_dir
    plot_args.exp_names = exp_names
    plot_args.exp_ids = exp_ids
    plot_args.display_names = display_names
    plot_args.output_name = "cost_vs"
    plot_args.mode = "acc"
    plot_args.use_wandb = False
    
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
                               slurm_additional_parameters={})
    #submit_job(executor, cost_vs_plot, plot_args)
    cost_vs_plot(plot_args)


if __name__ == "__main__":
    main()