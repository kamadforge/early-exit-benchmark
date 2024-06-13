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


# command:
# python -m scripts.sdn_example


def main():
    # ════════════════════════ submitit setup ════════════════════════ #

    job_name = "effbench"

    account = "plgccbench-gpu-a100"
    # account = None

    qos = "normal"

    partition = "plgrid-gpu-a100"
    # partition = 'dgxmatinf,rtx3080'
    # partition = 'batch'

    # timeout = 60 * 24 * 7
    timeout = 60 * 24 // 2

    gpus_per_task = 1
    cpus_per_gpu = 16
    mem_per_gpu = "64G"

    # executor = submitit.AutoExecutor(folder=os.environ["LOGS_DIR"])
    # executor.update_parameters(
    #     stderr_to_stdout=True,
    #     timeout_min=timeout,
    #     slurm_job_name=job_name,
    #     slurm_account=account,
    #     slurm_qos=qos,
    #     slurm_partition=partition,
    #     slurm_ntasks_per_node=1,
    #     slurm_gpus_per_task=gpus_per_task,
    #     slurm_cpus_per_gpu=cpus_per_gpu,
    #     slurm_mem_per_gpu=mem_per_gpu,
    # )

    # ════════════════════════ common experiment settings ════════════════════════ #

    common_args = get_default_args()

    # exp_ids = [1, 2, 3]
    exp_ids = [1]

    common_args.runs_dir = Path(os.environ["RUNS_DIR"])
    common_args.dataset = "tinyimagenet"
    

    # ════════════════════════ plot cost vs acc ════════════════════════ #
    
    exp_names = ['cifar10_sdn_5MGCTDFH', 'cifar10_sdn_2R3BNYXM', 'cifar10_sdn_JEVO7YRR', 'cifar10_sdn_56H6WYXL']
    display_names = ['0:200', '50:150', '100:100', '150:50']
    # display_names = ['ResNet-50', '200:0', '150:50', '100:100', '50:150', '0:200', '150:50 (frozen backbone)']

    exp_names = ['cifar10_vit_SSRSOGDV', 'cifar10_sdn_FTVJ7MBU', 'cifar10_sdn_KHZEQHHY', 'cifar10_sdn_S3X3V6IO', 'cifar10_sdn_HWTMXSNI']
    display_names = ['ViT-120', '75:25', '25:75', '0:100', '50:50']


    plot_args = get_default_cost_plot_args()
    out_dir_name = f"ee_{common_args.dataset}"
    output_dir = Path(os.environ["RESULTS_DIR"]) / out_dir_name / "together"
    plot_args.output_dir = output_dir
    plot_args.runs_dir = common_args.runs_dir
    plot_args.exp_names = exp_names
    plot_args.exp_ids = exp_ids
    plot_args.display_names = display_names
    plot_args.output_name = "cost_vs"
    plot_args.mode = "acc"
    plot_args.use_wandb = False
    # dependency_str = f'afterany:{":".join(job.job_id for job in jobs)}'  # wait for all jobs to finish before plotting
    # executor.update_parameters(stderr_to_stdout=True,
    #                            timeout_min=timeout,
    #                            slurm_job_name=job_name,
    #                            slurm_account=account,
    #                            slurm_qos=qos,
    #                            slurm_partition=partition,
    #                            slurm_ntasks_per_node=1,
    #                            slurm_gpus_per_task=1,
    #                            slurm_cpus_per_gpu=cpus_per_gpu,
    #                            slurm_mem_per_gpu=mem_per_gpu,
    #                            slurm_additional_parameters={"dependency": dependency_str})
    # submit_job(executor, cost_vs_plot, plot_args)
    cost_vs_plot(plot_args)


if __name__ == "__main__":
    main()
