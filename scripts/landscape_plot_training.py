import os
from pathlib import Path

import submitit

from common import get_default_args
from utils import submit_job
from visualize.cost_vs_plot import get_default_args as get_default_cost_plot_args

from visualize.ee_landscape import visualize_ee_training_loss_landscape

# command:
# python -m scripts.landscape_plot_training


def main():
    # ════════════════════════ submitit setup ════════════════════════ #

    job_name = "effbench"

    account = None
    qos = "normal"
    partition = "plgrid-gpu-a100"
    timeout = 60 * 24 * 2

    gpus_per_task = 1
    cpus_per_gpu = 16
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

    exp_ids = [1]
    # exp_ids = [1,2,3]

    common_args.runs_dir = Path(os.environ["RUNS_DIR"])

    plot_args = get_default_cost_plot_args()
    out_dir_name = "ee_landscapes"
    output_dir = Path(os.environ["RESULTS_DIR"]) / out_dir_name
    plot_args.output_dir = output_dir
    plot_args.runs_dir = common_args.runs_dir

    plot_args.exp_names = [
        "cifar100_sdn_JRUXF5H6",
        "cifar100_sdn_HELG6SMN",
        "cifar100_sdn_OO3DMALB",
    ]
    plot_args.exp_ids = exp_ids
    plot_args.use_wandb = False
    plot_args.how_to_plot = "3d"
    plot_args.steps = 100
    plot_args.landscape_transform = None
    plot_args.use_full_loader, plot_args.batch_size = False, None
    plot_args.clips = [1.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0]
    plot_args.zoom = 1.0
    plot_args.only_heads = False
    plot_args.only_backbone = False
    submit_job(executor, visualize_ee_training_loss_landscape, plot_args)


if __name__ == "__main__":
    main()
