import os

import submitit


def get_executor(
    slurm_job_name="effbench",
    stderr_to_stdout=True,
    timeout_min=60 * 24 * 2,
    slurm_account=None,
    slurm_qos="normal",
    slurm_partition="plgrid-gpu-a100",
    slurm_ntasks_per_node=1,
    slurm_gpus_per_task=1,
    slurm_cpus_per_gpu=16,
    slurm_mem_per_gpu="64G",
):
    executor = submitit.AutoExecutor(folder=os.environ["LOGS_DIR"])
    executor.update_parameters(
        stderr_to_stdout=stderr_to_stdout,
        timeout_min=timeout_min,
        slurm_job_name=slurm_job_name,
        slurm_account=slurm_account,
        slurm_qos=slurm_qos,
        slurm_partition=slurm_partition,
        slurm_ntasks_per_node=slurm_ntasks_per_node,
        slurm_gpus_per_task=slurm_gpus_per_task,
        slurm_cpus_per_gpu=slurm_cpus_per_gpu,
        slurm_mem_per_gpu=slurm_mem_per_gpu,
    )
    return executor
