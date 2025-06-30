import os
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Iterable, Optional

from omegaconf import DictConfig
from submitit import AutoExecutor

from methods.early_exit import train as train_ee
from methods.early_exit_alternating import train as train_ee_alternating
from methods.early_exit_branch_wise import train as train_ee_branch_wise
from methods.early_exit_late_first import train as train_ee_late_first
from methods.early_exit_separate import train as train_ee_separate
from scripts.regimes.configuration.dependency import Dependency
from scripts.regimes.configuration.regimes_configuration import (
    AlternatingConfiguration, BranchWiseConfiguration, DisjointConfiguration,
    JointConfiguration, LateFirstConfiguration, MixedConfiguration,
    RegimeConfiguration, SeparateConfiguration)
from train import train
from utils import generate_run_name, submit_job
from visualize.cost_vs_plot import \
    get_default_args as get_default_cost_plot_args
from visualize.cost_vs_plot import main as cost_vs_plot
from visualize.cost_vs_table import main as cost_vs_table


@dataclass
class RegimeChoice:
    disjoint: bool = True
    joint: bool = True
    mixed: bool = True
    alternating: bool = True
    branch_wise: bool = True
    late_first: bool = True
    separate: bool = True


def update_additional_parameters(executor: AutoExecutor, parameters_update: dict):
    updated_parameters = executor.parameters.get("additional_parameters", {})
    updated_parameters.update(**parameters_update)
    executor.update_parameters(slurm_additional_parameters=updated_parameters)

class RegimeTrainer(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def get_common_args(self) -> DictConfig:
        pass

    @abstractmethod
    def get_executor(self, job_name: str) -> AutoExecutor:
        pass

    @abstractmethod
    def setup_base_args(self, config: DictConfig):
        pass

    @abstractmethod
    def setup_ee_args(self, config: DictConfig):
        pass

    def setup_gpf_args(self, gpf_model_args: DictConfig):
        raise NotImplementedError

    def get_base_args(self, regime_config: RegimeConfiguration, max_epochs_amount_per_phase: int, lr: float) -> DictConfig:
        args = self.get_common_args()
        args.optimizer_args.lr = lr

        self.setup_base_args(args)
        regime_config.update_base_config(
            args, max_epochs_amount_per_phase)
        if args.epochs == 0:
            args.scheduler_class = None
            args.scheduler_args = None
        return args

    def get_gpf_args(self, regime_config: RegimeConfiguration, max_epochs_amount_per_phase: int, lr: float) -> DictConfig:
        args = self.get_common_args()
        args.optimizer_args.lr = lr
        args.model_class = 'gpf'
        self.setup_ee_args(args)
        self.setup_gpf_args(args)
        regime_config.update_ee_config(
            args, max_epochs_amount_per_phase)
        args.training_regime = regime_config.regime_name()
        return args

    def get_sdn_args(self, regime_config: RegimeConfiguration, max_epochs_amount_per_phase: int, lr: float) -> DictConfig:
        args = self.get_common_args()
        args.optimizer_args.lr = lr
        args.model_class = 'sdn'
        self.setup_ee_args(args)
        regime_config.update_ee_config(
            args, max_epochs_amount_per_phase)
        args.training_regime = regime_config.regime_name()
        return args

    def get_pbee_args(self, regime_config: RegimeConfiguration, max_epochs_amount_per_phase: int, lr: float) -> DictConfig:
        args = self.get_common_args()
        args.model_class = 'pbee'

        args.with_backbone = False
        args.model_args = {}
        args.epochs = 0
        args.epochs_common = 0
        args.scheduler_class = None
        args.scheduler_args = None

        args.training_regime = regime_config.regime_name()
        return args

    def train_base(
        self,
        job_name: str,
        regime_config: RegimeConfiguration,
        exp_ids: list[int],
        max_epochs_amount_per_phase: int,
        lr: float,
        do_trainining: bool = True,
    ) -> dict[int, Dependency]:
        base_model_args = self.get_base_args(
            regime_config=regime_config,
            max_epochs_amount_per_phase=max_epochs_amount_per_phase,
            lr=lr
        )
        base_dependency_mapping = {}
        for exp_id in exp_ids:
            args = deepcopy(base_model_args)
            args.exp_id = exp_id

            full_job_name = f"{job_name}_{exp_id}"
            executor = self.get_executor(job_name=full_job_name)

            job_id = None
            if do_trainining:
                job = submit_job(
                    executor,
                    train,
                    args,
                    num_gpus=executor.parameters['gpus_per_task'] *
                    executor.parameters['ntasks_per_node'],
                )
                print(
                    f'submited SLURM JobName: {full_job_name}, JID: {job.job_id}')
                job_id = job.job_id

            exp_name, run_name = generate_run_name(args)
            base_dependency_mapping[exp_id] = Dependency(
                exp_name=exp_name, run_name=run_name, job_id=job_id
            )

        return base_dependency_mapping

    def train_ee(
        self,
        job_name: str,
        dependency_mapping: dict[int, Dependency],
        exp_ids: list[int],
        ee_model_args: DictConfig,
        do_trainining: bool = True,
    ):
        training_method_map = {
            "Disjoint": train_ee,
            "Joint": train_ee,
            "Mixed": train_ee,
            "Alternating": train_ee_alternating,
            "BranchWise": train_ee_branch_wise,
            "LateFirst": train_ee_late_first,
            "Separate": train_ee_separate,
        }

        sdn_dependency_mapping = {}
        for exp_id in exp_ids:
            args = deepcopy(ee_model_args)
            args.exp_id = exp_id
            args.base_on = dependency_mapping[exp_id].exp_name

            full_job_name = f"{job_name}_{exp_id}"
            executor = self.get_executor(job_name=full_job_name)
            if dependency_mapping[exp_id].job_id is not None:
                dependency_str = f"afterany:{dependency_mapping[exp_id].job_id}"
                update_additional_parameters(
                    executor, {"dependency": dependency_str})

            job_id = None
            if do_trainining:
                job = submit_job(
                    executor,
                    training_method_map[args.training_regime],
                    args,
                    num_gpus=executor.parameters['gpus_per_task'] *
                    executor.parameters['ntasks_per_node'],
                )
                print(
                    f'submited SLURM JobName: {full_job_name}, JID: {job.job_id}')
                job_id = job.job_id

            exp_name, run_name = generate_run_name(args)
            sdn_dependency_mapping[exp_id] = Dependency(
                job_id=job_id, exp_name=exp_name, run_name=run_name
            )

        return sdn_dependency_mapping

    def train_sdn(
        self,
        job_name: str,
        regime_config: RegimeConfiguration,
        exp_ids: list[int],
        max_epochs_amount_per_phase: int,
        lr: float,
        dependency_mapping: dict[int, Dependency],
        do_trainining: bool = True,
    ) -> dict[int, Dependency]:
        ee_model_args = self.get_sdn_args(
            regime_config=regime_config, max_epochs_amount_per_phase=max_epochs_amount_per_phase, lr=lr)
        return self.train_ee(job_name, dependency_mapping, exp_ids, ee_model_args, do_trainining)

    def train_gpf(
        self,
        job_name: str,
        regime_config: RegimeConfiguration,
        exp_ids: list[int],
        max_epochs_amount_per_phase: int,
        lr: float,
        dependency_mapping: dict[int, Dependency],
        do_trainining: bool = True,
    ) -> dict[int, Dependency]:
        ee_model_args = self.get_gpf_args(
            regime_config=regime_config, max_epochs_amount_per_phase=max_epochs_amount_per_phase, lr=lr)
        return self.train_ee(job_name, dependency_mapping, exp_ids, ee_model_args, do_trainining)

    def train_pbee(
        self,
        job_name: str,
        regime_config: RegimeConfiguration,
        exp_ids: list[int],
        max_epochs_amount_per_phase: int,
        lr: float,
        dependency_mapping: dict[int, Dependency],
        do_trainining: bool = True,
    ) -> dict[int, Dependency]:
        ee_model_args = self.get_pbee_args(
            regime_config=regime_config, max_epochs_amount_per_phase=max_epochs_amount_per_phase, lr=lr)
        return self.train_ee(job_name, dependency_mapping, exp_ids, ee_model_args, do_trainining)


def visualize_ee(trainer: RegimeTrainer, out_dir_name: str, plot_dependencies: list[Dependency], ee_configs: Iterable[RegimeConfiguration], exp_ids: list[int], runs_dir: Path, mode: str, confidence_type='max_prob'):
    plot_args = get_default_cost_plot_args()
    output_dir = Path(os.environ['RESULTS_DIR']) / out_dir_name
    plot_args.output_dir = output_dir
    plot_args.runs_dir = runs_dir
    plot_args.exp_names = list(dict.fromkeys(
        [d.exp_name for d in plot_dependencies]))
    print("Exp names", plot_args.exp_names)
    plot_args.exp_ids = exp_ids
    plot_args.display_names = ["Base"] + \
        [r.regime_name()
         for r in ee_configs]
    plot_args.mode = mode
    plot_args.confidence_type = confidence_type
    plot_args.output_name = 'cost_vs'
    if confidence_type is not None:
        plot_args.output_name = f"{confidence_type}_{plot_args.output_name}"

    # ════════════════════════ plot cost vs acc ════════════════════════ #

    executor = trainer.get_executor(f"plot-{trainer.name()}")
    blocking_jobs_list = [
        d.job_id for d in plot_dependencies if d.job_id is not None]
    if blocking_jobs_list:
        dependency_str = f'afterany:{":".join(blocking_jobs_list)}'
        update_additional_parameters(executor, {"dependency": dependency_str})

    job = submit_job(executor, cost_vs_plot, plot_args)
    print(f'submited SLURM JobName: plot, JID: {job.job_id}')
    # ════════════════════════ generate a table ════════════════════════ #

    executor = trainer.get_executor(f"table-{trainer.name()}")
    blocking_jobs_list = [
        d.job_id for d in plot_dependencies if d.job_id is not None]
    if blocking_jobs_list:
        dependency_str = f'afterany:{":".join(blocking_jobs_list)}'
        update_additional_parameters(executor, {"dependency": dependency_str})

    job = submit_job(executor, cost_vs_table, plot_args)
    print(f'submited SLURM JobName: table, JID: {job.job_id}')


def train_regimes(
    trainer: RegimeTrainer,
    lr: float, exp_ids: list[int],
    max_epochs_amount_per_phase: int,
    train_base: bool,
    train_sdn: bool,
    train_gpf: bool,
    train_pbee: bool,
    visualize_sdn: Optional[bool] = None,
    visualize_gpf: Optional[bool] = None,
    visualize_pbee: Optional[bool] = None,
    mode: str = 'acc',
    confidence_type: Optional[str] = None,
    chosen_regimes: RegimeChoice = RegimeChoice()
):
    if visualize_sdn is None:
        visualize_sdn = train_sdn
    if visualize_gpf is None:
        visualize_gpf = train_gpf
    if visualize_pbee is None:
        visualize_pbee = train_pbee

    plot_dependencies = []
    if any([chosen_regimes.disjoint, chosen_regimes.mixed]):
        trained_backbone_dependencies = trainer.train_base(
            f"{trainer.name()}_base",
            regime_config=DisjointConfiguration(),
            exp_ids=exp_ids,
            max_epochs_amount_per_phase=max_epochs_amount_per_phase,
            lr=lr,
            do_trainining=train_base,
        )
        plot_dependencies += trained_backbone_dependencies.values()
    if any([chosen_regimes.joint, chosen_regimes.alternating, chosen_regimes.branch_wise, chosen_regimes.separate, chosen_regimes.late_first]):
        untrained_backbone_dependencies = trainer.train_base(
            f"{trainer.name()}_stub",
            regime_config=JointConfiguration(),
            exp_ids=exp_ids,
            max_epochs_amount_per_phase=max_epochs_amount_per_phase,
            lr=lr,
            do_trainining=train_base
        )
        # plot_dependencies += untrained_backbone_dependencies.values()

    regime_configs: list[RegimeConfiguration] = []
    if chosen_regimes.disjoint:
        regime_configs.append(DisjointConfiguration())
    if chosen_regimes.mixed:
        regime_configs.append(MixedConfiguration())

    untrained_backbone_regime_configs: list[RegimeConfiguration] = []
    if chosen_regimes.joint:
        untrained_backbone_regime_configs.append(JointConfiguration()),
    if chosen_regimes.alternating:
        untrained_backbone_regime_configs.append(AlternatingConfiguration())
    if chosen_regimes.branch_wise:
        untrained_backbone_regime_configs.append(BranchWiseConfiguration())
    if chosen_regimes.late_first:
        untrained_backbone_regime_configs.append(LateFirstConfiguration())
    if chosen_regimes.separate:
        untrained_backbone_regime_configs.append(SeparateConfiguration())

    common_args = trainer.get_common_args()
    ee_configs = list(chain(regime_configs, untrained_backbone_regime_configs))

    sdn_plot_dependencies = deepcopy(plot_dependencies)
    sdn_regimes_dependencies: list[dict[int, Dependency]] = []
    for regime_config in regime_configs:
        dependencies = trainer.train_sdn(
            f"{trainer.name()}_{regime_config.regime_name()}",
            regime_config=regime_config,
            exp_ids=exp_ids,
            max_epochs_amount_per_phase=max_epochs_amount_per_phase,
            lr=lr,
            dependency_mapping=trained_backbone_dependencies,
            do_trainining=train_sdn
        )
        sdn_plot_dependencies += dependencies.values()
        sdn_regimes_dependencies.append(dependencies)

    for regime_config in untrained_backbone_regime_configs:
        dependencies = trainer.train_sdn(
            f"{trainer.name()}_{regime_config.regime_name()}",
            regime_config=regime_config,
            exp_ids=exp_ids,
            max_epochs_amount_per_phase=max_epochs_amount_per_phase,
            lr=lr,
            dependency_mapping=untrained_backbone_dependencies,
            do_trainining=train_sdn
        )
        sdn_plot_dependencies += dependencies.values()
        sdn_regimes_dependencies.append(dependencies)

    if visualize_sdn:
        visualize_ee(trainer, f'regimes_sdn_{trainer.name()}',
                     sdn_plot_dependencies, ee_configs, exp_ids, common_args.runs_dir, mode=mode, confidence_type=confidence_type)

    if train_gpf or visualize_gpf:
        gpf_plot_dependencies = deepcopy(plot_dependencies)
        for regime_config in regime_configs:
            dependencies = trainer.train_gpf(
                f"{trainer.name()}_{regime_config.regime_name()}_gpf",
                regime_config=regime_config,
                exp_ids=exp_ids,
                max_epochs_amount_per_phase=max_epochs_amount_per_phase,
                lr=lr,
                dependency_mapping=trained_backbone_dependencies,
                do_trainining=train_gpf
            )
            gpf_plot_dependencies += dependencies.values()

        for regime_config in untrained_backbone_regime_configs:
            dependencies = trainer.train_gpf(
                f"{trainer.name()}_{regime_config.regime_name()}_gpf",
                regime_config=regime_config,
                exp_ids=exp_ids,
                max_epochs_amount_per_phase=max_epochs_amount_per_phase,
                lr=lr,
                dependency_mapping=untrained_backbone_dependencies,
                do_trainining=train_gpf
            )
            gpf_plot_dependencies += dependencies.values()

    if visualize_gpf:
        visualize_ee(trainer, f'regimes_gpf_{trainer.name()}',
                     gpf_plot_dependencies, ee_configs, exp_ids, common_args.runs_dir, mode=mode, confidence_type=confidence_type)

    pbee_plot_dependencies = deepcopy(plot_dependencies)
    for regime_config, regime_dependency in zip(chain(regime_configs, untrained_backbone_regime_configs), sdn_regimes_dependencies):
        dependencies = trainer.train_pbee(
            f"{trainer.name()}_{regime_config.regime_name()}_pbee",
            regime_config=regime_config,
            exp_ids=exp_ids,
            max_epochs_amount_per_phase=max_epochs_amount_per_phase,
            lr=lr,
            dependency_mapping=regime_dependency,
            do_trainining=train_pbee
        )
        pbee_plot_dependencies += dependencies.values()

    if visualize_pbee:
        visualize_ee(trainer, f'regimes_pbee_{trainer.name()}',
                     pbee_plot_dependencies, ee_configs, exp_ids, common_args.runs_dir, mode=mode)
