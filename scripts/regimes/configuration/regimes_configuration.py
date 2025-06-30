from abc import ABC, abstractmethod

from omegaconf import DictConfig


class RegimeConfiguration(ABC):
    @abstractmethod
    def update_base_config(self, config: DictConfig, max_epochs_amount_per_phase: int):
        pass

    @abstractmethod
    def update_ee_config(self, config: DictConfig, max_epochs_amount_per_phase: int):
        pass

    @abstractmethod
    def regime_name(self) -> str:
        pass


class DisjointConfiguration(RegimeConfiguration):
    def update_base_config(self, config: DictConfig, max_epochs_amount_per_phase: int):
        config.epochs = max_epochs_amount_per_phase

    def update_ee_config(self, config: DictConfig, max_epochs_amount_per_phase: int):
        config.phase2 = 0
        config.phase3 = max_epochs_amount_per_phase
        config.epochs = config.phase2 + config.phase3
        config.epochs_common = config.phase2
        config.with_backbone = True if config.phase2 != 0 else False
        config.with_phase3 = True if config.phase3 != 0 else False

    def regime_name(self) -> str:
        return "Disjoint"


class JointConfiguration(RegimeConfiguration):

    def update_base_config(self, config: DictConfig, max_epochs_amount_per_phase: int):
        config.epochs = 0

    def update_ee_config(self, config: DictConfig, max_epochs_amount_per_phase: int):
        config.phase2 = max_epochs_amount_per_phase
        config.phase3 = 0
        config.epochs = config.phase2 + config.phase3
        config.epochs_common = config.phase2
        config.with_backbone = True if config.phase2 != 0 else False
        config.with_phase3 = True if config.phase3 != 0 else False

    def regime_name(self) -> str:
        return "Joint"


class MixedConfiguration(RegimeConfiguration):

    def update_base_config(self, config: DictConfig, max_epochs_amount_per_phase: int):
        config.epochs = max_epochs_amount_per_phase

    def update_ee_config(self, config: DictConfig, max_epochs_amount_per_phase: int):
        config.phase2 = max_epochs_amount_per_phase
        config.phase3 = 0
        config.epochs = config.phase2 + config.phase3
        config.epochs_common = config.phase2
        config.with_backbone = True if config.phase2 != 0 else False
        config.with_phase3 = True if config.phase3 != 0 else False

    def regime_name(self) -> str:
        return "Mixed"


class AlternatingConfiguration(RegimeConfiguration):

    def update_base_config(self, config: DictConfig, max_epochs_amount_per_phase: int):
        config.epochs = 0

    def update_ee_config(self, config: DictConfig, max_epochs_amount_per_phase: int):
        config.epochs = max_epochs_amount_per_phase
        config.with_backbone = True

    def regime_name(self) -> str:
        return "Alternating"


class BranchWiseConfiguration(RegimeConfiguration):

    def update_base_config(self, config: DictConfig, max_epochs_amount_per_phase: int):
        config.epochs = 0

    def update_ee_config(self, config: DictConfig, max_epochs_amount_per_phase: int):
        config.epochs = max_epochs_amount_per_phase
        config.with_backbone = True

    def regime_name(self) -> str:
        return "BranchWise"


class SeparateConfiguration(RegimeConfiguration):

    def update_base_config(self, config: DictConfig, max_epochs_amount_per_phase: int):
        config.epochs = 0

    def update_ee_config(self, config: DictConfig, max_epochs_amount_per_phase: int):
        config.epochs = max_epochs_amount_per_phase
        config.with_backbone = True

    def regime_name(self) -> str:
        return "Separate"


class LateFirstConfiguration(RegimeConfiguration):

    def update_base_config(self, config: DictConfig, max_epochs_amount_per_phase: int):
        config.epochs = 0

    def update_ee_config(self, config: DictConfig, max_epochs_amount_per_phase: int):
        config.epochs = max_epochs_amount_per_phase
        config.with_backbone = True

    def regime_name(self) -> str:
        return "LateFirst"
