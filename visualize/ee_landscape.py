import loss_landscapes as ll
from loss_landscapes.model_interface.model_parameters import ModelParameters
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.utils
import torch.utils.data


from accelerate import Accelerator
from common import LOSS_NAME_MAP
from datasets import DATASETS_NAME_MAP
from loss_landscapes_modification.random_plane import random_plane 
from metrics.ee_loss_wrappers import DataloaderListLoss, ListLoss
from metrics.ee_training_loss_wrapper import EETrainingLossWrapper
from utils import get_loader, load_model


class EEWrapper(ll.ModelWrapper):
    def __init__(self, model: torch.nn.Module):
        super().__init__([model])

    def forward(self, x):
        return self.modules[0](x)

    @property
    def number_of_heads(self):
        return self.modules[0].number_of_heads


class HeadsOnlyWrapper(EEWrapper):
    def get_module_parameters(self) -> ModelParameters:
        ee_model = self.modules[0]
        return ModelParameters([p for p in ee_model.head_modules().parameters()])


class BackboneOnlyWrapper(EEWrapper):
    def get_module_parameters(self) -> ModelParameters:
        ee_model = self.modules[0]
        return ModelParameters([p for p in ee_model._base_model.parameters()])


def save_landscape_plot(
    landscape,
    transform,
    clip_min,
    clip_max,
    directory,
    how_to_plot,
    search_zoom,
    loss_info,
    data_info,
    specific_info,
):
    match transform:
        case None:
            pass
        case "log":
            landscape = np.log(landscape)
    a_min = np.nanmin(landscape)
    if clip_min != None or clip_max != None:
        landscape = np.clip(landscape, a_min=clip_min, a_max=clip_max + a_min)

    loss_name = f"{transform}({loss_info})" if transform else loss_info

    match how_to_plot:
        case "3d":
            search_distance = 1.0 / search_zoom
            x_plot = np.linspace(-search_distance, search_distance, landscape.shape[0])
            y_plot = np.linspace(-search_distance, search_distance, landscape.shape[1])
            X_plot, Y_plot = np.meshgrid(x_plot, y_plot)

            fig = plt.figure()
            ax = plt.axes(projection="3d")
            surface = ax.plot_surface(X_plot, Y_plot, landscape, cmap="cool", alpha=0.8)

            ax.set_title(
                f"landscape {loss_name}, clip={clip_max} {data_info=}, {specific_info}",
                fontsize=8,
            )
            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel("y", fontsize=12)

            fig.colorbar(surface, shrink=0.5, aspect=5)

        case "contour":
            plt.contour(landscape, levels=50)
    directory.mkdir(parents=True, exist_ok=True)
    plt.savefig(directory / f"loss_landscape,{specific_info}.png")
    plt.clf()
    plt.close()


def generic_experiment_runner(args, inner_fn):
    accelerator = Accelerator(split_batches=True)

    for exp_name in args.exp_names:
        for exp_id in args.exp_ids:
            model, run_args, final_results = load_model(args, exp_name, exp_id)
            if "thresholds" in final_results:
                model = accelerator.prepare(model)
                del final_results["model_state"]
                _, train_data, id_data = DATASETS_NAME_MAP[run_args.dataset](
                    run_args.dataset_args
                )

                batch_size = args.batch_size or run_args.batch_size
                # testloader = get_loader(id_data, batch_size, accelerator)
                trainloader = get_loader(train_data, batch_size, accelerator)

                loss_cls = LOSS_NAME_MAP[run_args.loss_type]

                directory = (
                    args.output_dir
                    / f"{run_args.local_description}"
                    / f"{exp_name}_{exp_id}"
                )
                if args.only_backbone:
                    directory = directory / "backbone"
                directory.mkdir(parents=True, exist_ok=True)

                if args.only_heads:
                    model = HeadsOnlyWrapper(model)
                elif args.only_backbone:
                    model = BackboneOnlyWrapper(model)

                inner_fn(
                    run_args,
                    model,
                    accelerator,
                    trainloader,
                    batch_size,
                    loss_cls,
                    directory,
                )


def visualize_ee_training_loss_landscape(args):
    def inner(run_args, model, accelerator, loader, batch_size, loss_cls, directory):
        if args.use_full_loader:
            dataloader = loader
        else:
            X, y = next(iter(loader))
            from torch.utils.data import TensorDataset, DataLoader

            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=X.shape[0])

        loss_wrap = EETrainingLossWrapper(
            data_loader=dataloader,
            args=run_args,
            accelerator=accelerator,
            criterion=loss_cls(),
            gen_device=torch.get_default_device(),
        )

        with torch.device(accelerator.device):
            landscape = random_plane(
                model,
                loss_wrap,
                distance=args.zoom,
                steps=args.steps,
                normalization="filter",
                noise_type="gaussian"
            )

        with open(directory / "dump", "wb") as write_file:
            np.save(write_file, landscape)

        save_landscape_plot(
            landscape=landscape,
            transform=args.landscape_transform,
            clip_min=None,
            clip_max=None,
            directory=directory,
            how_to_plot=args.how_to_plot,
            search_zoom=args.zoom,
            loss_info=run_args.loss_type,
            data_info=("full dataloader" if args.use_full_loader else batch_size),
            specific_info=f"training loss,_{run_args.local_description}",
        )

        for clip in args.clips:
            save_landscape_plot(
                landscape=landscape,
                transform=args.landscape_transform,
                clip_min=None,
                clip_max=clip,
                directory=directory / "with_clips",
                how_to_plot=args.how_to_plot,
                search_zoom=args.zoom,
                loss_info=run_args.loss_type,
                data_info=("full dataloader" if args.use_full_loader else batch_size),
                specific_info=f"training loss,_{run_args.local_description},_{clip=}",
            )

    generic_experiment_runner(args, inner)


def visualize_ee_landscape(args):
    def inner(run_args, model, accelerator, loader, batch_size, loss_cls, directory):
        directory = directory / "heads"
        directory.mkdir(parents=True, exist_ok=True)

        if args.use_full_loader:
            loss_wrap = DataloaderListLoss(
                loss_cls(),
                loader,
                model.number_of_heads,
                gen_device=torch.get_default_device(),
            )
        else:
            X, y = next(iter(loader))
            loss_wrap = ListLoss(loss_cls(), X, y)

        with torch.device(accelerator.device):
            landscape = random_plane(
                model,
                loss_wrap,
                distance=args.zoom,
                steps=args.steps,
                normalization="filter",
                noise_type="gaussian"
            )

        with open(directory / "dump", "wb") as write_file:
            np.save(write_file, landscape)

        landscapes = [landscape[:, :, i] for i in range(model.number_of_heads)]

        for i, land in enumerate(landscapes):
            save_landscape_plot(
                landscape=land,
                transform=args.landscape_transform,
                clip_min=None,
                clip_max=None,
                directory=directory,
                how_to_plot=args.how_to_plot,
                search_zoom=args.zoom,
                loss_info=run_args.loss_type,
                data_info=("full dataloader" if args.use_full_loader else batch_size),
                specific_info=f"head{i}",
            )
            for clip in args.clips:
                save_landscape_plot(
                    landscape=land,
                    transform=args.landscape_transform,
                    clip_min=None,
                    clip_max=clip,
                    directory=directory,
                    how_to_plot=args.how_to_plot,
                    search_zoom=args.zoom,
                    loss_info=run_args.loss_type,
                    data_info=(
                        "full dataloader" if args.use_full_loader else batch_size
                    ),
                    specific_info=f"head{i}_{clip=}",
                )

    generic_experiment_runner(args, inner)


def visualize_ee_landscape_from_file(args):
    with open(args.directory / "dump", "rb") as file:
        landscape = np.load(file=file)
        landscapes = [landscape[:, :, i] for i in range(landscape.shape[-1])]
        for i, land in enumerate(landscapes):
            save_landscape_plot(
                landscape=land,
                transform=args.transform,
                clip_min=args.clip_min,
                clip_max=args.clip_max,
                directory=args.directory / args.how_to_plot,
                how_to_plot=args.how_to_plot,
                search_zoom=args.zoom,
                loss_info=args.loss_info,
                data_info=(
                    "full dataloader" if args.use_full_loader else args.batch_size
                ),
                specific_info=f"head{i}, clip={args.clip_max}",
            )
