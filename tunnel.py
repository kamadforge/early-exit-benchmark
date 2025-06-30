import os
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch


def get_every_but_forbidden_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_every_but_forbidden_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]

    result += list(model._parameters.keys())
    return result


FORBIDDEN_LAYER_TYPES = [torch.nn.Embedding, torch.nn.LayerNorm,
                         torch.nn.BatchNorm1d, torch.nn.BatchNorm2d]


class BaseAnalysis:
    def export(self, name):
        torch.save(self.result, os.path.join(self.rpath, name + ".pt"))

    def clean_up(self):
        for attr in self.attributes_on_gpu:
            try:
                a = getattr(self, attr)
                a.to("cpu")
                del a
            except AttributeError:
                pass
        del self
        torch.cuda.empty_cache()

    @abstractmethod
    def analysis(self):
        pass

    @abstractmethod
    def plot(self, path):
        pass


class RepresentationsSpectra(BaseAnalysis):

    def __init__(self, model, loader, modules_list, layers=None, rpath='.', MAX_REPR_SIZE=8000):
        self.model = model
        self.loader = loader
        self.names_of_layers_to_analyze = layers if layers is not None else [
            n for n, _ in model.named_modules()]
        self.modules_list = modules_list
        self.handels = []
        self._insert_hooks()
        self.representations = {}
        self.MAX_REPR_SIZE = MAX_REPR_SIZE
        self.rpath = rpath

        self.attributes_on_gpu = ["model"]
        self.logger = None
        self.is_able = False
        self.device = next(model.parameters()).device
        self.penalized_parameter_names = get_every_but_forbidden_parameter_names(
            self.model, FORBIDDEN_LAYER_TYPES)
        self.subsampling = {}
        self.eps = torch.finfo(torch.float32).eps

    def _spectra_hook(self, name):
        def spectra_hook(model, input, output):
            if self.is_able:
                output = output.flatten(start_dim=1)
                representation_size = output.shape[1]
                if name in self.subsampling:
                    output = torch.index_select(
                        output, 1, self.subsampling[name].to(self.device))
                elif representation_size > self.MAX_REPR_SIZE:
                    self.subsampling[name] = torch.randperm(representation_size)[
                        :self.MAX_REPR_SIZE].sort()[0]
                    output = torch.index_select(
                        output, 1, self.subsampling[name].to(self.device))

                self.representations[name] = self.representations.get(
                    name, []) + [output]
        return spectra_hook

    def _insert_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.names_of_layers_to_analyze:
                if any(isinstance(module, module_type) for module_type in self.modules_list):
                    self.handels.append(
                        module.register_forward_hook(self._spectra_hook(name)))

    def disable(self):
        self.is_able = False

    def enable(self):
        self.is_able = True

    @torch.no_grad()
    def collect_representations(self):
        self.model.eval()
        y_true = torch.empty((0,)).to(next(self.model.parameters()).device)
        with torch.no_grad():
            for x_true, y_data in self.loader:

                _ = self.model(x_true)
                y_true = torch.cat((y_true, y_data))
        for name, rep in self.representations.items():
            self.representations[name] = torch.cat(rep, dim=0).detach()
        self.model.train()
        return y_true

    def collect_weights(self):
        named_weights = {n: p.reshape(p.size(0), -1) for n, p in self.model.named_parameters(
        ) if 'weight' in n and n in self.penalized_parameter_names}
        return named_weights

    def analysis(self, step, scope, phase, tc):
        main_prefix = f'ranks_representations_{phase}'
        postfix = f'____{scope}____{phase}'
        y_true = self.collect_representations()

        from collections import Counter
        print(Counter(y_true.cpu().numpy()))

        evaluators1 = {}
        evaluators2 = {}
        prefix1 = main_prefix
        prefix2 = f'square_stable_{main_prefix}'
        for name, rep in self.representations.items():
            name_dict1 = f'{prefix1}/{name}{postfix}'
            name_dict2 = f'{prefix2}/{name}{postfix}'
            rep = torch.cov(rep.T)
            singular_squared = torch.linalg.eig(rep)[0].float()
            max_sigma = max(singular_squared)
            threshold = rep.size(0) * self.eps * max_sigma
            evaluators1[name_dict1] = (
                (singular_squared > threshold).sum()).item()
            square_stable_rank = singular_squared.sum() / max_sigma
            evaluators2[name_dict2] = square_stable_rank.item()

        self.plot(evaluators1, prefix1, postfix, tc, step)
        self.plot(evaluators2, prefix2, postfix, tc, step)

        evaluators = evaluators1 | evaluators2

        variance_eucl(self.representations, y_true,
                      evaluators, phase=phase, eps=self.eps)

        if phase == 'train':
            main_prefix = f'ranks_weights_branch'
            postfix = f'____{scope}____{phase}'
            named_weights = self.collect_weights()

            evaluators3 = {}
            evaluators4 = {}
            prefix3 = main_prefix
            prefix4 = f'square_stable_{main_prefix}'
            for name, weights in named_weights.items():
                name_dict3 = f'{prefix3}/{name}{postfix}'
                name_dict4 = f'{prefix4}/{name}{postfix}'
                weights = torch.cov(weights)
                singular_squared = torch.linalg.eig(weights)[0].float()
                max_sigma = max(singular_squared)
                threshold = weights.size(0) * self.eps * max_sigma
                evaluators3[name_dict3] = (
                    (singular_squared > threshold).sum()).item()
                square_stable_rank = singular_squared.sum() / max_sigma
                evaluators4[name_dict4] = square_stable_rank.item()

            self.plot(evaluators3, prefix3, postfix, tc, step)
            self.plot(evaluators4, prefix4, postfix, tc, step)

            evaluators = evaluators | evaluators3 | evaluators4

        evaluators[f'steps/ranks_{phase}'] = step
        tc.writer.log_scalars(evaluators)

        self.representations = {}
        self.subsampling = {}
        torch.cuda.empty_cache()

    def plot(self, evaluators, prefix, postfix, tc):
        plot_name = f'{prefix}_plots/{postfix}'
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        axs.plot(list(range(len(evaluators))), list(evaluators.values()), "o-")

        axs.set_title("Rank Across Layers")
        axs.set_xlabel("Layer")
        axs.set_ylabel("Rank")
        plot_images = {plot_name: fig}
        tc.writer.log_plots(plot_images)
        plt.close()


def variance_eucl(x_data, y_true, evaluators, phase, eps, batch_size=50):
    y_true = y_true.cpu()
    classes = np.unique(y_true.numpy())
    denom_class = classes.shape[0]
    for name, internal_repr in x_data.items():
        within_class_cov = 0.0
        between_class_cov = 0.0
        general_mean = torch.mean(internal_repr, dim=0, keepdim=True)
        for c in classes:
            class_internal_repr = internal_repr[y_true == c]
            class_mean = torch.mean(class_internal_repr, dim=0, keepdim=True)
            class_internal_repr_sub = class_internal_repr - class_mean

            S = batch_size
            K = class_internal_repr_sub.shape[0] // S
            within_class_cov += sum([(class_internal_repr_sub[S*i:S*(i+1)].unsqueeze(
                2) @ class_internal_repr_sub[S*i:S*(i+1)].unsqueeze(1)).mean(dim=0) for i in range(K)]) / K
            between_sample = (class_mean - general_mean)
            between_class_cov += between_sample.T @ between_sample

        within_class_cov /= denom_class
        between_class_cov /= denom_class
        total_class_cov = within_class_cov + between_class_cov

        trace_wcc = torch.trace(within_class_cov)
        trace_bcc = torch.trace(between_class_cov)
        trace_tcc = torch.trace(total_class_cov)

        evaluators[f'bcc_square_stable_rank_{phase}/{name}'] = trace_bcc.item()
        evaluators[f'tcc_square_stable_rank_{phase}/{name}'] = trace_tcc.item()
        evaluators[f'wwc_square_stable_rank_{phase}/{name}'] = trace_wcc.item()

        normalized_wcc = trace_wcc / (trace_tcc + eps)
        normalized_bcc = trace_bcc / (trace_tcc + eps)

        evaluators[f'within_cov_eucl_normalized_{phase}/{name}'] = normalized_wcc.item(
        )
        evaluators[f'between_cov_eucl_normalized_{phase}/{name}'] = normalized_bcc.item(
        )
