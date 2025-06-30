import base64
import hashlib
import logging
import numbers
import warnings
from collections import Counter, defaultdict
from functools import reduce, partial
from pathlib import Path
from random import randint
from typing import Union, Tuple, List, Dict, Set, FrozenSet, Callable, Iterator, Any, Optional

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import patch_environment, PrepareForLaunch
from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import elementwise_flop_counter, get_shape
from omegaconf import OmegaConf
from torch import nn, Tensor, inf

from datasets_config import DATASETS_NAME_MAP


def clip_by_norm_(tensors, max_norm: float, norm_type: float = 2.0):
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(tensors) == 0:
        return torch.tensor(0.)
    if norm_type == inf:
        norms = [t.detach().abs().max() for t in tensors]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        norms = [torch.norm(t, norm_type) for t in tensors]
        total_norm = torch.norm(torch.stack([norm for norm in norms]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for t in tensors:
        t.detach().mul_(clip_coef_clamped)
    return total_norm


class BCEWithLogitsLossWrapper(nn.BCEWithLogitsLoss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean',
                 pos_weight: Optional[Tensor] = None):
        super().__init__(weight, size_average, reduce, reduction, pos_weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if input.dim() - 1 == target.dim():
            target = nn.functional.one_hot(target, num_classes=input.size(-1)).to(input.dtype)
        return nn.functional.binary_cross_entropy_with_logits(input, target,
                                                              self.weight,
                                                              pos_weight=self.pos_weight,
                                                              reduction=self.reduction)


# mixup code from:
# https://github.com/huggingface/pytorch-image-models/blob/624266148d8fa5ddb22a6f5e523a53aaf0e8a9eb/timm/data/mixup.py#L90


def one_hot(x, num_classes, on_value=1., off_value=0.):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=x.device).scatter_(1, x, on_value)


def mixup_target(target, num_classes, lam=1., smoothing=0.0):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value)
    return y1 * lam + y2 * (1. - lam)


def rand_bbox(img_shape, lam, margin=0., count=None):
    """ Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def rand_bbox_minmax(img_shape, minmax, count=None):
    """ Min-Max CutMix bounding-box
    Inspired by Darknet cutmix impl, generates a random rectangular bbox
    based on min/max percent values applied to each dimension of the input image.

    Typical defaults for minmax are usually in the  .2-.3 for min and .8-.9 range for max.

    Args:
        img_shape (tuple): Image shape as tuple
        minmax (tuple or list): Min and max bbox ratios (as percent of image size)
        count (int): Number of bbox to generate
    """
    assert len(minmax) == 2
    img_h, img_w = img_shape[-2:]
    cut_h = np.random.randint(int(img_h * minmax[0]), int(img_h * minmax[1]), size=count)
    cut_w = np.random.randint(int(img_w * minmax[0]), int(img_w * minmax[1]), size=count)
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yu = yl + cut_h
    xu = xl + cut_w
    return yl, yu, xl, xu


def cutmix_bbox_and_lam(img_shape, lam, ratio_minmax=None, correct_lam=True, count=None):
    """ Generate bbox and apply lambda correction.
    """
    if ratio_minmax is not None:
        yl, yu, xl, xu = rand_bbox_minmax(img_shape, ratio_minmax, count=count)
    else:
        yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam or ratio_minmax is not None:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam


class Mixup:
    """ Mixup/Cutmix that applies different params to each element or whole batch

    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
        prob (float): probability of applying mixup or cutmix per batch or element
        switch_prob (float): probability of switching to cutmix instead of mixup when both are active
        mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        correct_lam (bool): apply lambda correction when cutmix bbox clipped by image borders
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
    """

    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=1000):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)

    def _params_per_elem(self, batch_size):
        lam = np.ones(batch_size, dtype=np.float32)
        use_cutmix = np.zeros(batch_size, dtype=bool)
        if self.mixup_enabled:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand(batch_size) < self.switch_prob
                lam_mix = np.where(
                    use_cutmix,
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size),
                    np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size))
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
            elif self.cutmix_alpha > 0.:
                use_cutmix = np.ones(batch_size, dtype=bool)
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = np.where(np.random.rand(batch_size) < self.mix_prob, lam_mix.astype(np.float32), lam)
        return lam, use_cutmix

    def _params_per_batch(self):
        lam = 1.
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
                    np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_elem(self, x):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size)
        x_orig = x.clone()  # need to keep an unmodified original for mixing source
        for i in range(batch_size):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)

    def _mix_pair(self, x):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
        x_orig = x.clone()  # need to keep an unmodified original for mixing source
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    x[j][:, yl:yh, xl:xh] = x_orig[i][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
                    x[j] = x[j] * lam + x_orig[i] * (1 - lam)
        lam_batch = np.concatenate((lam_batch, lam_batch[::-1]))
        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)

    def _mix_batch(self, x):
        lam, use_cutmix = self._params_per_batch()
        if lam == 1.:
            return 1.
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                x.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
            x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh]
        else:
            x_flipped = x.flip(0).mul_(1. - lam)
            x.mul_(lam).add_(x_flipped)
        return lam

    def __call__(self, x, target):
        if len(x) % 2 != 0:
            warnings.warn('Batch size is not even as required by mixup - skipping last element.')
            x, target = x[:-1], target[:-1]
        if self.mode == 'elem':
            lam = self._mix_elem(x)
        elif self.mode == 'pair':
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing)
        return x, target


def coefficient_of_variation(x, dim, unbiased):
    return x.std(dim=dim, unbiased=unbiased) / x.mean(dim=dim).abs()


def gumbel_sigmoid(x, tau=1.0, noise=True, eps=1e-10):
    if noise:
        u1, u2 = torch.rand_like(x), torch.rand_like(x)
        g1, g2 = -torch.log(-torch.log(u1 + eps) + eps), -torch.log(-torch.log(u2 + eps) + eps)
        x = x + g1 - g2
    res = torch.sigmoid(x / tau)
    res = ((res >= 0.5).to(x.dtype) - res).detach() + res
    assert not torch.any(torch.isnan(res))
    return res


def get_lrs(optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups]


def unfrozen_parameters(module: nn.Module) -> Iterator[nn.Parameter]:
    return (p for p in module.parameters() if p.requires_grad == True)


def find_module_names(model: nn.Module, filter: Callable[[nn.Module, nn.Module], bool]):
    found_names = []
    for name, module in model.named_modules():
        if filter(model, module):
            found_names.append(name)
    return found_names


def get_module_name(module: nn.Module, submodule: nn.Module):
    for name, m in module.named_modules():
        if m is submodule:
            return name


def get_module_by_name(module: nn.Module, name: str):
    if not name:
        return module
    else:
        names = name.split(sep='.')
        return reduce(getattr, names, module)


def get_parent_module_name(name: str):
    names = name.split(sep='.')
    return '.'.join(names[:-1])


def set_module_by_name(module: nn.Module, name: str, replacement):
    names = name.split(sep='.')
    parent = reduce(getattr, names[:-1], module)
    setattr(parent, names[-1], replacement)


def set_parameter_by_name(module: nn.Module, name: str, replacement):
    names = name.split(sep='.')
    parent = reduce(getattr, names[:-1], module)
    parent._parameters[names[-1]] = replacement


def add_save_activations_hook(model, module_names):
    module_inputs = {}
    module_outputs = {}
    module_id_to_name = {}
    hook_handles = []
    for name in module_names:
        module = get_module_by_name(model, name)
        module_id_to_name[id(module)] = name

        def save_activations_hook(m, input, output):
            module_name = module_id_to_name[id(m)]
            # TODO will multiple inputs/outputs ever be used?
            module_inputs[module_name] = input
            module_outputs[module_name] = output

        handle = module.register_forward_hook(save_activations_hook)
        hook_handles.append(handle)
    return module_inputs, module_outputs, hook_handles


def add_save_inputs_hook(model, module_names):
    module_inputs = {}
    module_id_to_name = {}
    hook_handles = []
    for name in module_names:
        module = get_module_by_name(model, name)
        module_id_to_name[id(module)] = name

        def save_activations_hook(m, input, output):
            module_name = module_id_to_name[id(m)]
            # TODO will multiple inputs/outputs ever be used?
            module_inputs[module_name] = input

        handle = module.register_forward_hook(save_activations_hook)
        hook_handles.append(handle)
    return module_inputs, hook_handles


def add_save_outputs_hook(model, module_names):
    module_outputs = {}
    module_id_to_name = {}
    hook_handles = []

    for name in module_names:
        module = get_module_by_name(model, name)
        module_id_to_name[id(module)] = name

        def save_gating_hook(m, _input, output):
            module_name = module_id_to_name[id(m)]
            # TODO will multiple inputs/outputs ever be used?
            module_outputs[module_name] = output

        handle = module.register_forward_hook(save_gating_hook)
        hook_handles.append(handle)
    return module_outputs, hook_handles


def remove_hooks(handles):
    for handle in handles:
        handle.remove()


# def count_flops_of_some_op(inputs: List[Any], outputs: List[Any]) -> typing.Counter[str]:
#     # counting helper - only some of these inputs and outputs are relevant
#     print(f'inputs len: {len(inputs)}')
#     print(f'outputs len: {len(outputs)}')
#     for i, t in enumerate(inputs):
#         print(f'input {i}: {t}')
#         print(f'input {i} shape: {get_shape(t)}')
#     for i, t in enumerate(outputs):
#         print(f'output {i}: {t}')
#         print(f'output {i} shape: {get_shape(t)}')
#     # calculate ops here
#     # return them here
#     return Counter(
#         {"some_op": 42}
#     )

def count_scaled_dot_product_attention_ops(inputs: List[Any], _outputs: List[Any]) -> Counter[str]:
    q_samples, q_heads, q_tokens, q_dim = get_shape(inputs[0])
    k_samples, k_heads, k_tokens, k_dim = get_shape(inputs[1])
    v_samples, v_heads, v_tokens, v_dim = get_shape(inputs[2])
    assert q_dim == k_dim
    assert q_tokens == k_tokens == v_tokens
    assert q_heads == k_heads == v_heads
    assert q_samples == k_samples == v_samples
    # query @ key.transpose(-2, -1)
    q_k_matmul_ops = q_samples * q_heads * q_tokens * q_tokens * q_dim
    # / math.sqrt(query.size(-1))
    div_ops = q_samples * q_heads * q_tokens * q_tokens
    # + attn_mask
    if get_shape(inputs[3]) is not None or isinstance(inputs[3], numbers.Number):
        mask_add_ops = q_samples * q_heads * q_tokens * q_tokens
    else:
        mask_add_ops = 0
    # torch.softmax(..., dim=-1)
    softmax_ops = 2 * q_samples * q_heads * q_tokens * q_tokens
    # attn_weight @ value
    v_matmul_ops = q_samples * q_heads * q_tokens * q_tokens * v_dim
    return Counter({
        'matmul': q_k_matmul_ops + v_matmul_ops,
        'div': div_ops,
        'add': mask_add_ops,
        'softmax': softmax_ops
    })


OP_HANDLERS = {
    # TODO verify correctness of these
    'aten::add': elementwise_flop_counter(0, 1),
    'aten::add_': elementwise_flop_counter(0, 1),
    'aten::radd': elementwise_flop_counter(0, 1),
    'aten::sub': elementwise_flop_counter(0, 1),
    'aten::sub_': elementwise_flop_counter(0, 1),
    'aten::rsub': elementwise_flop_counter(0, 1),
    'aten::mul': elementwise_flop_counter(0, 1),
    'aten::mul_': elementwise_flop_counter(0, 1),
    'aten::rmul': elementwise_flop_counter(0, 1),
    'aten::div': elementwise_flop_counter(0, 1),
    'aten::div_': elementwise_flop_counter(0, 1),
    'aten::rdiv': elementwise_flop_counter(0, 1),
    'aten::exp': elementwise_flop_counter(0, 1),
    'aten::cumsum': elementwise_flop_counter(0, 1),
    'aten::ne': elementwise_flop_counter(0, 1),
    'aten::lt': elementwise_flop_counter(0, 1),
    'aten::gelu': elementwise_flop_counter(0, 1),
    'aten::silu_': elementwise_flop_counter(0, 1),
    'aten::dropout_': elementwise_flop_counter(0, 1),
    'aten::sigmoid': elementwise_flop_counter(0, 1),
    'aten::tanh': elementwise_flop_counter(0, 1),
    'aten::softmax': elementwise_flop_counter(0, 2),
    'aten::log_softmax': elementwise_flop_counter(0, 2),
    'aten::argmax': elementwise_flop_counter(0, 1),
    'aten::one_hot': elementwise_flop_counter(0, 1),
    'aten::flatten': elementwise_flop_counter(0, 0),
    'aten::unflatten': elementwise_flop_counter(0, 0),
    'aten::mean': elementwise_flop_counter(1, 0),
    'aten::sum': elementwise_flop_counter(1, 0),
    # TODO write a proper counter for these
    'aten::topk': elementwise_flop_counter(1, 1),
    'aten::scatter': elementwise_flop_counter(1, 1),
    'aten::scatter_': elementwise_flop_counter(1, 1),
    'aten::gather': elementwise_flop_counter(1, 1),
    'aten::gather_': elementwise_flop_counter(1, 1),
    'aten::adaptive_max_pool2d': elementwise_flop_counter(1, 0),
    # custom
    'aten::scaled_dot_product_attention': count_scaled_dot_product_attention_ops,
    # 'aten::_native_multi_head_attention': count_native_attention_ops,
}


def flop_count(model: torch.nn.Module, input) -> FlopCountAnalysis:
    return FlopCountAnalysis(model, input).set_op_handle(**OP_HANDLERS)


def get_loader(data: torch.utils.data.Dataset, batch_size: int, accelerator: Accelerator,
               shuffle: bool = True, num_workers: int = 16, pin: bool = True):
    loader = torch.utils.data.DataLoader(dataset=data,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         pin_memory=pin,
                                         num_workers=num_workers)
    return accelerator.prepare(loader)


def make_hashable(o: Union[Tuple, List, Dict, Set, FrozenSet]) -> Union[Tuple, List, Dict, Set, FrozenSet]:
    if isinstance(o, (tuple, list)):
        return tuple((make_hashable(e) for e in o))
    if isinstance(o, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in o.items()))
    if isinstance(o, (set, frozenset)):
        return tuple(sorted(make_hashable(e) for e in o))
    return o


def make_hash_sha256(o: Union[Tuple, List, Dict, Set, FrozenSet]) -> str:
    hasher = hashlib.sha256()
    hasher.update(repr(make_hashable(o)).encode())
    return base64.b32encode(hasher.digest()).decode()


def generate_run_name(args) -> Tuple[str, str]:
    # Properties from the config that are NOT to be included into hashing.
    omitted_keys = ['exp_id', 'runs_dir', 'eval_points', 'eval_batches', 'save_every', 'use_wandb', 'eval_thresholds',
                    'k_to_eval']
    args_dict = OmegaConf.to_container(args, resolve=True)
    hashed_flags = {k: v for k, v in args_dict.items() if k not in omitted_keys and v is not None}
    short_hash = make_hash_sha256(hashed_flags)[:8]
    exp_name = f'{args.dataset}_{args.model_class}_{short_hash}'
    run_name = f'{exp_name}_{args.exp_id}'
    logging.info(f'Generated run name: {run_name}')
    return exp_name, run_name


def get_run_id(run_name: str):
    api = wandb.Api()
    entity = os.environ['WANDB_ENTITY']
    project = os.environ['WANDB_PROJECT']
    retrieved_runs = api.runs(f'{entity}/{project}', filters={'display_name': run_name})
    logging.info(f'Retrieved {len(retrieved_runs)} for run_name: {run_name}')
    assert len(retrieved_runs) <= 1, f'retrieved_runs: {retrieved_runs}'
    if len(retrieved_runs) == 1:
        return retrieved_runs[0].id


def load_state(accelerator: Accelerator, state_path: Path):
    # TODO add wandb support?
    if state_path.exists() and state_path.is_dir():
        accelerator.load_state(state_path)
    elif accelerator.is_main_process:
        logging.info(f'No state file found - training from scratch.')


def save_state(accelerator: Accelerator, state_path: Path):
    # TODO add wandb support?
    logging.info(f'Saving state.')
    accelerator.save_state(state_path)


def retrieve_final(args, run_name: str, device: Union[torch.device, str] = 'cpu'):
    final_path = args.runs_dir / run_name / f'final.pth'
    if final_path.exists() and final_path.is_file():
        state = torch.load(final_path, map_location=device)
        logging.info(f'Loaded state for {run_name} from {str(final_path)}')
    elif args.use_wandb:
        api = wandb.Api()
        entity = os.environ['WANDB_ENTITY']
        project = os.environ['WANDB_PROJECT']
        retrieved_runs = api.runs(f'{entity}/{project}', filters={'display_name': run_name})
        assert len(retrieved_runs) <= 1, f'retrieved_runs: {retrieved_runs}'
        assert len(retrieved_runs) > 0, f'Run {run_name} not found'
        run = retrieved_runs[0]
        run.file('final.pth').download(root=str(final_path.parent), replace=True)
        state = torch.load(final_path, map_location=device)
        logging.info(f'Loaded state for {run_name} from W&B')
    else:
        raise FileNotFoundError('Cannot find the final.pth file')
    return state


def save_final(args, final_path, final_results):
    torch.save(final_results, final_path)
    logging.info(f'Saved final results to {str(final_path)}')
    # if args.use_wandb:
    #     wandb.save(str(final_path))
    #     logging.info(f'Saved final results to W&B')


def create_model(model_class, model_args):
    from common import MODEL_NAME_MAP
    return MODEL_NAME_MAP[model_class](**model_args)


def load_model(args, exp_name: str, exp_id: str, device: Union[torch.device, str] = 'cpu'):
    # TODO refactor this monster somehow :)
    run_states = []
    run_args = []
    run_model_args = []
    while exp_name is not None:
        state = retrieve_final(args, f'{exp_name}_{exp_id}', device)
        arg = state['args']
        model_arg = arg.model_args
        run_states.append(state)
        run_args.append(arg)
        run_model_args.append(model_arg)
        exp_name = arg.base_on if hasattr(arg, 'base_on') else None
        exp_id = arg.exp_id
    # create model from the most nested base model
    for arg, model_arg in zip(reversed(run_args), reversed(run_model_args)):
        if arg.base_on is None:
            model = create_model(arg.model_class, model_arg).to(device)
        else:
            model_arg = {'base_model': model, **model_arg}
            model = create_model(arg.model_class, model_arg).to(device)
    # state loading is only necessary for the model being loaded
    state_dict = run_states[0]['model_state']
    model.load_state_dict(state_dict)
    return model, run_args[0], run_states[0]


def load_run(args, run_name: str, device: Union[torch.device, str] = 'cpu'):
    split_run_name = run_name.split('_')
    exp_id = split_run_name[-1]
    exp_name = '_'.join(split_run_name[:-1])
    return load_model(args, exp_name, exp_id, device)


def accelerate_launcher(function, num_processes=None, mixed_precision="no", use_port="29500"):
    from torch.multiprocessing import start_processes
    from torch.multiprocessing.spawn import ProcessRaisedException
    if len(AcceleratorState._shared_state) > 0:
        raise ValueError(
            "To launch a multi-GPU training, the `Accelerator` should only be initialized "
            "inside your training function."
        )
    if torch.cuda.is_initialized():
        raise ValueError(
            "To launch a multi-GPU training, you need to avoid running any instruction "
            "using `torch.cuda` before starting DDP/accelerate."
        )
    with patch_environment(
            world_size=num_processes, master_addr="127.0.0.1", master_port=use_port, mixed_precision=mixed_precision
    ):
        launcher = PrepareForLaunch(function, distributed_type="MULTI_GPU")
        print(f"Launching training on {num_processes} GPUs.")
        try:
            start_processes(launcher, nprocs=num_processes, start_method="spawn")
            print(f"All processes exited")
        except ProcessRaisedException as e:
            print(f"ProcessRaisedException: {e}")


def submit_job(executor, job_func, *job_args, num_gpus=1, gpu_type=''):
    executor.update_parameters(slurm_gpus_per_task=f'{gpu_type}{num_gpus}')
    if 0 <= num_gpus <= 1:
        return executor.submit(job_func, *job_args)
    elif num_gpus > 1:
        # TODO check if port is available first (without TOCTOU, somehow)
        distributed_port = 29500 + randint(0, 30000)
        print(f'Choosing random port for accelerate multi-gpu: {distributed_port}')
        job_partial = partial(job_func, *job_args)
        return executor.submit(accelerate_launcher, job_partial, num_processes=num_gpus, use_port=distributed_port)


def train_val_split(dataset, test_size=0.1, seed=42, stratify_by_labels=True):
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(dataset))

    labels = dataset.targets
    train_indices, val_indices = train_test_split(indices, test_size=test_size, stratify=labels if stratify_by_labels else None, random_state=seed)

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    return train_subset, val_subset


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
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


FORBIDDEN_LAYER_TYPES = [torch.nn.Embedding, torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d]


def gradients_norm(tb_logger, module, step, is_ee=False, norm_type=2.0):
    prefix = 'ic_' if is_ee else ''
    allowed_parameter_names = get_every_but_forbidden_parameter_names(module, FORBIDDEN_LAYER_TYPES)
    evaluators = defaultdict(float)
    named_parameters = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
    grad_norm_layers = []
    for n, p in named_parameters:
        if p.grad is not None:
            grad_norm_per_layer = torch.norm(p.grad, norm_type)
            evaluators[f'gradient_norm/{prefix}{n}'] = grad_norm_per_layer.item()
            # evaluators[f'gradient_norm/{prefix}{n}'] = grad_norm_per_layer.item()
            if n in allowed_parameter_names:
                grad_norm_layers.append(grad_norm_per_layer)
    grad_norm = torch.norm(torch.stack(grad_norm_layers), norm_type) if len(grad_norm_layers) != 0 else torch.tensor(
        0.0)
    evaluators[f'gradient_norm/{prefix}overall'] = grad_norm.item()
    # evaluators[f'gradient_norm/{prefix}overall'] = grad_norm.item()
    # dopisz zbieranie zakumulowanego gradientu dla każdego z bloków
    tb_logger.log_scalars(evaluators)


def count_monotonical_intervals(numbers):
    """
    Zlicza zmiany monotoniczności w ciągu liczb.

    :param numbers: Lista liczb.
    :return: Liczba zmian monotoniczności.
    """
    if len(numbers) < 2:
        return 0

    # Inicjalizacja zmiennych
    changes = 0
    direction = 0  # 0: nieokreślona, 1: rosnąca, -1: malejąca

    for i in range(1, len(numbers)):
        if numbers[i] > numbers[i - 1] and direction != 1:
            changes += 1
            direction = 1
        elif numbers[i] < numbers[i - 1] and direction != -1:
            changes += 1
            direction = -1

    return changes


from copy import deepcopy
from scipy.spatial.distance import cosine


class GradientStatisticianEE:
    # NARAZIE BEZ BIASÓW
    # WYSZUKUJE ICS POPRZEZ ZAŁOŻENIE ŻE NAZYWAJĄ SIĘ _fc
    def __init__(self, held_out, norm_type=2.0):
        self.held_out = held_out
        self.norm_type = norm_type
        self.model_zero = None
        self.model_last_checkpoint = None
        self.logger = None
        self.batch_size_gsee = 500

    def gather_pred(self, model, phase, device):
        held_out_x = self.held_out[f'x_{phase}'].to(device)
        y_preds = [torch.tensor([], device=device) for _ in range(model.number_of_attached_heads + 1)]
        for i in range(held_out_x.shape[0] // self.batch_size_gsee):
            x_data = held_out_x[i * self.batch_size_gsee: (i + 1) * self.batch_size_gsee]
            preds = model(x_data)
            for i in range(len(preds)):
                y_preds[i] = torch.cat([y_preds[i], preds[i]], axis=0)
        return y_preds

    def gather_gradients(self, model, criterion, phase):
        device = next(model.parameters()).device
        y_true = self.held_out[f'y_{phase}'].to(device)
        # for param in model.parameters():
        #     print('1', param.grad)
        y_preds = self.gather_pred(model, phase, device)  # number_of_heads x batch_size x classes_num
        # for param in model.parameters():
        #     print('2', param.grad)
        losses = [criterion(y_pred, y_true) for y_pred in y_preds]  # loss for each head

        named_params = self.get_refined_params(model)  # only trainable weights

        gradients_per_param_name = defaultdict(list)
        gradients_per_ic = defaultdict(list)
        # for param in model.parameters():
        #     print('3', param.grad)

        for i, loss in enumerate(losses):
            # print('loss.requires_grad', loss.requires_grad)
            grads = torch.autograd.grad(loss, list(named_params.values()), retain_graph=True, allow_unused=True)
            # grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, allow_unused=True)
            for (param_name, _), grad in zip(named_params.items(), grads):
                if grad is not None and 'weight' in param_name:
                    grad = grad.flatten()
                    gradients_per_param_name[param_name].append(grad)
                    gradients_per_ic[i].append((param_name, grad))

        # gradients_per_param_name['ics'] = [v[0] for k,v in gradients_per_param_name.items() if '_fc' in k]  # różne ics mogą mieć różne wymiary
        gradients_per_param_name = {k: v for k, v in gradients_per_param_name.items() if
                                    len(v) > 1}  # I accept only non singular lists (all but gradients after the last attached IC)
        return gradients_per_param_name, gradients_per_ic

    def compute_gradient_stats(self, tc, phase, scope, step):
        unwrapped_model = tc.accelerator.unwrap_model(tc.model)
        unwrapped_model._base_model.eval()
        print(next(unwrapped_model.parameters()).grad)
        # unwrapped_model.zero_grad()  # resetujemy gradienty
        postfix = f'____{scope}____{phase}'

        evaluators = defaultdict(float)
        self.allowed_parameter_names = get_every_but_forbidden_parameter_names(unwrapped_model, FORBIDDEN_LAYER_TYPES)
        gradients_per_param_name, gradients_per_ic = self.gather_gradients(unwrapped_model, tc.criterion, phase)

        self.gradient_norm(gradients_per_ic, gradients_per_param_name, evaluators, postfix)
        self.gradient_direction_variance(gradients_per_param_name, evaluators, postfix)
        self.gradient_direction_error(gradients_per_param_name, evaluators, postfix)
        self.gradient_domination(gradients_per_param_name, evaluators, postfix)

        if phase == 'train':  # między fazami zmieniają się jedynie gradienty
            self.distance_between_ics(unwrapped_model, evaluators, postfix)
            self.distance_between_models(unwrapped_model, self.model_zero, evaluators, postfix,
                                         prefix='distance_from_initialization')  # przekaż model zero przed treningiem
            self.distance_between_models(unwrapped_model, self.model_last_checkpoint, evaluators, postfix,
                                         prefix='distance_from_last_checkpoint')  # zrób ifa jeżeli pierwszy raz to nie
            self.weight_norms(unwrapped_model, evaluators, postfix)

        unwrapped_model._base_model.train()
        # unwrapped_model.zero_grad()  # resetujemy gradienty
        self.model_last_checkpoint = deepcopy(unwrapped_model)
        evaluators[f'steps/grads_stats_ee_{phase}'] = step
        tc.writer.log_scalars(evaluators)
        # for name, value in evaluators.items():
        #     self.logger.add_scalar(name, value, global_step=step)
        # self.logger.log_scalars(evaluators, step)

    def gradient_norm(self, gradients, gradients2, evaluators, postfix):
        grad_norm_layers = defaultdict(list)
        for ic_i, gradient_list in gradients.items():  # po ics   # osobno każdy gradient
            for param_name, grad in gradient_list:  # po parametrach
                grad_norm_per_layer = torch.norm(grad, self.norm_type)
                evaluators[f'gradient_norm_ic_{ic_i}/{param_name}_{postfix}'] = grad_norm_per_layer.item()
                if param_name in self.allowed_parameter_names:
                    grad_norm_layers[ic_i].append(grad_norm_per_layer)

        for ic_i in grad_norm_layers:  # konkatenując wszystkie dla tego samego ic
            grad_norm = torch.norm(torch.stack(grad_norm_layers[ic_i]), self.norm_type)
            evaluators[f'run_stats/gradient_norm_ic_{ic_i}_overall_{postfix}'] = grad_norm.item()

        grad_norm_layers = []
        for param_name, gradient_list in gradients2.items():  # klasycznie
            grad = torch.stack(gradient_list).sum(axis=0)
            grad_norm_per_layer = torch.norm(grad, self.norm_type)
            evaluators[f'gradient_norm_classic/{param_name}_{postfix}'] = grad_norm_per_layer.item()
            if param_name in self.allowed_parameter_names and 'ics' not in param_name:
                grad_norm_layers.append(grad_norm_per_layer)

        grad_norm = torch.norm(torch.stack(grad_norm_layers), self.norm_type)  # całosciowo
        evaluators[f'run_stats/gradient_norm_classic_overall_{postfix}'] = grad_norm.item()

    def gradient_direction_variance(self, gradients, evaluators,
                                    postfix):  # zamiast cos do średniej, rozważ między wszystkimi możliwymi wektorami
        # taken from https://arxiv.org/abs/2303.01500 (GDV, page 4)
        eps = torch.finfo(torch.float32).eps
        for param_name, gradient_list in gradients.items():
            gradient_list = torch.stack(gradient_list)
            gradient_list /= (torch.norm(gradient_list, dim=1, keepdim=True) + eps)
            nb_gradients = gradient_list.shape[0]

            cos = gradient_list @ gradient_list.T  # cosine similarity matrix, number_of_heads x number_of_heads

            coses = torch.triu(cos, diagonal=1)  # wybieram tak by kąty się nie powtarzały
            coses = coses[coses.nonzero(as_tuple=True)].flatten()
            gdv = (1 - coses.mean().item()) / 2

            evaluators[f'gradient_direction_variance/{param_name}_{nb_gradients=}{postfix}'] = gdv

    def gradient_direction_error(self, gradients, evaluators, postfix):
        # taken from https://arxiv.org/abs/2303.01500 (GDE, page 4)
        eps = torch.finfo(torch.float32).eps
        for param_name, gradient_list in gradients.items():
            gradient_list = torch.stack(gradient_list)
            step_gradient = gradient_list.sum(axis=0, keepdim=True)
            step_gradient /= (torch.norm(step_gradient, dim=1, keepdim=True) + eps)
            gradient_list /= (torch.norm(gradient_list, dim=1, keepdim=True) + eps)
            nb_gradients = gradient_list.shape[0]

            cos = gradient_list @ step_gradient.T
            gde = (1 - cos.mean().item()) / 2
            evaluators[f'gradient_direction_error/{param_name}_{nb_gradients=}{postfix}'] = gde

    def gradient_domination(self, gradients, evaluators, postfix):
        # taken from https://arxiv.org/abs/2303.01500 (GDE, page 4)
        eps = torch.finfo(torch.float32).eps
        for param_name, gradient_list in gradients.items():
            gradient_list = torch.stack(gradient_list)
            step_gradient = gradient_list.sum(axis=0, keepdim=True)

            l2_norm = torch.norm(gradient_list - step_gradient, dim=1)
            for ic, l2 in enumerate(l2_norm):
                evaluators[f'gradient_domination_l2_{ic=}/{param_name}{postfix}'] = l2.item()

            step_gradient /= (torch.norm(step_gradient, dim=1, keepdim=True) + eps)
            gradient_list /= (torch.norm(gradient_list, dim=1, keepdim=True) + eps)

            coses = gradient_list @ step_gradient.T
            for ic, cos in enumerate(coses):
                evaluators[f'gradient_domination_cosine_{ic=}/{param_name}{postfix}'] = cos.item()

    def distance_between_ics(self, model, evaluators, postfix):
        named_params = self.get_refined_params(model._heads)
        ics_params = [param.flatten().tolist() for param_name, param in named_params.items() if
                      '_fc' in param_name and 'weight' in param_name]
        # distances = pdist(ics_params, 'euclidean')
        # idx = 0
        for i in range(0, len(ics_params) - 1):
            for j in range(i + 1, len(ics_params)):
                if len(ics_params[i]) == len(ics_params[j]):
                    ic_i_params = np.array(ics_params[i])
                    ic_j_params = np.array(ics_params[j])
                    evaluators[f'distance_between_ics_l2/{i}_and_{j}_{postfix}'] = np.linalg.norm(
                        ic_i_params - ic_j_params, ord=2)
                    evaluators[f'distance_between_ics_cosine/{i}_and_{j}_{postfix}'] = cosine(ic_i_params,
                                                                                              ic_j_params) / 2

    def distance_between_models(self, model1, model2, evaluators, postfix, prefix):
        if model2 is None: return
        overall_distance = 0.0
        named_params1 = self.get_refined_params(model2)
        named_params2 = self.get_refined_params(model1)

        for (name1, param1), (_, param2) in zip(named_params1.items(), named_params2.items()):
            dist = torch.norm(param1 - param2, self.norm_type).item()
            if name1 in self.allowed_parameter_names:
                overall_distance += dist
            evaluators[f'{prefix}/{name1}{postfix}'] = dist
        evaluators[f'run_stats/{prefix}_overall_{postfix}'] = np.sqrt(overall_distance)

    def weight_norms(self, model, evaluators, postfix):
        total_weights_norm = 0.0
        total_biases_norm = 0.0
        for param_name, param in self.get_refined_params(model).items():
            norm = torch.norm(param.data, self.norm_type).item()
            evaluators[f'model_weight_norm/{param_name}{postfix}'] = norm

            if param_name in self.allowed_parameter_names:
                if 'weight' in param_name:
                    total_weights_norm += norm ** 2
                elif 'bias' in param_name:
                    total_biases_norm += norm ** 2

        evaluators['total_weights_norm'] = np.sqrt(total_weights_norm)
        # evaluators['total_biases_norm'] = np.sqrt(total_biases_norm)

    def get_refined_params(self, model, only_params=False):
        named_params = {name: param for name, param in model.named_parameters() if
                        'weight' in name and param.requires_grad}
        if only_params:
            return list(named_params.values())
        else:
            return named_params


import torch
from scipy.stats import zscore
from sklearn.cross_decomposition import CCA


def compute_CKA(classifier_head1, classifier_head2, input_data):
    # Przekształć dane wejściowe do reprezentacji dla każdej głowicy
    representation_head1 = classifier_head1(input_data).detach().numpy()
    representation_head2 = classifier_head2(input_data).detach().numpy()

    # Normalizuj reprezentacje
    representation_head1 = zscore(representation_head1, axis=0)
    representation_head2 = zscore(representation_head2, axis=0)

    # Oblicz CKA
    cca = CCA(n_components=1)
    cca.fit(representation_head1, representation_head2)
    cca_score = cca.score(representation_head1, representation_head2)

    return cca_score


class GradientRescaleFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input)
        # logging.info(f'Setting rescale weight to: {weight}')
        ctx.gd_scale_weight = weight
        output = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            # logging.info(f'Rescaling by: {ctx.gd_scale_weight}')
            grad_input = ctx.gd_scale_weight * grad_output
        return grad_input, grad_weight


from torch.utils.data import Subset, DataLoader


def create_dataloader(dataset, indices, loader_params):
    subset = Subset(dataset, indices)
    return DataLoader(subset, shuffle=False, **loader_params)


import os

import wandb


class WandbLogger:
    def __init__(self, entity, project, run_id, run_name, run_dir, args):
        self.writer = wandb
        if run_id is not None:
            self.writer.init(
                entity=entity,
                project=project,
                id=run_id,
                resume='must',
                dir=str(run_dir.resolve())
            )
        else:
            self.writer.init(
                entity=entity,
                project=project,
                config=dict(args),
                name=run_name,
                dir=str(run_dir.resolve())
            )
        self.writer.run.log_code('.', include_fn=lambda path: path.endswith('.py'))

    def close(self):
        self.writer.finish()

    def log_model(self, model, criterion, log, log_freq: int = 1000, log_graph: bool = True):
        self.writer.watch(model, criterion, log=log, log_freq=log_freq, log_graph=log_graph)

    def log_histograms(self, hists):
        self.writer.log(hists)

    def log_scalars(self, evaluators, step=''):
        self.writer.log(evaluators)

    def log_plots(self, plot_images):
        self.writer.log(plot_images)

class StubLogger:
    def close(self):
        pass

    def log_model(self, model, criterion, log, log_freq: int=1000, log_graph: bool=True):
        pass
    
    def log_histograms(self, hists):
        pass

    def log_scalars(self, evaluators, step=''):
        pass
        
    def log_plots(self, plot_images):
        pass
