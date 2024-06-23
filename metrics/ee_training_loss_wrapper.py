import loss_landscapes as ll
from loss_landscapes import metrics
import torch

from methods.early_exit import AUX_LOSS_MAP
from utils import Mixup


class EETrainingLossWrapper(metrics.Metric):
    def __init__(self, data_loader, args, accelerator, criterion, gen_device):
        super().__init__()
        self.data_loader = data_loader
        self.args = args
        self.accelerator = accelerator
        self.criterion = criterion
        self.gen_device = gen_device

    def __call__(self, model_wrapper: ll.ModelWrapper) -> float:
        total_loss = 0.0
        with torch.inference_mode():
            with self.gen_device:
                for X, y in self.data_loader:
                    total_loss += training_loss(
                        model_wrapper.modules[0],
                        X,
                        y,
                        self.args,
                        self.accelerator,
                        self.criterion,
                    ).item()
        return total_loss


def training_loss(model, X, y, args, accelerator, criterion):
    X, y = mixup(X, y, args=args, accelerator=accelerator, model=model)
    output = model(X)

    losses = [criterion(y_pred, y) for y_pred in output]
    if "gp" in args.model_class:
        denominator = len(losses) * (len(losses) + 1) / 2
        loss = sum((i + 1) * loss_ for i, loss_ in enumerate(losses)) / denominator
        gpf_aux_loss = model.aux_loss()
        loss = loss + gpf_aux_loss
    else:
        denominator = len(losses)
        loss = sum(l for l in losses) / denominator
    if args.auxiliary_loss_type is not None:
        aux_loss = AUX_LOSS_MAP[args.auxiliary_loss_type](output)
        loss = loss + args.auxiliary_loss_weight * aux_loss
    return loss


class EEHeadLossWrapper(metrics.Metric):
    def __init__(self, data_loader, args, accelerator, criterion, head_idx, gen_device):
        super().__init__()
        self.data_loader = data_loader
        self.args = args
        self.accelerator = accelerator
        self.criterion = criterion
        self.head_idx = head_idx
        self.gen_device = gen_device

    def __call__(self, model_wrapper: ll.ModelWrapper) -> float:
        total_loss = 0.0
        with torch.inference_mode():
            with self.gen_device:
                for X, y in self.data_loader:
                    total_loss += head_loss(
                        model_wrapper.modules[0],
                        X,
                        y,
                        self.args,
                        self.accelerator,
                        self.criterion,
                        self.head_idx,
                    ).item()
        return total_loss


def head_loss(model, X, y, args, accelerator, criterion, head_idx):
    X, y = mixup(X, y, args=args, accelerator=accelerator, model=model)
    output = model(X)

    y_pred = output[head_idx]
    loss = criterion(y_pred, y)
    return loss


class EEHeadAccuracyWrapper(metrics.Metric):
    def __init__(self, data_loader, args, accelerator, head_idx, gen_device):
        super().__init__()
        self.data_loader = data_loader
        self.args = args
        self.accelerator = accelerator
        self.head_idx = head_idx
        self.gen_device = gen_device

    def __call__(self, model_wrapper: ll.ModelWrapper) -> float:
        total_correct = 0
        total_count = 0
        with torch.inference_mode():
            with self.gen_device:
                for X, y in self.data_loader:
                    correct, count = head_correct(
                        model_wrapper.modules[0],
                        X,
                        y,
                        self.args,
                        self.accelerator,
                        self.head_idx,
                    )
                    total_correct += correct
                    total_count += count
        return total_correct / total_count


def head_correct(model, X, y, args, accelerator, head_idx):
    X, y = mixup(X, y, args=args, accelerator=accelerator, model=model)
    output = model(X)

    y_pred = output[head_idx]

    y_pred_max = y_pred.argmax(dim=1)
    accuracy = (y_pred_max == y).sum().item()
    return accuracy, y.size(0)


def mixup(X, y, args, accelerator, model):
    if args.mixup_alpha is not None or args.cutmix_alpha is not None:
        mixup_mode = "batch" if args.mixup_mode is None else args.mixup_mode
        mixup_smoothing = 0.1 if args.mixup_smoothing is None else args.mixup_smoothing
        mixup_fn = Mixup(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mode=mixup_mode,
            label_smoothing=mixup_smoothing,
            num_classes=accelerator.unwrap_model(model).number_of_classes,
        )
    else:
        mixup_fn = None

    if mixup_fn is not None:
        X, y = mixup_fn(X, y)
    return X, y
