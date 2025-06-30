import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePath
from typing import Any, Callable, List, Type

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from omegaconf import OmegaConf

from common import (INIT_NAME_MAP, LOSS_NAME_MAP, OPTIMIZER_NAME_MAP,
                    SCHEDULER_NAME_MAP, get_default_args)
from datasets_config import DATASETS_NAME_MAP
from eval import benchmark, test_classification
from mutual_information import MutualInformation
from tunnel import RepresentationsSpectra
from utils import (Mixup, StubLogger, WandbLogger, count_monotonical_intervals,
                   create_model, generate_run_name, get_loader, get_lrs,
                   get_run_id, gradients_norm, load_state, save_final,
                   save_state, train_val_split, unfrozen_parameters)


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0, metric_type='loss'):
        assert metric_type in {'loss', 'acc'}, f'Validation metric {metric_type} is not supported'
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.extreme_validation_measure = float('inf') if metric_type == 'loss' else - float('inf')
        self.metric_type = metric_type
        self.validation_measures_history = []

    def should_early_stop(self, tc, validation_measure):
        self.validation_measures_history.append(validation_measure)
        if self.metric_type == 'loss':
            if validation_measure < self.extreme_validation_measure:
                self.extreme_validation_measure = validation_measure
                save_state(tc.accelerator, tc.state_path)
                self.counter = 0
            elif validation_measure >= (self.extreme_validation_measure + self.min_delta):
                self.counter += 1
        elif self.metric_type == 'acc':
            if validation_measure > self.extreme_validation_measure:
                self.extreme_validation_measure = validation_measure
                save_state(tc.accelerator, tc.state_path)
                self.counter = 0
            elif validation_measure <= (self.extreme_validation_measure - self.min_delta):
                self.counter += 1
        if tc.accelerator.is_main_process:
            es_scalars = {}
            es_scalars[f'best_early_stopping/{self.metric_type}'] = self.extreme_validation_measure
            es_scalars[f'early_stopping_counter/{self.metric_type}'] = self.counter
            if len(self.validation_measures_history) == 50:
                self.validation_measures_history.pop(0)
                oscilation = max(self.validation_measures_history) - min(self.validation_measures_history)
                n_monotonical_intervals = count_monotonical_intervals(self.validation_measures_history)
                es_scalars[f'early_stopping_oscillations/{self.metric_type}'] = oscilation
                es_scalars[f'early_stopping_n_monotonical_intervals/{self.metric_type}'] = n_monotonical_intervals
            tc.writer.log_scalars(es_scalars)
        if self.counter >= self.patience:
            load_state(tc.accelerator, tc.state_path)
            return True
        return False

    def state_dict(self):
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "counter": self.counter,
            "extreme_validation_measure": self.extreme_validation_measure,
            "metric_type": self.metric_type,
            "validation_measures_history": self.validation_measures_history,
        }

    def load_state_dict(self, state_dict):
        self.patience = state_dict["patience"]
        self.min_delta = state_dict["min_delta"]
        self.counter = state_dict["counter"]
        self.extreme_validation_measure = state_dict["extreme_validation_measure"]
        self.metric_type = state_dict["metric_type"]
        self.validation_measures_history = state_dict["validation_measures_history"]


@dataclass
class TrainingState:
    current_batch: int = 0
    head_index: int = 0

    def state_dict(self):
        return {
            'current_batch': self.current_batch,
            'head_index': self.head_index,
        }

    def load_state_dict(self, state_dict):
        self.current_batch = state_dict['current_batch']
        self.head_index = state_dict['head_index']


@dataclass
class TrainingContext:
    accelerator: Accelerator = None
    model: torch.nn.Module = None
    # savable state
    state: TrainingState = None
    early_stopper: EarlyStopper = None
    # data
    train_loader: torch.utils.data.DataLoader = None
    train_eval_loader: torch.utils.data.DataLoader = None
    test_loader: torch.utils.data.DataLoader = None
    last_batch: int = None
    eval_batch_list: List[int] = None
    # optimization
    criterion_type: Type = None
    criterion: Callable = None
    optimizer: torch.optim.Optimizer = None
    scheduler: Any = None
    # files and logging
    state_path: PurePath = None
    final_path: PurePath = None
    stopfile_path: PurePath = None
    writer: WandbLogger = None


def setup_accelerator(args, tc):
    # do not change the split_batches argument
    # resuming a run with different resources and split_batches=False would cause the batches_per_epoch to be different
    tc.accelerator = Accelerator(split_batches=True, mixed_precision=args.mixed_precision,
                                 # find_unused_parameters finds unused parameters when using DDP
                                 # but may affect performance negatively
                                 # sometimes it happens that somehow it works with this argument, but fails without it
                                 # so - do not remove
                                 # see: https://github.com/pytorch/pytorch/issues/43259
                                 kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
                                 )


def setup_model(args, tc):
    model = create_model(args.model_class, args.model_args)
    init_fun = INIT_NAME_MAP[args.init_fun]
    if init_fun is not None:
        init_fun(model)
    tc.model = tc.accelerator.prepare(model)
    tc.model.train()


def setup_data(args, tc):
    dataset_args = {} if args.dataset_args is None else args.dataset_args
    batch_size = args.batch_size

    unwrapped_model = tc.accelerator.unwrap_model(tc.model)
    with tc.accelerator.main_process_first():
        train_data, train_eval_data, test_data = DATASETS_NAME_MAP[args.dataset](**dataset_args)
    if args.early_stopping:
        train_data, eval_data = train_val_split(train_data, test_size=args.train_val_split_test_size,
                                                seed=args.train_val_split_seed, stratify_by_labels=unwrapped_model.number_of_classes != 1)
        if hasattr(test_data, 'transform'):
            eval_data.transform = test_data.transform
        tc.eval_loader = get_loader(eval_data, batch_size, tc.accelerator)

    tc.train_loader = get_loader(train_data, batch_size, tc.accelerator)
    tc.train_eval_loader = get_loader(train_eval_data, batch_size, tc.accelerator)
    tc.test_loader = get_loader(test_data, batch_size, tc.accelerator)
    batches_per_epoch = len(tc.train_loader)
    tc.last_batch = args.epochs * batches_per_epoch - 1
    tc.eval_batch_list = [
        round(x) for x in torch.linspace(0, tc.last_batch, steps=args.eval_points, device='cpu').tolist()
    ]
    if tc.accelerator.is_main_process:
        logging.info(f'{args.epochs=} {batches_per_epoch=} {tc.last_batch=}')


def setup_optimization(args, tc):
    criterion_args = args.loss_args
    tc.criterion_type = LOSS_NAME_MAP[args.loss_type]
    tc.criterion = tc.criterion_type(reduction='mean', **criterion_args)
    optimizer_args = args.optimizer_args
    tc.optimizer = tc.accelerator.prepare(
        OPTIMIZER_NAME_MAP[args.optimizer_class](unfrozen_parameters(tc.model), **optimizer_args))
    if args.scheduler_class is not None:
        scheduler_args = deepcopy(args.scheduler_args)
        if 'patience' in scheduler_args:
            scheduler_args['patience'] = int(scheduler_args['patience'] * tc.last_batch)
        if args.scheduler_class == 'cosine':
            scheduler_args['T_max'] = tc.last_batch if 'T_max' not in args.scheduler_args else args.scheduler_args[
                'T_max']
        tc.scheduler = tc.accelerator.prepare(SCHEDULER_NAME_MAP[args.scheduler_class](tc.optimizer, **scheduler_args))


def setup_files_and_logging(args, tc):
    # files setup
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    _, run_name = generate_run_name(args)
    run_dir = args.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tc.state_path = run_dir / f'state'
    tc.final_path = run_dir / f'final.pth'
    tc.stopfile_path = run_dir / f'stop'
    # logging setup
    if tc.accelerator.is_main_process:
        # log config
        logging.info(f'{run_name} args:\n{args}')
        if args.use_wandb:
            entity = os.environ['WANDB_ENTITY']
            project = os.environ['WANDB_PROJECT']
            run_id = get_run_id(run_name)
            tc.writer = WandbLogger(entity, project, run_id, run_name, run_dir, args)
    else:
        if args.use_wandb:
            tc.writer = StubLogger()


def setup_state(tc):
    tc.state = TrainingState()
    tc.accelerator.register_for_checkpointing(tc.state)


def setup_tunnel(args, tc):
    from utils import create_dataloader
    dataset_args = {} if args.dataset_args is None else args.dataset_args
    train_loader_rank = deepcopy(tc.train_loader)
    eval_loader_rank = deepcopy(tc.eval_loader)
    indices = torch.load(f'data/train_{args.dataset}_indices.pt').tolist()
    train_loader_rank = create_dataloader(train_loader_rank.dataset.dataset, indices, dataset_args)
    train_loader_rank = tc.accelerator.prepare(train_loader_rank)
    # eval_loader_rank = tc.accelerator.prepare(eval_loader_rank)
    tc.rank_train = RepresentationsSpectra(tc.model, loader=train_loader_rank,
                                           modules_list=[torch.nn.Conv2d, torch.nn.Linear],
                                           MAX_REPR_SIZE=args.MAX_REPR_SIZE)
    tc.rank_test = RepresentationsSpectra(tc.model, loader=eval_loader_rank,
                                          modules_list=[torch.nn.Conv2d, torch.nn.Linear],
                                          MAX_REPR_SIZE=args.MAX_REPR_SIZE)


def setup_mi(args, tc):
    from utils import create_dataloader
    dataset_args = {} if args.dataset_args is None else args.dataset_args
    train_loader_rank = deepcopy(tc.train_loader)
    eval_loader_rank = deepcopy(tc.eval_loader)
    indices = torch.load(f'data/train_{args.dataset}_indices.pt').tolist()
    train_loader_rank = create_dataloader(train_loader_rank.dataset.dataset, indices, dataset_args)
    train_loader_rank = tc.accelerator.prepare(train_loader_rank)
    # eval_loader_rank = tc.accelerator.prepare(eval_loader_rank)
    tc.mi_train = MutualInformation(tc.model, loader=train_loader_rank, modules_list=[torch.nn.Conv2d, torch.nn.Linear],
                                    MAX_REPR_SIZE=args.MAX_REPR_SIZE)
    tc.mi_test = MutualInformation(tc.model, loader=eval_loader_rank, modules_list=[torch.nn.Conv2d, torch.nn.Linear],
                                   MAX_REPR_SIZE=args.MAX_REPR_SIZE)


def setup_early_stopper(args, tc):
    if args.early_stopping:
        tc.early_stopper = EarlyStopper(patience=args.early_stopping_patience,
                                        min_delta=args.early_stopping_delta,
                                        metric_type=args.early_stopping_metric_type)
        tc.accelerator.register_for_checkpointing(tc.early_stopper)
    else:
        tc.early_stopper = None


def in_training_eval(args, tc):
    # if tc.state.current_batch in tc.eval_batch_list:
    eval_scalars = {}
    test_loss, test_acc = test_classification(tc.accelerator,
                                              tc.model,
                                              tc.test_loader,
                                              tc.criterion_type,
                                              batches=0)
    train_loss, train_acc = test_classification(tc.accelerator,
                                                tc.model,
                                                tc.train_eval_loader,
                                                tc.criterion_type,
                                                batches=args.eval_batches)
    if tc.accelerator.is_main_process:
        eval_scalars['Train/Progress'] = tc.state.current_batch / tc.last_batch
        eval_scalars['Train/Progress_Epochs'] = tc.state.current_batch / len(tc.train_loader)
        eval_scalars['Eval/Test loss'] = test_loss
        eval_scalars['Eval/Test accuracy'] = test_acc
        eval_scalars['Eval/Train loss'] = train_loss
        eval_scalars['Eval/Train accuracy'] = train_acc
        tc.writer.log_scalars(eval_scalars, tc.state.current_batch)


def training_loop(args, tc):
    # TODO reduce code duplication - see code in `methods`
    if tc.accelerator.is_main_process:
        model_saved = datetime.now()
    train_iter = iter(tc.train_loader)
    unwrapped_model = tc.accelerator.unwrap_model(tc.model)
    if args.mixup_alpha is not None or args.cutmix_alpha is not None:
        mixup_mode = 'batch' if args.mixup_mode is None else args.mixup_mode
        mixup_smoothing = 0.1 if args.mixup_smoothing is None else args.mixup_smoothing
        mixup_fn = Mixup(
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, mode=mixup_mode,
            label_smoothing=mixup_smoothing, num_classes=unwrapped_model.number_of_classes)
    else:
        mixup_fn = None
    batches_per_epoch = len(tc.train_loader)
    # "temporary" (i.e. will never rewrite this) workaround
    signalled_stop_condition = tc.stopfile_path.exists()
    if signalled_stop_condition and args.scheduler_class == 'cosine_warm_restarts':
        lr = get_lrs(tc.optimizer)[0]
        if lr != args.optimizer_args.lr:
            signalled_stop_condition = False
    while tc.state.current_batch <= tc.last_batch and not tc.final_path.exists() and not signalled_stop_condition:
        train_scalars = {}
        # save model conditionally
        if tc.accelerator.is_main_process and tc.early_stopper is None:
            now = datetime.now()
            if (now - model_saved).total_seconds() > 60 * args.save_every:
                save_state(tc.accelerator, tc.state_path)
                model_saved = datetime.now()
        # model evaluation
        if tc.state.current_batch % batches_per_epoch == 0:
            # temporary workaround TODO
            if tc.accelerator.is_main_process:
                gradients_norm(tc.writer, tc.model, tc.state.current_batch)
            in_training_eval(args, tc)
        # batch preparation
        try:
            X, y = next(train_iter)
        except StopIteration:
            train_iter = iter(tc.train_loader)
            X, y = next(train_iter)
        if mixup_fn is not None:
            X, y = mixup_fn(X, y)
        # forward
        tc.model.train()
        y_pred = tc.model(X)
        # loss computation
        loss = tc.criterion(y_pred, y)
        # gradient computation and training step
        tc.optimizer.zero_grad(set_to_none=True)
        tc.accelerator.backward(loss)
        if args.clip_grad_norm is not None:
            total_norm = tc.accelerator.clip_grad_norm_(tc.model.parameters(), args.clip_grad_norm)
            if tc.accelerator.is_main_process:
                train_scalars['Train/Gradient norm'] = total_norm.item()
                # tc.writer.add_scalar(f'Train/Gradient norm', total_norm.item(), global_step=tc.state.current_batch)
        tc.optimizer.step()
        if tc.scheduler is not None:
            # log LRs
            if tc.accelerator.is_main_process:
                for i, lr in enumerate(get_lrs(tc.optimizer)):
                    train_scalars[f'Train/Group {i} LR'] = lr
                    # tc.writer.add_scalar(f'Train/Group {i} LR', lr, global_step=tc.state.current_batch)
            if args.scheduler_class == 'reduce_on_plateau':
                tc.scheduler.step(loss)
            else:
                tc.scheduler.step()
        if tc.accelerator.is_main_process:
            # acc = (torch.argmax(y_pred, axis=-1)[1] == y).float().mean()
            # tc.writer.add_scalar(f'Train/Acc____per_batch', acc, global_step=tc.state.current_batch)
            train_scalars['Train/Loss____per_batch'] = loss.item()
            # tc.writer.add_scalar(f'Train/Loss____per_batch', loss.item(), global_step=tc.state.current_batch)
            if args.is_rank_train and tc.state.current_batch % (args.rank_multiplier * batches_per_epoch) == 0:
                tc.rank_train.enable()
                tc.rank_train.analysis(tc.state.current_batch, scope='periodic', phase='train', tc=tc)
                tc.rank_train.disable()
            if args.is_rank_test and tc.state.current_batch % (args.rank_multiplier * batches_per_epoch) == 0:
                tc.rank_test.enable()
                tc.rank_test.analysis(tc.state.current_batch, scope='periodic', phase='test', tc=tc)
                tc.rank_test.disable()
            if args.is_mi_train and tc.state.current_batch % (args.mi_multiplier * batches_per_epoch) == 0:
                tc.mi_train.enable()
                tc.mi_train.analysis(tc.state.current_batch, scope='periodic', phase='train', tc=tc)
                tc.mi_train.disable()
            if args.is_mi_test and tc.state.current_batch % (args.mi_multiplier * batches_per_epoch) == 0:
                tc.mi_test.enable()
                tc.mi_test.analysis(tc.state.current_batch, scope='periodic', phase='test', tc=tc)
                tc.mi_test.disable()
            tc.writer.log_scalars(train_scalars)
        # bookkeeping
        tc.state.current_batch += 1
        if tc.early_stopper is not None and tc.state.current_batch % batches_per_epoch == 0:
            early_scalars = {}
            # one epoch finished
            train_loss, train_acc = test_classification(tc.accelerator,
                                                        tc.model,
                                                        tc.eval_loader,
                                                        tc.criterion_type,
                                                        batches=0)
            if tc.accelerator.is_main_process:
                early_scalars['Early stopping/eval loss'] = train_loss
                early_scalars['Early stopping/eval acc'] = train_acc
                tc.writer.log_scalars(early_scalars)

            if tc.early_stopper.should_early_stop(tc,
                                                  validation_measure=train_loss if args.early_stopping_metric_type == 'loss' else train_acc):
                logging.info(
                    f'Early stopping after {tc.state.current_batch // batches_per_epoch} epochs: {train_loss=}, {train_acc=}')
                break


def final_eval(args, tc):
    if not tc.final_path.exists():
        if tc.accelerator.is_main_process:
            save_state(tc.accelerator, tc.state_path)
        # test on testsets
        test_loss, test_acc = test_classification(tc.accelerator, tc.model, tc.test_loader, tc.criterion_type)
        if tc.accelerator.is_main_process:
            final_results_scalars = {}
            final_results = {}
            final_results['args'] = args
            unwrapped_model = tc.accelerator.unwrap_model(tc.model)
            final_results['model_state'] = unwrapped_model.state_dict()
            final_results_scalars['Eval/Test loss'] = test_loss
            final_results_scalars['Eval/Test accuracy'] = test_acc
            final_results['final_score'] = test_acc
            final_results['final_loss'] = test_loss
            logging.info(f'Test accuracy: {test_acc}')
            # benchmark model efficiency
            model_costs, model_params = benchmark(unwrapped_model, tc.test_loader)
            final_results['model_flops'] = model_costs.total()
            final_results['model_flops_by_module'] = dict(model_costs.by_module())
            final_results['model_flops_by_operator'] = dict(model_costs.by_operator())
            final_results['model_params'] = dict(model_params)
            final_results_scalars['Eval/Model FLOPs'] = model_costs.total()
            final_results_scalars['Eval/Model Params'] = model_params['']
            save_final(args, tc.final_path, final_results)
            tc.writer.log_scalars(final_results_scalars)


def train(args):
    logging.basicConfig(
        format=(
            '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'
        ),
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    logging.info('Configured logging')
    tc = TrainingContext()
    setup_accelerator(args, tc)
    setup_files_and_logging(args, tc)
    setup_model(args, tc)
    setup_data(args, tc)
    setup_optimization(args, tc)
    setup_state(tc)
    # setup_tunnel(args, tc)
    if args.is_mi_train or args.is_mi_test:
        setup_mi(args, tc)
    setup_early_stopper(args, tc)
    load_state(tc.accelerator, tc.state_path)
    training_loop(args, tc)
    final_eval(args, tc)


def main():
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    train(args)


if __name__ == '__main__':
    main()
