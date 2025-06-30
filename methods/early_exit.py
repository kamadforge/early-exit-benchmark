import logging
from collections import defaultdict
from copy import deepcopy
from datetime import datetime

import torch
from omegaconf import OmegaConf

from architectures.resnets import ResNet
from architectures.vit import VisionTransformer
from common import (INIT_NAME_MAP, OPTIMIZER_NAME_MAP, SCHEDULER_NAME_MAP,
                    get_default_args)
from eval import (benchmark_earlyexiting, evaluate_earlyexiting_classification,
                  get_preds_earlyexiting, test_earlyexiting_classification)
from mutual_information import MutualInformation
from train import (TrainingContext, setup_accelerator, setup_data,
                   setup_files_and_logging, setup_optimization, setup_state)
from tunnel import RepresentationsSpectra
from utils import (Mixup, count_monotonical_intervals, create_model, get_lrs,
                   gradients_norm, load_model, load_state, save_final,
                   save_state, unfrozen_parameters)


class EarlyStopperEarlyExit:
    def __init__(self, losses_num=1, patience=5, min_delta=0, metric_type='loss'):
        assert metric_type in {'loss', 'acc'}, f'Validation metric {metric_type} is not supported'
        self.patience = patience
        self.min_delta = min_delta

        self.losses_num = losses_num
        self.counter = [0] * losses_num
        self.extreme_validation_measures = [float('inf') if metric_type == 'loss' else - float('inf')] * losses_num
        self.metric_type = metric_type
        self.validation_measures_histories = defaultdict(list)

    def should_early_stop(self, tc, validation_measures):
        better_model_found = False
        for idx, validation_measure in enumerate(validation_measures):
            self.validation_measures_histories[idx].append(validation_measure)
            if self.metric_type == 'loss':
                if validation_measure < self.extreme_validation_measures[idx]:
                    self.extreme_validation_measures[idx] = validation_measure
                    better_model_found = True
                    self.counter[idx] = 0
                elif validation_measure >= (self.extreme_validation_measures[idx] + self.min_delta):
                    self.counter[idx] += 1
            elif self.metric_type == 'acc':
                if validation_measure > self.extreme_validation_measures[idx]:
                    self.extreme_validation_measures[idx] = validation_measure
                    better_model_found = True
                    self.counter[idx] = 0
                elif validation_measure <= (self.extreme_validation_measures[idx] - self.min_delta):
                    self.counter[idx] += 1

        es_scalars = {}
        for i, best_value in enumerate(self.extreme_validation_measures):
            es_scalars[f'best_early_stopping/{self.metric_type}_{i}'] = best_value
            es_scalars[f'early_stopping_counter/{self.metric_type}_{i}'] = self.counter[i]
            if len(self.validation_measures_histories[i]) == 50:
                oscilation = max(self.validation_measures_histories[i]) - min(self.validation_measures_histories[i])
                n_monotonical_intervals = count_monotonical_intervals(self.validation_measures_histories[i])
                es_scalars[f'early_stopping_oscillations/{self.metric_type}_{i}_window={50}'] = oscilation
                es_scalars[
                    f'early_stopping_n_monotonical_intervals/{self.metric_type}_{i}_window={50}'] = n_monotonical_intervals
                oscilation = max(self.validation_measures_histories[i][-20:]) - min(
                    self.validation_measures_histories[i][-20:])
                n_monotonical_intervals = count_monotonical_intervals(self.validation_measures_histories[i][-20:])
                es_scalars[f'early_stopping_oscillations/{self.metric_type}_{i}_window={20}'] = oscilation
                es_scalars[
                    f'early_stopping_n_monotonical_intervals/{self.metric_type}_{i}_window={20}'] = n_monotonical_intervals
                self.validation_measures_histories[i].pop(0)

        if tc.accelerator.is_main_process:
            tc.writer.log_scalars(es_scalars)
        if better_model_found:
            save_state(tc.accelerator, tc.state_path)
            return False
        all_converged = sum([count >= self.patience for count in self.counter]) == self.losses_num
        if all_converged:
            load_state(tc.accelerator, tc.state_path)
            return True
        return False
    
    def state_dict(self):
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "losses_num": self.losses_num,
            "counter": self.counter,
            "extreme_validation_measures": self.extreme_validation_measures,
            "metric_type": self.metric_type,
            "validation_measures_histories": self.validation_measures_histories,
        }

    def load_state_dict(self, state_dict):
        self.patience = state_dict["patience"]
        self.min_delta = state_dict["min_delta"]
        self.losses_num = state_dict["losses_num"]
        self.counter = state_dict["counter"]
        self.extreme_validation_measures = state_dict["extreme_validation_measures"]
        self.metric_type = state_dict["metric_type"]
        self.validation_measures_histories = state_dict["validation_measures_histories"]


def distill_last(output):
    distillation_loss = 0.0
    num_heads = len(output) - 1
    for i in range(num_heads):
        dis_loss = torch.nn.functional.cross_entropy(output[i], torch.softmax(output[num_heads].detach(), dim=-1))
        distillation_loss += dis_loss
    distillation_loss /= num_heads
    return distillation_loss


def distill_later(output):
    # distill each head to all deeper heads
    distillation_loss = 0.0
    num_outputs = len(output)
    addition_counter = 0
    for i in range(num_outputs):
        for j in range(i + 1, num_outputs):
            dis_loss = torch.nn.functional.cross_entropy(output[i], torch.softmax(output[j].detach(), dim=-1))
            distillation_loss += dis_loss
            addition_counter += 1
    distillation_loss /= addition_counter
    return distillation_loss


def distill_next(output):
    # distill each head to the next head
    distillation_loss = 0.0
    num_outputs = len(output)
    for i in range(num_outputs - 1):
        dis_loss = torch.nn.functional.cross_entropy(output[i], torch.softmax(output[i + 1].detach(), dim=-1))
        distillation_loss += dis_loss
    distillation_loss /= num_outputs - 1
    return distillation_loss


AUX_LOSS_MAP = {
    'distill_last': distill_last,
    'distill_later': distill_later,
    'distill_next': distill_next,
}


def set_for_training(args, tc):
    model = tc.accelerator.unwrap_model(tc.model)
    if args.with_backbone:
        model.train('all')
    else:
        model.train('without_backbone')
    model.all_mode()


def setup_model(args, tc):
    base_model, base_args, _ = load_model(args, args.base_on, args.exp_id)
    model_args = {'base_model': base_model, **args.model_args}
    model = create_model(args.model_class, model_args)
    init_fun = INIT_NAME_MAP[args.init_fun if args.init_fun is not None else base_args.init_fun]
    if init_fun is not None:
        for head_module in model.head_modules():
            init_fun(head_module)
    tc.model = tc.accelerator.prepare(model)
    set_for_training(args, tc)


def setup_tunnel(args, tc):
    from utils import create_dataloader
    dataset_args = {} if args.dataset_args is None else args.dataset_args
    train_loader_rank = deepcopy(tc.train_loader)
    eval_loader_rank = deepcopy(tc.eval_loader)
    indices = torch.load(f'data/train_{args.dataset}_indices.pt').tolist()
    train_loader_rank = create_dataloader(train_loader_rank.dataset.dataset, indices, dataset_args)
    train_loader_rank = tc.accelerator.prepare(train_loader_rank)
    # eval_loader_rank = tc.accelerator.prepare(eval_loader_rank)
    unwrapped_model = tc.accelerator.unwrap_model(tc.model)
    base_model = unwrapped_model._base_model
    tc.rank_train = RepresentationsSpectra(base_model, loader=train_loader_rank,
                                           modules_list=[torch.nn.Conv2d, torch.nn.Linear],
                                           MAX_REPR_SIZE=args.MAX_REPR_SIZE)
    tc.rank_test = RepresentationsSpectra(base_model, loader=eval_loader_rank,
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
    base_model = tc.accelerator.unwrap_model(tc.model)
    tc.mi_train = MutualInformation(base_model, loader=train_loader_rank,
                                    modules_list=[torch.nn.Conv2d, torch.nn.Linear], MAX_REPR_SIZE=args.MAX_REPR_SIZE)
    tc.mi_test = MutualInformation(base_model, loader=eval_loader_rank, modules_list=[torch.nn.Conv2d, torch.nn.Linear],
                                   MAX_REPR_SIZE=args.MAX_REPR_SIZE)


def reset_optimizer(args, tc):
    set_for_training(args, tc)
    optimizer_args = args.optimizer_args
    opt = tc.accelerator.prepare(OPTIMIZER_NAME_MAP[args.optimizer_class](unfrozen_parameters(tc.model), **optimizer_args))
    tc.optimizer.load_state_dict(opt.state_dict())

def reset_scheduler(args, tc):
    if args.scheduler_class == None:
        return
    scheduler_args = deepcopy(args.scheduler_args)
    # if 'T_0' in scheduler_args:
    #     scheduler_args['T_0'] = int(math.ceil(scheduler_args['T_0'] * tc.last_batch))
    if 'patience' in scheduler_args:
        scheduler_args['patience'] = int(scheduler_args['patience'] * tc.last_batch)
    if args.scheduler_class == 'cosine':
        scheduler_args['T_max'] = tc.last_batch if 'T_max' not in args.scheduler_args else args.scheduler_args['T_max']
    sched = tc.accelerator.prepare(SCHEDULER_NAME_MAP[args.scheduler_class](tc.optimizer, **scheduler_args))
    tc.scheduler.load_state_dict(sched.state_dict())


def get_early_stopper(args, tc):
    losses_num = tc.accelerator.unwrap_model(tc.model).number_of_heads
    early_stopper = EarlyStopperEarlyExit(losses_num=losses_num,
                                          patience=args.early_stopping_patience,
                                          min_delta=args.early_stopping_delta,
                                          metric_type=args.early_stopping_metric_type,
    )
    return early_stopper

def reset_early_stopper(args, tc):
    tc.early_stopper.load_state_dict(get_early_stopper(args, tc).state_dict())

def setup_early_stopper(args, tc):
    # gdy uczymy jedynie heady to powinno być number_of_heads - 1, chyba że główny head również decydujemy się wytrenować
    if args.early_stopping:
        tc.early_stopper = get_early_stopper(args, tc)
        tc.accelerator.register_for_checkpointing(tc.early_stopper)
    else:
        tc.early_stopper = None


def setup_gradient_statistician_ee(args, tc):
    pass
    # if args.gather_stats:
    #     logging.info('Entering Gradient Statistician setup.')
    #     held_out = {}
    #     held_out['x_train'] = torch.load(f'data/train_{args.dataset}_held_out_x.pt')
    #     held_out['y_train'] = torch.load(f'data/train_{args.dataset}_held_out_y.pt')
    #     held_out['x_test'] = torch.load(f'data/test_{args.dataset}_held_out_x.pt')
    #     held_out['y_test'] = torch.load(f'data/test_{args.dataset}_held_out_y.pt')
    #     # held_out = {name: tc.accelerator.prepare(tensor) for name, tensor in held_out.items()}
    #     tc.grad_statistician_ee = GradientStatisticianEE(held_out)
    #     tc.grad_statistician_ee.model_zero = deepcopy(tc.accelerator.unwrap_model(tc.model))
    #     tc.grad_statistician_ee.logger = tc.writer


def setup_gradient_equalibrum(args, tc):
    if args.gradient_equilibrium:  # GRADIENT EQUALIBRUM
        from itertools import chain

        from torch.nn import functional as F

        from utils import GradientRescaleFunction
        def set_attributes(self, gradient_rescale, head_place_at):
            self.gradient_rescale = gradient_rescale
            self.head_place_at = head_place_at
            self.number_of_attached_heads = len(head_place_at)

        # 2019
        # def ``forward_ge``nerator(self, x):
        #     x = torch.relu(self.bn1(self.conv1(x)))
        #     head_idx = 0
        #     for i, block in enumerate(chain(self.layer1, self.layer2, self.layer3, self.layer4)):
        #         x = block(x)
        #         if i in self.head_place_at:
        #             x = self.gradient_rescale(x, 1. / (self.number_of_attached_heads - head_idx + 1))
        #         x = yield x, None
        #         if i in self.head_place_at:
        #             x = self.gradient_rescale(x, (self.number_of_attached_heads - head_idx))
        #             head_idx += 1
        #     x = F.adaptive_avg_pool2d(x, (1, 1))
        #     x = x.view(x.size(0), -1)
        #     y = self.linear(x)
        #     _ = yield None, y
        # 2022
        # def forward_generator_resnet(self, x):
        #     x = torch.relu(self.bn1(self.conv1(x)))
        #     head_idx = 0
        #     for i, block in enumerate(chain(self.layer1, self.layer2, self.layer3, self.layer4)):
        #         x = block(x)
        #         x = yield x, None
        #         if i in self.head_place_at:
        #             x = self.gradient_rescale(x, (self.number_of_attached_heads - head_idx) / (
        #                     self.number_of_attached_heads - head_idx + 1))
        #             head_idx += 1
        #     x = F.adaptive_avg_pool2d(x, (1, 1))
        #     x = x.view(x.size(0), -1)
        #     y = self.linear(x)
        #     _ = yield None, y

        def forward_generator_resnet(self, x):
            # we actually need to account for the original head
            n_heads = self.number_of_attached_heads + 1
            x = torch.relu(self.bn1(self.conv1(x)))
            head_idx = 0
            for i, block in enumerate(chain(self.layer1, self.layer2, self.layer3, self.layer4)):
                x = block(x)
                if i in self.head_place_at:
                    if args.gradient_equilibrium is True:
                        x = self.gradient_rescale(x, 1 / (n_heads - head_idx + 1))
                    elif args.gradient_equilibrium == 'reversed':
                        x = self.gradient_rescale(x, n_heads - head_idx)
                x = yield x, None
                if i in self.head_place_at:
                    if args.gradient_equilibrium is True:
                        x = self.gradient_rescale(x, n_heads - head_idx)
                    elif args.gradient_equilibrium == 'reversed':
                        x = self.gradient_rescale(x, 1 / (n_heads - head_idx + 1))
                    head_idx += 1
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
            y = self.linear(x)
            yield None, y


        def forward_generator_vit(self, x):
            n_heads = self.number_of_attached_heads + 1
            # Reshape and permute the input tensor
            x = self._process_input(x)
            n = x.shape[0]
            # Expand the class token to the full batch
            batch_class_token = self.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            # BEGIN ENCODER
            # equivalent to: x = self.encoder(x)
            x = x + self.encoder.pos_embedding
            x = self.encoder.dropout(x)
            # go through encoder blocks
            head_idx = 0
            for i, block in enumerate(self.encoder.layers):
                x = block(x)
                if i in self.head_place_at:
                    if args.gradient_equilibrium is True:
                        x = self.gradient_rescale(x, 1 / (n_heads - head_idx + 1))
                    elif args.gradient_equilibrium == 'reversed':
                        x = self.gradient_rescale(x, n_heads - head_idx)
                x = yield x, None
                if i in self.head_place_at:
                    if args.gradient_equilibrium is True:
                        x = self.gradient_rescale(x, n_heads - head_idx)
                    elif args.gradient_equilibrium == 'reversed':
                        x = self.gradient_rescale(x, 1 / (n_heads - head_idx + 1))
                    head_idx += 1
            x = self.encoder.ln(x)
            # END OF ENCODER
            # classifier token
            x = x[:, 0]
            x = self.heads(x)
            _ = yield None, x

        unwrapped_model = tc.accelerator.unwrap_model(tc.model)
        base_model = unwrapped_model._base_model
        base_model.set_attributes = set_attributes.__get__(base_model)
        if isinstance(base_model, ResNet):
            base_model.forward_generator = forward_generator_resnet.__get__(base_model)
        elif isinstance(base_model, VisionTransformer):
            base_model.forward_generator = forward_generator_vit.__get__(base_model)
        else:
            raise NotImplementedError('GE not supported for this model')
        base_model.set_attributes(GradientRescaleFunction.apply, unwrapped_model._place_at)


def in_training_eval(args, tc):
    # if tc.state.current_batch in tc.eval_batch_list:
    eval_scalars = {}
    test_losses, test_accs = test_earlyexiting_classification(tc.accelerator,
                                                              tc.model,
                                                              tc.test_loader,
                                                              tc.criterion_type,
                                                              batches=0)
    train_losses, train_accs = test_earlyexiting_classification(tc.accelerator,
                                                                tc.model,
                                                                tc.train_eval_loader,
                                                                tc.criterion_type,
                                                                batches=args.eval_batches)
    if tc.accelerator.is_main_process:
        eval_scalars['Train/Progress'] = tc.state.current_batch / tc.last_batch
        for i in range(tc.accelerator.unwrap_model(tc.model).number_of_heads):
            eval_scalars[f'Eval/Head {i} test loss'] = test_losses[i]
            eval_scalars[f'Eval/Head {i} test accuracy'] = test_accs[i]
            eval_scalars[f'Eval/Head {i} train loss'] = train_losses[
                i]  # train dataset without augmentation or eval dataset
            eval_scalars[f'Eval/Head {i} train accuracy'] = train_accs[
                i]  # train dataset without augmentation or eval dataset

        tc.writer.log_scalars(eval_scalars, tc.state.current_batch)


def training_loop(args, tc):
    logging.info('Entering the second phase')
    tc.phase2_end = args.epochs_common * len(tc.train_loader)  # at which step second phase ends
    is_phase2 = True
    if tc.accelerator.is_main_process:
        model_saved = datetime.now()
    train_iter = iter(tc.train_loader)
    if args.mixup_alpha is not None or args.cutmix_alpha is not None:
        mixup_mode = 'batch' if args.mixup_mode is None else args.mixup_mode
        mixup_smoothing = 0.1 if args.mixup_smoothing is None else args.mixup_smoothing
        mixup_fn = Mixup(
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, mode=mixup_mode,
            label_smoothing=mixup_smoothing, num_classes=tc.accelerator.unwrap_model(tc.model).number_of_classes)
    else:
        mixup_fn = None
    if args.aggregate_loss == 'sdn':
        unwrapped_model = tc.accelerator.unwrap_model(tc.model)
        head_costs, model_params = benchmark_earlyexiting(unwrapped_model, tc.test_loader)
        # https://github.com/yigitcankaya/Shallow-Deep-Networks/blob/a55644624c9dc064d1372f6a486055edd94c9def/model_funcs.py#L86-L87
        max_coeffs = [cost.total() / head_costs[-1].total() for cost in head_costs]
    batches_per_epoch = len(tc.train_loader)
    signalled_stop_condition = tc.stopfile_path.exists()
    if signalled_stop_condition and args.scheduler_class == 'cosine_warm_restarts':
        lr = get_lrs(tc.optimizer)[0]
        if lr != args.optimizer_args.lr:
            signalled_stop_condition = False
    while tc.state.current_batch <= tc.last_batch and not tc.final_path.exists() and not signalled_stop_condition:
        train_scalars = {}
        if tc.state.current_batch == tc.phase2_end and is_phase2:
            is_phase2 = False  # the second phase ends here
            if args.with_phase3:
                logging.info(f'Entering the third phase at epoch {tc.state.current_batch // batches_per_epoch}')
                args.with_backbone = False
                reset_early_stopper(args, tc)
            else:
                break
        # save model conditionally
        if tc.accelerator.is_main_process and tc.early_stopper is None:
            now = datetime.now()
            if (now - model_saved).total_seconds() > 60 * args.save_every:
                save_state(tc.accelerator, tc.state_path)
                model_saved = datetime.now()
        # model evaluation
        if tc.state.current_batch % batches_per_epoch == 0:
            epoch_scalars = {}
            unwrapped_model = tc.accelerator.unwrap_model(tc.model)
            if is_phase2:
                # temporary workaround TODO
                if tc.accelerator.is_main_process:
                    gradients_norm(tc.writer, unwrapped_model._base_model, tc.state.current_batch, is_ee=False)
            # temporary workaround TODO
            if tc.accelerator.is_main_process:
                gradients_norm(tc.writer, unwrapped_model._heads, tc.state.current_batch, is_ee=True)
            in_training_eval(args, tc)
            if tc.accelerator.is_main_process:
                epoch_scalars[f'Train/Progress_Epoch_EE_Phase'] = tc.state.current_batch // batches_per_epoch
                epoch_scalars[
                    f'Train/Progress_Epoch_Phase{2 if is_phase2 else 3}'] = tc.state.current_batch // batches_per_epoch
                tc.writer.log_scalars(epoch_scalars)
        # batch preparation
        try:
            X, y = next(train_iter)
        except StopIteration:
            train_iter = iter(tc.train_loader)
            X, y = next(train_iter)
        if mixup_fn is not None:
            X, y = mixup_fn(X, y)
        # forward
        set_for_training(args, tc)
        output = tc.model(X)

        for head_i, head_output in enumerate(output):
            assert not head_output.isnan().any(), f'{tc.state.current_batch=} {head_i=} {head_output=}'
        # loss computation
        losses = [tc.criterion(y_pred, y) for y_pred in output]
        if args.aggregate_loss == 'gp_style':
            # TODO implement the freezing scheme from:
            # https://github.com/lancopku/Early-Exit/blob/main/model/modeling_bert.py#L901-L903
            denominator = len(losses) * (len(losses) + 1) / 2
            loss = sum((i + 1) * loss_ for i, loss_ in enumerate(losses)) / denominator
            gpf_aux_loss = tc.model.aux_loss()
            loss = loss + gpf_aux_loss
            if tc.accelerator.is_main_process:
                train_scalars[f'Train/GPF auxiliary loss'] = gpf_aux_loss.item()
        elif args.aggregate_loss == 'arith_mean':
            denominator = len(losses)
            loss = sum(losses) / denominator
        elif args.aggregate_loss == 'sum':
            loss = sum(losses)
        elif args.aggregate_loss == 'alternative_ge':
            nb_heads = len(losses)
            loss = sum(l / (nb_heads - i) for i, l in enumerate(losses))
        elif args.aggregate_loss == 'ablation_from_l2w_increasing':
            nb_heads = len(losses)
            weights = torch.linspace(0.6, 1.4, steps=nb_heads).tolist()
            loss = sum(l * w for l, w in zip(losses, weights))
        elif args.aggregate_loss == 'ablation_from_l2w_deceasing':
            nb_heads = len(losses)
            weights = torch.linspace(1.4, 0.6, steps=nb_heads).tolist()
            loss = sum(l * w for l, w in zip(losses, weights))
        elif args.aggregate_loss == 'sdn':
            cur_coeffs = [0.01 + (tc.state.current_batch + 1) *
                          (max_coeff / (tc.state.current_batch + 1)) for max_coeff in max_coeffs]
            cur_coeffs = [min(max_coeff, cur_coeff) for max_coeff, cur_coeff in zip(max_coeffs, cur_coeffs)]
            # divides by number of exits so that hyperaparms could be more or less
            # ported from 'arith_mean' runs
            loss = sum(l * w for l, w in zip(losses, cur_coeffs)) / len(losses)
        else:
            raise ValueError("Chosen aggregation of losses is not implemented!")

        if args.auxiliary_loss_type is not None:
            aux_loss = AUX_LOSS_MAP[args.auxiliary_loss_type](output)
            loss = loss + args.auxiliary_loss_weight * aux_loss
            if tc.accelerator.is_main_process:
                train_scalars[f'Train/Auxiliary loss'] = aux_loss.item()

        # gradient computation and training step
        tc.optimizer.zero_grad(set_to_none=True)
        # if args.gather_stats and tc.state.current_batch % batches_per_epoch == 0:  # GRADIENT STATS
        #     tc.grad_statistician_ee.compute_gradient_stats(tc, phase='train', scope='periodic', step=tc.state.current_batch)
        #     tc.grad_statistician_ee.compute_gradient_stats(tc, phase='test', scope='periodic', step=tc.state.current_batch)
        if args.gather_stats and tc.state.current_batch % 20 == 0:
            dominance_values = defaultdict(float)
            unwrapped_model = tc.accelerator.unwrap_model(tc.model)
            named_params = {k: v for k, v in unwrapped_model.named_parameters()}
            # compute the aggregated gradient, save it
            grad_total_list = torch.autograd.grad(loss, list(named_params.values()), retain_graph=True,
                                                  allow_unused=True)
            grad_total = torch.cat([g.flatten() for g in grad_total_list])
            # for each exit
            for i, ic_loss in enumerate(losses):
                # compute the gradient for the current exit
                grad_ic_list = torch.autograd.grad(ic_loss, list(named_params.values()), retain_graph=True,
                                                   allow_unused=True, materialize_grads=True)
                grad_ic = torch.cat([g.flatten() for g in grad_ic_list])
                # compute and save the cosine
                ic_grad_dominance = torch.nn.functional.cosine_similarity(grad_total, grad_ic, dim=0)
                dominance_values[f'gradient_dominance_cosine_{i=}'] = ic_grad_dominance.item()
                # for each parameter
                for n, g_total, g_ic in zip(named_params.keys(), grad_total_list, grad_ic_list):
                    param_ic_grad_dominance = torch.nn.functional.cosine_similarity(g_total.flatten(), g_ic.flatten(),
                                                                                    dim=0)
                    dominance_values[f'gradient_dominance_cosine_{i=}/{n}'] = param_ic_grad_dominance.item()
            # log the values
            tc.writer.log_scalars(dominance_values)

        tc.accelerator.backward(loss)
        if args.clip_grad_norm is not None:
            total_norm = tc.accelerator.clip_grad_norm_(tc.model.parameters(), args.clip_grad_norm)
            if tc.accelerator.is_main_process:
                train_scalars['Train/Gradient norm'] = total_norm.item()
        tc.optimizer.step()
        if tc.scheduler is not None:
            # log LRs
            if tc.accelerator.is_main_process:
                for i, lr in enumerate(get_lrs(tc.optimizer)):
                    train_scalars[f'Train/Group {i} LR'] = lr
            if args.scheduler_class == 'reduce_on_plateau':
                tc.scheduler.step(loss)
            else:
                tc.scheduler.step()
        if tc.accelerator.is_main_process:
            train_scalars['Train/Loss____per_batch'] = loss.item()
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
            train_losses, train_accs = test_earlyexiting_classification(tc.accelerator,
                                                                        tc.model,
                                                                        tc.eval_loader,
                                                                        tc.criterion_type,
                                                                        batches=0)
            if tc.accelerator.is_main_process:
                for i in range(tc.accelerator.unwrap_model(tc.model).number_of_heads):
                    early_scalars[f'Early stopping/Head {i} eval loss'] = train_losses[i]
                    early_scalars[f'Early stopping/Head {i} eval accuracy'] = train_accs[i]
            if tc.early_stopper.should_early_stop(tc,
                                                  validation_measures=train_losses if args.early_stopping_metric_type == 'loss' else train_accs):
                logging.info(
                    f'Early stopping after {tc.state.current_batch // batches_per_epoch} epochs: {train_losses=}, {train_accs=}')
                if args.with_phase3 and is_phase2:
                    is_phase2 = False  # the second phase ends here
                    logging.info(f'Entering the third phase at epoch {tc.state.current_batch // batches_per_epoch}')
                    args.with_backbone = False
                    reset_early_stopper(args, tc)
                else:
                    break


def final_eval(args, tc):
    eval_thresholds = 100 if args.eval_thresholds is None else args.eval_thresholds
    if not tc.final_path.exists():
        if tc.accelerator.is_main_process:
            save_state(tc.accelerator, tc.state_path)
        # test on testset
        head_preds, labels = get_preds_earlyexiting(tc.accelerator, tc.model, tc.test_loader)
        if tc.accelerator.is_main_process:
            final_results_scalars = {}
            final_results = {}
            final_results['args'] = args
            unwrapped_model = tc.accelerator.unwrap_model(tc.model)
            final_results['model_state'] = unwrapped_model.state_dict()
            head_costs, model_params = benchmark_earlyexiting(unwrapped_model, tc.test_loader)
            eval_results = evaluate_earlyexiting_classification(unwrapped_model, head_preds, labels, head_costs,
                                                                eval_thresholds)
            final_results.update(eval_results)
            for i in range(unwrapped_model.number_of_heads):
                final_results_scalars[f'Eval/Head {i} test accuracy'] = eval_results['head_scores'][i]
            base_model_cost = head_costs[-1].by_module()['_base_model']
            head_cost_fraction = (head_costs[-1].total() - base_model_cost) / base_model_cost
            final_results_scalars[f'Eval/Head cost fraction'] = head_cost_fraction
            # tc.writer.add_scalar(f'Eval/Head cost fraction', head_cost_fraction, global_step=tc.state.current_batch)
            logging.info(f'Head cost fraction: {head_cost_fraction}')
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
    # setup_mi(args, tc)
    setup_early_stopper(args, tc)
    setup_gradient_equalibrum(args, tc)
    setup_gradient_statistician_ee(args, tc)
    load_state(tc.accelerator, tc.state_path)
    training_loop(args, tc)
    final_eval(args, tc)


def main():
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    train(args)


if __name__ == '__main__':
    main()
