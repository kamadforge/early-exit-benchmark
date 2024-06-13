import logging
from datetime import datetime

import torch
from omegaconf import OmegaConf

from common import INIT_NAME_MAP, get_default_args
from eval import benchmark_earlyexiting, evaluate_earlyexiting_classification, \
    test_earlyexiting_classification, get_preds_earlyexiting
from train import TrainingContext, setup_accelerator, setup_data, setup_optimization, \
    setup_files_and_logging, setup_state
from utils import load_state, save_state, load_model, get_lrs, save_final, create_model, Mixup, gradients_norm


class EarlyStopperEarlyExit:
    def __init__(self, losses_num=1, patience=5, min_delta=0, metric_type='loss'):
        assert metric_type in {'loss', 'acc'}, f'Validation metric {metric_type} is not supported'
        self.patience = patience
        self.min_delta = min_delta

        self.losses_num = losses_num
        self.counter = [0] * losses_num
        self.extreme_validation_measures = [float('inf') if metric_type == 'loss' else - float('inf')] * losses_num
        self.metric_type = metric_type

    def should_early_stop(self, tc, validation_measures):
        better_model_found = False
        for idx, validation_measure in enumerate(validation_measures):
            if self.metric_type == 'loss':
                if validation_measure < self.extreme_validation_measures[idx]:
                    self.extreme_validation_measures[idx] = validation_measure
                    better_model_found = True
                    self.counter[idx] = 0
                elif validation_measure > (self.extreme_validation_measures[idx] + self.min_delta):
                    self.counter[idx] += 1
            elif self.metric_type == 'acc':
                if validation_measure > self.extreme_validation_measures[idx]:
                    self.extreme_validation_measures[idx] = validation_measure
                    better_model_found = True
                    self.counter[idx] = 0
                elif validation_measure < (self.extreme_validation_measures[idx] - self.min_delta):
                    self.counter[idx] += 1

        if better_model_found:
            save_state(tc.accelerator, tc.state_path)
            return False
        for i, best_value in enumerate(self.extreme_validation_measures):
            tc.writer.add_scalar(f'best_early_stopping/{self.metric_type}_{i}',
                                   best_value,
                                   global_step=tc.state.current_batch) 
        all_converged = sum([count >= self.patience for count in self.counter]) == self.losses_num
        if all_converged:
            load_state(tc.accelerator, tc.state_path)
            return True
        return False

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


def setup_early_stopper(args, tc):
    losses_num = tc.accelerator.unwrap_model(tc.model).number_of_heads
    if args.early_stopping:
        tc.early_stopper = EarlyStopperEarlyExit(losses_num=losses_num,
                                                 patience=args.early_stopping_patience,
                                                 min_delta=args.early_stopping_delta,
                                                 metric_type=args.early_stopping_metric_type)
    else:
        tc.early_stopper = None


def in_training_eval(args, tc):
    # if tc.state.current_batch in tc.eval_batch_list:
    if tc.accelerator.is_main_process:
        tc.writer.add_scalar('Train/Progress',
                                tc.state.current_batch / tc.last_batch,
                                global_step=tc.state.current_batch)
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
        for i in range(tc.accelerator.unwrap_model(tc.model).number_of_heads):
            tc.writer.add_scalar(f'Eval/Head {i} test loss', test_losses[i], global_step=tc.state.current_batch)
            tc.writer.add_scalar(f'Eval/Head {i} test accuracy', test_accs[i], global_step=tc.state.current_batch)
            tc.writer.add_scalar(f'Eval/Head {i} train loss', train_losses[i], global_step=tc.state.current_batch)  # train dataset without augmentation or eval dataset 
            tc.writer.add_scalar(f'Eval/Head {i} train accuracy', train_accs[i], global_step=tc.state.current_batch)  # train dataset without augmentation or eval dataset 


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

    batches_per_epoch = len(tc.train_loader)
    while tc.state.current_batch <= tc.last_batch:
        if tc.state.current_batch == tc.phase2_end and is_phase2:
            is_phase2 = False
            if args.with_phase3:
                logging.info(f'Entering the third phase at epoch {tc.state.current_batch // batches_per_epoch}')
                tc.accelerator.unwrap_model(tc.model).train('without_backbone')  # here the second phase ends
                setup_early_stopper(args, tc)
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
            #if is_phase2:
                #gradients_norm(tc.writer, tc.model._base_model, tc.state.current_batch, is_ee=False)
            #gradients_norm(tc.writer, tc.model._heads, tc.state.current_batch, is_ee=True)
            in_training_eval(args, tc)
            if tc.accelerator.is_main_process:
                tc.writer.add_scalar(f'Train/Progress_Epoch_EE_Phase',
                                    tc.state.current_batch // batches_per_epoch,
                                    global_step=tc.state.current_batch)
                tc.writer.add_scalar(f'Train/Progress_Epoch_Phase{2 if is_phase2 else 3}',
                                    tc.state.current_batch // batches_per_epoch,
                                    global_step=tc.state.current_batch)
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
        if 'gp' in args.model_class:
            # TODO implement the freezing scheme from:
            # https://github.com/lancopku/Early-Exit/blob/main/model/modeling_bert.py#L901-L903
            denominator = len(losses) * (len(losses) + 1) / 2
            loss = sum((i + 1) * loss_ for i, loss_ in enumerate(losses)) / denominator
            gpf_aux_loss = tc.model.aux_loss()
            loss = loss + gpf_aux_loss
            if tc.accelerator.is_main_process:
                tc.writer.add_scalar(f'Train/GPF auxiliary loss', gpf_aux_loss.item(),
                                     global_step=tc.state.current_batch)
        else:
            denominator = len(losses)
            loss = sum(l for l in losses) / denominator
        if args.auxiliary_loss_type is not None:
            aux_loss = AUX_LOSS_MAP[args.auxiliary_loss_type](output)
            loss = loss + args.auxiliary_loss_weight * aux_loss
            if tc.accelerator.is_main_process:
                tc.writer.add_scalar(f'Train/Auxiliary loss', aux_loss.item(), global_step=tc.state.current_batch)
        # gradient computation and training step
        tc.optimizer.zero_grad(set_to_none=True)
        tc.accelerator.backward(loss)
        if args.clip_grad_norm is not None:
            total_norm = tc.accelerator.clip_grad_norm_(tc.model.parameters(), args.clip_grad_norm)
            if tc.accelerator.is_main_process:
                tc.writer.add_scalar(f'Train/Gradient norm', total_norm.item(), global_step=tc.state.current_batch)
        tc.optimizer.step()
        if tc.scheduler is not None:
            # log LRs
            if tc.accelerator.is_main_process:
                for i, lr in enumerate(get_lrs(tc.optimizer)):
                    tc.writer.add_scalar(f'Train/Group {i} LR', lr, global_step=tc.state.current_batch)
            if args.scheduler_class == 'reduce_on_plateau':
                tc.scheduler.step(loss)
            else:
                tc.scheduler.step()
        if tc.accelerator.is_main_process:
            tc.writer.add_scalar(f'Train/Loss', loss.item(), global_step=tc.state.current_batch)
        # bookkeeping
        tc.state.current_batch += 1
        if tc.early_stopper is not None and tc.state.current_batch % min(batches_per_epoch, 300) == 0:
            # one epoch finished
            train_losses, train_accs = test_earlyexiting_classification(tc.accelerator,
                                                                        tc.model,
                                                                        tc.eval_loader,
                                                                        tc.criterion_type,
                                                                        batches=0)
            if tc.accelerator.is_main_process:
                for i in range(tc.accelerator.unwrap_model(tc.model).number_of_heads):
                    tc.writer.add_scalar(f'Early stopping/Head {i} eval loss', train_losses[i], global_step=tc.state.current_batch)
                    tc.writer.add_scalar(f'Early stopping/Head {i} eval accuracy', train_accs[i], global_step=tc.state.current_batch)

            if tc.early_stopper.should_early_stop(tc, validation_measures=train_losses if args.early_stopping_metric_type == 'loss' else train_accs):
                logging.info(f'Early stopping after {tc.state.current_batch // batches_per_epoch} epochs: {train_losses=}, {train_accs=}')
                if tc.state.current_batch <= tc.phase2_end and is_phase2:
                    is_phase2 = False
                    if args.with_phase3:
                        logging.info(f'Entering the third phase at epoch {tc.state.current_batch // batches_per_epoch}')
                        tc.model.train('without_backbone')  # here the second phase ends
                        setup_early_stopper(args, tc)
                    else:
                        break
                else:
                    break



def final_eval(args, tc):
    eval_thresholds = 100 if args.eval_thresholds is None else args.eval_thresholds
    if not tc.final_path.exists():
        if tc.accelerator.is_main_process:
            tc.state.current_batch = tc.last_batch + 1
            logging.info(f'The last save of the state.')
            save_state(tc.accelerator, tc.state_path)
        # test on testset
        head_preds, labels = get_preds_earlyexiting(tc.accelerator, tc.model, tc.test_loader)
        if tc.accelerator.is_main_process:
            final_results = {}
            final_results['args'] = args
            unwrapped_model = tc.accelerator.unwrap_model(tc.model)
            final_results['model_state'] = unwrapped_model.state_dict()
            head_costs, model_params = benchmark_earlyexiting(unwrapped_model, tc.test_loader)
            eval_results = evaluate_earlyexiting_classification(unwrapped_model, head_preds, labels, head_costs,
                                                                eval_thresholds)
            final_results.update(eval_results)
            for i in range(unwrapped_model.number_of_heads):
                tc.writer.add_scalar(f'Eval/Head {i} test accuracy', eval_results['head_scores'][i],
                                     global_step=tc.state.current_batch)
            base_model_cost = head_costs[-1].by_module()['_base_model']
            head_cost_fraction = (head_costs[-1].total() - base_model_cost) / base_model_cost
            tc.writer.add_scalar(f'Eval/Head cost fraction', head_cost_fraction, global_step=tc.state.current_batch)
            logging.info(f'Head cost fraction: {head_cost_fraction}')
            save_final(args, tc.final_path, final_results)


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
    setup_early_stopper(args, tc)
    training_loop(args, tc)
    final_eval(args, tc)


def main():
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    train(args)


if __name__ == '__main__':
    main()
