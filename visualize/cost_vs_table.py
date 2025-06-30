import logging
import math

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf

from common import LOSS_NAME_MAP, PROB_TO_CONFIDENCE_MAP
from datasets_config import DATASETS_NAME_MAP
from eval import (benchmark_earlyexiting, evaluate_calibration, evaluate_classification, evaluate_earlyexiting_calibration, evaluate_earlyexiting_classification,
                  evaluate_earlyexiting_ood_detection,
                  evaluate_earlyexiting_regression, evaluate_ood_detection,
                  evaluate_regression, get_preds, get_preds_earlyexiting)
from utils import get_loader, load_model, retrieve_final
from visualize.cost_vs_plot import (PRETTY_NAME_DICT, compute_means_and_stds,
                                    get_default_args)


def generate_score_eff_table(core_stats, point_stats, ee_stats, name_dict, param, result_mult=100):
    saved_args = locals()
    cost_fractions = [0.25, 0.5, 0.75, 1.0, math.inf]
    # base_model_run_name = next(iter(core_stats.keys())) # TODO
    assert len(core_stats) > 0
    base_model_run_name = next(iter(core_stats.keys()))
    cost_100 = core_stats[base_model_run_name]['model_flops']
    # cost_100 = ee_stats[next(iter(ee_stats.keys()))]['head_flops'][-1]
    # collect
    ee_results = torch.zeros(len(ee_stats), len(cost_fractions))
    ee_stds = torch.zeros_like(ee_results)
    for i, run_name in enumerate(ee_stats.keys()):
        for j, cost_fraction in enumerate(cost_fractions):
            for k, threshold_cost in enumerate(ee_stats[run_name]['threshold_flops']):
                if threshold_cost > cost_fraction * cost_100:
                    if k > 0:
                        result_valid = True
                        k -= 1
                    else:
                        result_valid = False
                    break
            if result_valid:
                ee_results[i, j] = ee_stats[run_name]['threshold_scores'][k].item()
                ee_stds[i, j] = ee_stats[run_name]['threshold_scores_std'][k].item()
            else:
                ee_results[i, j] = math.nan
                ee_stds[i, j] = math.nan
    latex_code = ""
    for i, run_name in enumerate(ee_stats.keys()):
        name = name_dict[run_name]

        latex_code += f'{name}'
        for j in range(ee_results.size(-1)):
            result = f'${ee_results[i, j] * result_mult:.2f}$' if not math.isnan(ee_results[i, j]) else '-'
            result_std = f' $ \\pm {ee_stds[i, j] * result_mult:.2f}$' if not math.isnan(ee_stds[i, j]) else ''
            latex_code += f' & {result}{result_std}'
        latex_code += ' \\\\\n'
    return saved_args, latex_code

def main(args):
    logging.basicConfig(
        format=(
            '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'
        ),
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    core_stats = {}
    ee_stats = {}
    point_stats = {}
    name_dict = {}
    display_names = args.exp_names if args.display_names is None else args.display_names
    assert len(args.exp_names) == len(display_names)
    accelerator = Accelerator(split_batches=True)
    for exp_name in args.exp_names:
        for exp_id in args.exp_ids:
            run_name = f'{exp_name}_{exp_id}'
            logging.info(f'Processing for: {run_name} ({args.mode})')
            # TODO possibly split this into two scripts: one that generates outputs for both datasets and one that plots the data
            if args.mode == 'acc':
                state = retrieve_final(args, run_name)
                if 'thresholds' in state:
                    ee_stats[run_name] = state
                elif 'hyperparam_values' in state:
                    point_stats[run_name] = state
                else:
                    core_stats[run_name] = state
            elif args.mode == 'acc_by_confidence':
                model, run_args, state = load_model(args, exp_name, exp_id)
                model = accelerator.prepare(model)
                del state["model_state"]
                _, _, id_data = DATASETS_NAME_MAP[run_args.dataset](**run_args.dataset_args)
                id_dataloader = get_loader(id_data, run_args.batch_size, accelerator)
                if "thresholds" in state:
                    head_costs, _ = benchmark_earlyexiting(model, id_dataloader)
                    id_preds, id_labels = get_preds_earlyexiting(
                        accelerator, model, id_dataloader
                    )
                    state.update(
                        evaluate_earlyexiting_classification(
                            model,
                            id_preds,
                            id_labels,
                            head_costs,
                            100,
                            PROB_TO_CONFIDENCE_MAP[args.confidence_type],
                        )
                    )
                    ee_stats[run_name] = state
                else:
                    id_preds, id_labels = get_preds(accelerator, model, id_dataloader)
                    criterion_type = LOSS_NAME_MAP[run_args.loss_type]
                    _, score = evaluate_classification(id_preds, id_labels, criterion_type)
                    state.update({'final_score': score})
                    core_stats[run_name] = state   
            elif args.mode == 'calibration':
                model, run_args, state = load_model(args, exp_name, exp_id)
                _, _, id_data = DATASETS_NAME_MAP[run_args.dataset](run_args.dataset_args)
                model = accelerator.prepare(model)
                id_dataloader = get_loader(id_data, run_args.batch_size, accelerator)
                if 'thresholds' in state:
                    id_preds, id_labels = get_preds_earlyexiting(accelerator, model, id_dataloader)
                    state.update(
                        evaluate_earlyexiting_calibration(id_preds, id_labels, state['head_flops'],
                                                          state['thresholds']))
                    ee_stats[run_name] = state
                elif 'hyperparam_values' in state:
                    raise NotImplementedError('TODO')
                else:
                    id_preds, id_labels = get_preds(accelerator, model, id_dataloader)
                    state.update(evaluate_calibration(id_preds, id_labels))
                    core_stats[run_name] = state
            elif args.mode == 'ood_detection':
                model, run_args, state = load_model(args, exp_name, exp_id)
                model = accelerator.prepare(model)
                _, _, id_data = DATASETS_NAME_MAP[run_args.dataset](run_args.dataset_args)
                id_dataloader = get_loader(id_data, run_args.batch_size, accelerator)
                _, _, ood_data = DATASETS_NAME_MAP[args.ood_dataset](run_args.dataset_args)
                ood_dataloader = get_loader(ood_data, run_args.batch_size, accelerator)
                if 'thresholds' in state:
                    id_preds, _ = get_preds_earlyexiting(accelerator, model, id_dataloader)
                    ood_preds, _ = get_preds_earlyexiting(accelerator, model, ood_dataloader)
                    state.update(
                        evaluate_earlyexiting_ood_detection(id_preds, ood_preds, state['head_flops'],
                                                            state['thresholds']))
                    ee_stats[run_name] = state
                elif 'hyperparam_values' in state:
                    raise NotImplementedError('TODO')
                else:
                    id_preds, _ = get_preds(accelerator, model, id_dataloader)
                    ood_preds, _ = get_preds(accelerator, model, ood_dataloader)
                    state.update(evaluate_ood_detection(id_preds, ood_preds))
                    core_stats[run_name] = state
            elif args.mode == 'regression':
                model, run_args, state = load_model(args, exp_name, exp_id)
                _, _, id_data = DATASETS_NAME_MAP[run_args.dataset](**run_args.dataset_args)
                model = accelerator.prepare(model)
                id_dataloader = get_loader(id_data, run_args.batch_size, accelerator)
                if 'threshold_flops' in state:
                    id_preds, id_labels = get_preds_earlyexiting(accelerator, model, id_dataloader)
                    state.update(
                        evaluate_earlyexiting_regression(id_preds, id_labels,
                                                          state['head_flops']))
                    ee_stats[run_name] = state
                elif 'hyperparam_values' in state:
                    raise NotImplementedError('TODO')
                else:
                    id_preds, id_labels = get_preds(accelerator, model, id_dataloader)
                    state.update(evaluate_regression(id_preds, id_labels))
                    core_stats[run_name] = state
    core_stats, point_stats, ee_stats = compute_means_and_stds(args.exp_names, args.exp_ids, core_stats, point_stats,
                                                               ee_stats)
    for exp_name, display_name in zip(args.exp_names, display_names):
        name_dict[exp_name] = display_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output_dir / f'{args.output_name}_{args.mode}.txt'
    args_save_path = args.output_dir / f'{args.output_name}_{args.mode}_table.pt'
    with open(save_path, 'w') as f:
        saved_args, latex_code = generate_score_eff_table(core_stats, point_stats, ee_stats, name_dict, PRETTY_NAME_DICT[args.mode], result_mult=100 if args.mode != 'regression' else 1.0)
        torch.save(saved_args, args_save_path)
        f.write(latex_code)
        logging.info(f'Table saved to {str(save_path)}')


if __name__ == "__main__":
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    main(args)
