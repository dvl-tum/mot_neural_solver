import os.path as osp

from mot_neural_solver.path_cfg import OUTPUT_PATH
from mot_neural_solver.utils.evaluation import CrossValidationEvaluator

import pandas as pd

from sacred import Experiment

ex = Experiment()
ex.add_config('configs/tracking_cfg.yaml')
ex.add_config({'run_id': ''})


@ex.automain
def main(eval_params, run_id):
    evaluator = CrossValidationEvaluator(run_id=run_id,
                                         path_to_search = osp.join(OUTPUT_PATH, 'experiments'))

    per_iter_metrics, best_epoch, best_row, best_metric_val = evaluator.evaluate(cols_to_norm=eval_params['mot_metrics_to_norm'],
                                                                                best_method_metric = eval_params['best_method_criteria'])


    # Log the overall results obtained
    print( f"Best Metrics where obtained at epoch {best_epoch}, with {eval_params['best_method_criteria']} = {best_metric_val:.3f}")

    print("Per Iteration overall metrics:")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'expand_frame_repr', False):
        per_iter_overall_vals_ = per_iter_metrics.loc[per_iter_metrics['has_all_scenes']].copy()
        per_iter_overall_vals_['scene'] = per_iter_metrics['scene'].apply(
            lambda x: sorted(tuple(set([name.split('-')[1] for name in x]))))

        print("\n" + str(per_iter_overall_vals_[eval_params['mot_metrics_to_log'] + ['scene']]))
