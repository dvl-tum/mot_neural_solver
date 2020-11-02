import sacred
from sacred import Experiment

from mot_neural_solver.utils.misc import make_deterministic, get_run_str_and_save_dir

from mot_neural_solver.path_cfg import OUTPUT_PATH, DATA_PATH
import os.path as osp

from mot_neural_solver.pl_module.pl_module import MOTNeuralSolver
from mot_neural_solver.utils.evaluation import compute_mot_metrics

import pandas as pd

from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG=False

ex = Experiment()
ex.add_config('configs/tracking_cfg.yaml')
ex.add_config({'run_id': 'evaluation',
               'add_date': True,
               'precomputed_embeddings': True})

@ex.automain
def main(_config, _run):

    #sacred.commands.print_config(_run) # No need to print config, as it's overwritten by the one from the ckpt.
    make_deterministic(12345)

    run_str, save_dir = get_run_str_and_save_dir(_config['run_id'], None, _config['add_date'])
    out_files_dir = osp.join(save_dir, 'mot_files')

    # Load model from checkpoint and update config entries that may vary from the ones used in training
    model = MOTNeuralSolver.load_from_checkpoint(checkpoint_path=_config['ckpt_path'] if osp.exists(_config['ckpt_path'])  else osp.join(OUTPUT_PATH, _config['ckpt_path']))
    model.hparams.update({'eval_params':_config['eval_params'],
                          'data_splits':_config['data_splits']})
    model.hparams['dataset_params']['precomputed_embeddings'] = _config['precomputed_embeddings']
    model.hparams['dataset_params']['img_batch_size'] = _config['dataset_params']['img_batch_size']

    # Get output MOT results files
    test_dataset = model.test_dataset()
    constr_satisf_rate = model.track_all_seqs(dataset=test_dataset,
                                              output_files_dir = out_files_dir,
                                              use_gt = False,
                                              verbose=True)

    # If there's GT available (e.g. if testing on train sequences) try to compute MOT metrics
    try:
        mot_metrics_summary = compute_mot_metrics(gt_path=osp.join(DATA_PATH, 'MOT_eval_gt'),
                                                  out_mot_files_path=out_files_dir,
                                                  seqs=test_dataset.seq_names,
                                                  print_results = False)
        mot_metrics_summary['constr_sr'] = constr_satisf_rate

        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'expand_frame_repr', False):
            cols = [col for col in mot_metrics_summary.columns if col in _config['eval_params']['mot_metrics_to_log']]
            print("\n" + str(mot_metrics_summary[cols]))

    except:
        print("Could not evaluate the given results")
