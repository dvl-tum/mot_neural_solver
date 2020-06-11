import sacred
from sacred import Experiment

from mot_neural_solver.utils.evaluation import MOTMetricsLogger
from mot_neural_solver.utils.misc import make_deterministic, get_run_str_and_save_dir, ModelCheckpoint

from mot_neural_solver.path_cfg import OUTPUT_PATH
import os.path as osp

from mot_neural_solver.pl_module.pl_module import MOTNeuralSolver

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
#from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG=False

ex = Experiment()
ex.add_config('configs/tracking_cfg.yaml')
ex.add_config({'run_id': 'train_w_default_config',
               'add_date': True,
               'cross_val_split': None})

@ex.config
def cfg(cross_val_split, eval_params, dataset_params, graph_model_params, data_splits):

    # Training requires the use of precomputed embeddings
    assert dataset_params['precomputed_embeddings'], "Training without precomp. embeddings is not supp"

    # Only use tracktor for postprocessing if tracktor was used for preprocessing
    if 'tracktor' not in dataset_params['det_file_name']:
        eval_params['add_tracktor_detects'] = False

    # Make sure that the edges encoder MLP input dim. matches the number of edge features used.
    graph_model_params['encoder_feats_dict']['edge_in_dim'] = len(dataset_params['edge_feats_to_use'])

    # Determine which sequences will be used for training  / validation
    if cross_val_split is not None:
        assert cross_val_split in (1, 2, 3), f"{cross_val_split} is not a valid cross validation split"
        data_splits['train'] =['mot15_train_gt', f'mot17_split_{cross_val_split}_train_gt']
        data_splits['val'] = [f'split_{cross_val_split}_val']

    # If we're training on all the available training data, disable validation
    if data_splits['train'] =='all_train' or data_splits['val'] is None:
        data_splits['val'] = []
        eval_params['val_percent_check'] = 0


@ex.automain
def main(_config, _run):

    sacred.commands.print_config(_run)
    make_deterministic(12345)

    model = MOTNeuralSolver(hparams = dict(_config))

    run_str, save_dir = get_run_str_and_save_dir(_config['run_id'], _config['cross_val_split'], _config['add_date'])

    if _config['train_params']['tensorboard']:
        logger = TensorBoardLogger(OUTPUT_PATH, name='experiments', version=run_str)

    else:
        logger = None

    ckpt_callback = ModelCheckpoint(save_epoch_start = _config['train_params']['save_epoch_start'],
                                    save_every_epoch = _config['train_params']['save_every_epoch'])

    trainer = Trainer(gpus=1,
                      callbacks=[MOTMetricsLogger(compute_oracle_results = _config['eval_params']['normalize_mot_metrics']), ckpt_callback],
                      weights_summary = None,
                      checkpoint_callback=False,
                      max_epochs=_config['train_params']['num_epochs'],
                      val_percent_check = _config['eval_params']['val_percent_check'],
                      check_val_every_n_epoch=_config['eval_params']['check_val_every_n_epoch'],
                      nb_sanity_val_steps=0,
                      logger =logger,
                      default_save_path=osp.join(OUTPUT_PATH, 'experiments', run_str))
    trainer.fit(model)





