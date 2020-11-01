import numpy as np
import pandas as pd

import os
import os.path as osp
import pickle
from pathlib import Path

import motmetrics as mm

from copy import deepcopy
from collections import OrderedDict

import torch
from torch_scatter import scatter_add
from pytorch_lightning import Callback

import re

from mot_neural_solver.path_cfg import DATA_PATH
from mot_neural_solver.utils.misc import load_pickle, save_pickle

###########################################################################
# MOT Metrics
###########################################################################

# Formatting for MOT Metrics reporting
mh = mm.metrics.create()
MOT_METRICS_FORMATERS = mh.formatters
MOT_METRICS_NAMEMAP = mm.io.motchallenge_metric_names
MOT_METRICS_NAMEMAP.update({'norm_' + key: 'norm_' + val for key, val in MOT_METRICS_NAMEMAP.items()})
MOT_METRICS_FORMATERS.update({'norm_' + key: val for key, val in MOT_METRICS_FORMATERS.items()})
MOT_METRICS_FORMATERS.update({'constr_sr': MOT_METRICS_FORMATERS['mota']})


def compute_mot_metrics(gt_path, out_mot_files_path, seqs, print_results = True):
    """
    The following code is adapted from
    https://github.com/cheind/py-motmetrics/blob/develop/motmetrics/apps/eval_motchallenge.py
    It computes all MOT metrics from a set of output tracking files in MOTChallenge format
    Args:
        gt_path: path where MOT ground truth files are stored. Each gt file must be stored as
        <SEQ NAME>/gt/gt.txt
        out_mot_files_path: path where output files are stored. Each file must be named <SEQ NAME>.txt
        seqs: Names of sequences to be evaluated

    Returns:
        Individual and overall MOTmetrics for all sequeces
    """
    def _compare_dataframes(gts, ts):
        """Builds accumulator for each sequence."""
        accs = []
        names = []
        for k, tsacc in ts.items():
            if k in gts:
                accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
                names.append(k)

        return accs, names

    mm.lap.default_solver = 'lapsolver'
    gtfiles = [os.path.join(gt_path, i, 'gt/gt.txt') for i in seqs]
    tsfiles = [os.path.join(out_mot_files_path, '%s.txt' % i) for i in seqs]

    gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D')) for f in tsfiles])

    mh = mm.metrics.create()
    accs, names = _compare_dataframes(gt, ts)

    # We will need additional metrics to compute IDF1, etc. from different splits inf CrossValidationEvaluator
    summary = mh.compute_many(accs, names=names,
                              metrics=mm.metrics.motchallenge_metrics + ['num_objects',
                                                                         'idtp', 'idfn', 'idfp', 'num_predictions'],
                              generate_overall=True)
    if print_results:
        print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

    return summary

class MOTMetricsLogger(Callback):
    """
    Callback to compute MOT Validation metrics during training
    """
    def __init__(self, compute_oracle_results):
        super(MOTMetricsLogger).__init__()
        self.compute_oracle_results = compute_oracle_results

    def _compute_mot_metrics(self, epoch_num, pl_module, oracle_results = False):
        constr_satisf_rate = pl_module.track_all_seqs(dataset=self.dataset,
                                                      output_files_dir=self.output_files_dir,
                                                      use_gt=oracle_results)

        # Compute MOT Metrics
        mot_metrics_summary = compute_mot_metrics(gt_path=osp.join(DATA_PATH, 'MOT_eval_gt'),
                                                  out_mot_files_path=self.output_files_dir,
                                                  seqs=self.dataset.seq_names)
        mot_metrics_summary['constr_sr'] = constr_satisf_rate
        mot_metrics_summary['epoch_num'] = epoch_num + 1

        return mot_metrics_summary

    def on_train_start(self, trainer, pl_module):
        self.available_data = len(trainer.val_dataloaders) > 0 and len(trainer.val_dataloaders[0]) > 0
        if self.available_data:
            self.dataset = trainer.val_dataloaders[0].dataset
            # Determine the path in which MOT results will be stored
            if trainer.logger is not None:
                save_dir = osp.join(trainer.logger.save_dir, trainer.logger.name, trainer.logger.version )

            else:
                save_dir = trainer.default_save_path

            self.output_files_dir = osp.join(save_dir, 'mot_files')
            self.output_metrics_dir = osp.join(save_dir, 'mot_metrics')
            os.makedirs(self.output_metrics_dir, exist_ok=True)

        # Compute oracle results if needed
        if self.available_data and self.compute_oracle_results:
            mot_metrics_summary = self._compute_mot_metrics(trainer.current_epoch, pl_module, oracle_results=True)
            print(mot_metrics_summary)
            oracle_path = osp.join(self.output_metrics_dir, 'oracle.npy')
            save_pickle(mot_metrics_summary.to_dict(), oracle_path)
            trainer.oracle_metrics = mot_metrics_summary

    def on_epoch_end(self, trainer, pl_module):
        # Compute MOT metrics on validation data, save them and log them
        if self.available_data:
            mot_metrics_summary = self._compute_mot_metrics(trainer.current_epoch, pl_module, oracle_results=False)
            metrics_path = osp.join(self.output_metrics_dir, f'epoch_{trainer.current_epoch + 1:03}.npy')
            save_pickle(mot_metrics_summary.to_dict(), metrics_path)

            if self.compute_oracle_results:
                for metric in pl_module.hparams['eval_params']['mot_metrics_to_norm']:
                    mot_metrics_summary['norm_' + metric] = mot_metrics_summary[metric] / trainer.oracle_metrics[metric]

            if pl_module.logger is not None and hasattr(pl_module.logger, 'experiment'):
                metric_names = pl_module.hparams['eval_params']['mot_metrics_to_log']
                if pl_module.hparams['eval_params']['log_per_seq_metrics']:
                    metrics_log ={f'{metric}/val/{seq}': met_dict[seq] for metric, met_dict in mot_metrics_summary.items() for seq in
                                  list(self.dataset.seq_names) + ['OVERALL'] if metric in metric_names}

                else:
                    metrics_log ={f'{metric}/val': met_dict['OVERALL'] for metric, met_dict in mot_metrics_summary.items()
                                  if metric in metric_names}
                    pl_module.logger.log_metrics(metrics_log, step = trainer.global_step)


###########################################################################
# Computation of MOT metrics with cross-validation
###########################################################################

class CrossValidationEvaluator:
    def __init__(self, path_to_search, run_id):
        self.path_to_search = path_to_search
        self.run_id = run_id


    def _extract_split_num(self, dir_name):
        split = dir_name.split('split_')[1].split('_')[0]
        return int(split)

    def _get_per_split_paths(self):
        """
        Given a path (path_to_search) and a a string (experiment_name), it will search for all files inside path_to_search that
        contain the string experiment_name and return a list with one name per split. If there is more than one file that
        contains tag_name and a given split, it will return the most recent one.
        """

        pattern = '(\d\d-\d\d_\d\d:\d\d_)?' + self.run_id + '_split_[0-9]' # Optional date + tag_name + split
        dir_candidates = [dir_name for dir_name in os.listdir(self.path_to_search) if re.search(pattern, dir_name)]

        # First, get all the different available paths per split
        paths_per_split = {}
        for dir_path in dir_candidates:
            split_num = self._extract_split_num(dir_path)
            if split_num != -1:
                if split_num not in paths_per_split:
                    paths_per_split[split_num] = [dir_path]
                else:
                    paths_per_split[split_num].append(dir_path)

        # In case there's more than one for a given split, choose the one that was created later
        return [max(dir_list) for dir_list in paths_per_split.values()]

    def get_metrics_data(self):
        """
        Get a list with Ids of experiments whose data we want to retrieve. It inferes them from tag_name. It searches
        output files that contain it and are as recent as possible and from their name, it gets their Experiment Id
        """

        # Get paths from which we will get experiment Ids:
        per_split_metrics_dir =  self._get_per_split_paths()

        print(f"Retrieving metrics from experiments with Run IDs: {sorted(per_split_metrics_dir)}")

        # Load metrics information from every split at every iteration number
        metric_dfs = []
        oracle_metric_dfs = []
        for dir in per_split_metrics_dir:
            per_epoch_files = os.listdir(osp.join(self.path_to_search, dir, 'mot_metrics'))
            for file in per_epoch_files:
                split_iter_metrics = load_pickle(osp.join(self.path_to_search, dir, 'mot_metrics', file))
                split_iter_metrics = pd.DataFrame(split_iter_metrics)
                split_iter_metrics = split_iter_metrics.drop('OVERALL').reset_index().rename(columns=  {'index': 'scene'})

                if file.lower().startswith('oracle'):
                    oracle_metric_dfs.append(split_iter_metrics)

                else:
                    metric_dfs.append(split_iter_metrics)

        # Concatenate all DataFrames into a single one containing metrics from all sequences at every iteration
        overall_metrics_df = pd.concat(metric_dfs)
        if oracle_metric_dfs:
            oracle_metric_dfs = pd.concat(oracle_metric_dfs)

        return overall_metrics_df, oracle_metric_dfs

    def _compute_per_epoch_MOTA_and_prec(self, metrics_df):
        """
        Computes overall MOT metrics over scenes from different training splits
        """
        # Group metrics by scene and keep record on how many scenes where evaluated at each iteration
        scenes_per_iter = metrics_df.reset_index().groupby('epoch_num')['scene'].agg(lambda x: tuple(x.unique()))
        per_epoch_overall_vals = metrics_df.groupby(['epoch_num']).sum()
        per_epoch_overall_vals = per_epoch_overall_vals.join(scenes_per_iter)
        all_scenes = per_epoch_overall_vals['scene'].iloc[0]
        per_epoch_overall_vals['has_all_scenes'] = per_epoch_overall_vals['scene'].apply(lambda x: set(all_scenes).issubset(x))

        if 'constr_sr' in per_epoch_overall_vals and per_epoch_overall_vals ['constr_sr'].isnull().sum()  == 0:
            per_epoch_overall_vals['constr_sr'] = metrics_df.groupby(['epoch_num'])['constr_sr'].mean()

        # Compute MOTA and MOTA Log
        per_epoch_overall_vals['mota'] = 100*(1 - (per_epoch_overall_vals['num_misses'] + per_epoch_overall_vals['num_false_positives']+
                                                  per_epoch_overall_vals['num_switches']) / per_epoch_overall_vals['num_objects'])
        per_epoch_overall_vals['MOTA Log'] = 100*(1 - (per_epoch_overall_vals['num_misses'] + per_epoch_overall_vals['num_false_positives']+
                                                  np.log10(per_epoch_overall_vals['num_switches'])) / per_epoch_overall_vals['num_objects'])

        # Compute ID metrics
        #per_epoch_overall_vals['idp'] = 100* per_epoch_overall_vals['idtp'] / (per_epoch_overall_vals['idtp'] + per_epoch_overall_vals['idfp'])
        per_epoch_overall_vals['idr'] = 100* per_epoch_overall_vals['idtp'] / (per_epoch_overall_vals['idtp'] + per_epoch_overall_vals['idfn'])
        per_epoch_overall_vals['idf1'] = 100*2*per_epoch_overall_vals['idtp'] / (per_epoch_overall_vals['num_predictions'] + per_epoch_overall_vals['num_objects'])

        return per_epoch_overall_vals

    def _choose_best_epoch_results(self, per_epoch_metrics, best_method_metric):
        """
        Chooses overall (best) results, based on the value of the metric (column) 'best_method_criteria'
        """
        valid_indices = per_epoch_metrics[per_epoch_metrics['has_all_scenes']].index
        best_iter = max(valid_indices, key=lambda x: per_epoch_metrics.loc[x][best_method_metric])
        best_row = per_epoch_metrics.loc[best_iter]
        best_metric_val = per_epoch_metrics.loc[best_iter][best_method_metric]

        return best_iter, best_row, best_metric_val


    def evaluate(self, cols_to_norm, best_method_metric):
        per_epoch_metrics, oracle_metrics=  self.get_metrics_data()
        per_epoch_metrics = self._compute_per_epoch_MOTA_and_prec(per_epoch_metrics)

        if isinstance(oracle_metrics, pd.DataFrame):
            oracle_metrics = self._compute_per_epoch_MOTA_and_prec(oracle_metrics)

        for col in cols_to_norm:
            per_epoch_metrics['norm_' + col] = 100 * per_epoch_metrics[col] / oracle_metrics[col].iloc[0]

        best_iter, best_row, best_metric_val  = self._choose_best_epoch_results(per_epoch_metrics, best_method_metric)

        return per_epoch_metrics, best_iter, best_row, best_metric_val


###########################################################################
# Computation of other metrics to monitor training
###########################################################################

def fast_compute_class_metric(test_preds, test_sols, class_metrics = ('accuracy', 'recall', 'precision')):
    """
    Computes manually (i.e. without sklearn functions) accuracy, recall and predicision.

    Args:
        test_preds: numpy array/ torch tensor of size N with discrete output vals
        test_sols: numpy array/torch tensor of size N with binary labels
        class_metrics: tuple with a subset of values from ('accuracy', 'recall', 'precision') indicating which
        metrics to report

    Returns:
        dictionary with values of accuracy, recall and precision
    """
    with torch.no_grad():

        TP = ((test_sols == 1) & (test_preds == 1)).sum().float()
        FP = ((test_sols == 0) & (test_preds == 1)).sum().float()
        TN = ((test_sols == 0) & (test_preds == 0)).sum().float()
        FN = ((test_sols == 1) & (test_preds == 0)).sum().float()

        accuracy = (TP + TN) / (TP + FP + TN + FN)
        recall = TP / (TP + FN) if TP + FN > 0 else torch.tensor(0)
        precision = TP / (TP + FP) if TP + FP > 0 else torch.tensor(0)

    class_metrics_dict =  {'accuracy': accuracy.item(), 'recall': recall.item(), 'precision': precision.item()}
    class_metrics_dict = {met_name: class_metrics_dict[met_name] for met_name in class_metrics}

    return class_metrics_dict

def compute_constr_satisfaction_rate(graph_obj, edges_out, undirected_edges = True, return_flow_vals = False):
    """
    Determines the proportion of Flow Conservation inequalities that are satisfied.
    For each node, the sum of incoming (resp. outgoing) edge values must be less or equal than 1.

    Args:
        graph_obj: 'Graph' object
        edges_out: BINARIZED output values for edges (1 if active, 0 if not active)
        undirected_edges: determines whether each edge in graph_obj.edge_index appears in both directions (i.e. (i, j)
        and (j, i) are both present (undirected_edges =True), or only (i, j), with  i<j (undirected_edges=False)
        return_flow_vals: determines whether the sum of incoming /outglong flow for each node must be returned

    Returns:
        constr_sat_rate: float between 0 and 1 indicating the proprtion of inequalities that are satisfied

    """
    # Get tensors indicataing which nodes have incoming and outgoing flows (e.g. nodes in first frame have no in. flow)
    edge_ixs = graph_obj.edge_index
    if undirected_edges:
        sorted, _ = edge_ixs.t().sort(dim = 1)
        sorted = sorted.t()
        div_factor = 2. # Each edge is predicted twice, hence, we divide by 2
    else:
        sorted = edge_ixs # Edges (i.e. node pairs) are already sorted
        div_factor = 1.  # Each edge is predicted once, hence, hence we divide by 1.

    # Compute incoming and outgoing flows for each node
    flow_out = scatter_add(edges_out, sorted[0],dim_size=graph_obj.num_nodes) / div_factor
    flow_in = scatter_add(edges_out, sorted[1], dim_size=graph_obj.num_nodes) / div_factor


    # Determine how many inequalitites are violated
    violated_flow_out = (flow_out > 1).sum()
    violated_flow_in = (flow_in > 1).sum()

    # Compute the final constraint satisfaction rate
    violated_inequalities = (violated_flow_in + violated_flow_out).float()
    flow_out_constr, flow_in_constr= sorted[0].unique(), sorted[1].unique()
    num_constraints = len(flow_out_constr) + len(flow_in_constr)
    constr_sat_rate = 1 - violated_inequalities / num_constraints
    if not return_flow_vals:
        return constr_sat_rate.item()

    else:
        return constr_sat_rate.item(), flow_in, flow_out

def compute_perform_metrics(graph_out, graph_obj):
    """
    Computes both classification metrics and constraint satisfaction rate
    Args:
        graph_out: output of MPN, dict with key 'classified' edges, and val a list torch.Tensor of unnormalized loggits for
        every edge, at every messagepassing step.
        graph_obj: Graph Object

    Returns:
        dictionary with metrics summary
    """

    edges_out = graph_out['classified_edges'][-1]
    edges_out = (edges_out.view(-1) > 0).float()

    # Compute Classification Metrics
    class_metrics = fast_compute_class_metric(edges_out, graph_obj.edge_labels)

    # Compute and store the Constr Satisf. Rate
    class_metrics['constr_sr'] = compute_constr_satisfaction_rate(graph_obj, edges_out=edges_out)

    return class_metrics
