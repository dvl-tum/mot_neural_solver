import os.path as osp

import numpy as np
import pandas as pd

from mot_neural_solver.path_cfg import  DATA_PATH
from mot_neural_solver.data.splits import _SPLITS
from mot_neural_solver.data.mot_graph import MOTGraph
from mot_neural_solver.data.seq_processing.seq_processor import MOTSeqProcessor

import random

class MOTGraphDataset:
    """
    Main Dataset Class. It is used to sample graphs from a a set of MOT sequences by instantiating MOTGraph objects.
    It is used both for sampling small graphs for training, as well as for loading entire sequence's graphs
    for testing.
    Its main method is 'get_from_frame_and_seq', where given sequence name and a starting frame position, a graph is
    returned.
    """
    def __init__(self, dataset_params, mode, splits, logger = None, cnn_model = None):
        assert mode in ('train', 'val', 'test')
        self.dataset_params = dataset_params
        self.mode = mode # Can be either 'train', 'val' or 'test'
        self.logger = logger
        self.augment = self.dataset_params['augment'] and mode == 'train'

        self.cnn_model = cnn_model

        seqs_to_retrieve = self._get_seqs_to_retrieve_from_splits(splits)

        # Load all dataframes containing detection information in each sequence of the dataset
        self.seq_det_dfs, self.seq_info_dicts, self.seq_names = self._load_seq_dfs(seqs_to_retrieve)

        if self.seq_names:
            # Update each sequence's meatinfo with step sizes
            self._compute_seq_step_sizes()

            # Index the dataset (i.e. assign a pair (scene, starting frame) to each integer from 0 to len(dataset) -1)
            self.seq_frame_ixs = self._index_dataset()

    def _get_seqs_to_retrieve_from_splits(self, splits):
        """
        Given a string (or list of strings) of data_splits to use (e.g. 'mot17_train'), it determines the sequences
        that correspond to this split, and the path at which they are stored and stores the result as pairs of
        (dataset_path, seq_list) in 'seqs_to_retrieve'
        """
        if isinstance(splits, str):
            splits = [splits]

        seqs_to_retrieve = {}
        for split_name in splits:
            #seqs_path, seq_list = _SPLITS[split_name]
            # seqs_to_retrieve[osp.join(DATA_PATH, seqs_path)] = seq_list
            seqs = {osp.join(DATA_PATH, seqs_path): seq_list for seqs_path, seq_list in  _SPLITS[split_name].items()}
            seqs_to_retrieve.update(seqs)


        return seqs_to_retrieve

    def _load_seq_dfs(self, seqs_to_retrieve):
        """
        Loads all the detections dataframes corresponding to the seq_names that constitute the dataset
        Args:
            seqs_to_retrieve: dictionary of pairs (dataset_path: seq_list), where each seq_list is a set of
             sequence names to include in the dataset.

        Returns:
            seq_det_dfs: dictionary of Dataframes of detections corresponding to each sequence in the dataset
            seq_info_dicts: dictionary of dictionarys with metainfo for each sequence
            seq_names: a list of names of all sequences in the dataset
        """
        seq_names = []
        seq_info_dicts = {}
        seq_det_dfs = {}
        for dataset_path, seq_list in seqs_to_retrieve.items():
            for seq_name in seq_list:

                seq_processor = MOTSeqProcessor(dataset_path=dataset_path, seq_name=seq_name,
                                                dataset_params=self.dataset_params, cnn_model=self.cnn_model,
                                                logger=self.logger)
                seq_det_df = seq_processor.load_or_process_detections()

                # If we are dealing with ground truth and we visibility score, filter our detections that are not visible
                if 'vis' in seq_det_df:
                    seq_det_df = seq_det_df[seq_det_df['vis'] > self.dataset_params['gt_training_min_vis']]

                seq_names.append(seq_name)
                seq_info_dicts[seq_name] = seq_det_df.seq_info_dict
                seq_det_dfs[seq_name] = seq_det_df

        if seq_names:

            return seq_det_dfs, seq_info_dicts, seq_names

        else:
            return None, None, []

    def _compute_seq_step_sizes(self):
        """
        Determines the sampling rate of frames within a sequence, and updates each seq_info_dict with it
        Example: if a sequence is recorded at 25fps and we want to process 5 fps, then step_size = 5
        (i.e., we process 1 out 5 frames)
        """
        for seq_name, seq_info_dict in self.seq_info_dicts.items():
            seq_type = 'moving' if seq_info_dict['mov_camera'] else 'static'
            target_fps= self.dataset_params['target_fps_dict'][seq_type]
            scene_fps = seq_info_dict['fps']
            if scene_fps <= target_fps:
                step_size = 1

            else:
                step_size=  round(scene_fps / target_fps)

            self.seq_info_dicts[seq_name]['step_size'] = step_size

    def _get_last_frame_df(self):
        """
        Used for indexing the dataset. Determines all valid (seq_name, start_frame) pairs. To determine which pairs
        are valid, we need to know whether there are sufficient future detections at a given frame in a sequence to
        meet our minimum detections and target frames per graph

        Returns:
            last_frame_df: dataframe containing the last valid starting frame for each sequence.
        """
        last_graph_frame = self.dataset_params['frames_per_graph'] if self.dataset_params['frames_per_graph'] != 'max' else 1
        min_detects = 1 if self.dataset_params['min_detects'] is None else self.dataset_params['min_detects']

        last_frame_dict = {}
        for scene in self.seq_names:
            scene_df = self.seq_det_dfs[scene]
            scene_step_size = self.seq_info_dicts[scene]['step_size']
            max_frame = scene_df.frame.max() - (last_graph_frame * scene_step_size)  # Maximum frame at which
                                                                                     # we can start a graph and still
                                                                                     # have enough frames.
            min_detects_max_frame = scene_df.iloc[-(min_detects * scene_step_size)]['frame'] # Maximum frame at which
                                                                                             # we cans start a graph
                                                                                             # and still have enough dets.
            max_frame = min(max_frame, min_detects_max_frame)
            last_frame_dict[scene] = max_frame

        # Create a dataframe with the result
        last_frame_df = pd.DataFrame().from_dict(last_frame_dict, orient='index')
        last_frame_df = last_frame_df.reset_index().rename(columns={'index': 'seq_name', 0: 'last_frame'})

        return last_frame_df

    def _index_dataset(self):
        """
        For each sequence in our dataset we consider all valid frame positions (see 'get_last_frame_df()').
        Then, we build a tuple with all pairs (scene, start_frame). The ith element of our dataset corresponds to the
        ith pair in this tuple.
        Returns:
            tuple of tuples of valid (seq_name, frame_num) pairs from which a graph can be created
        """
        # Create  df containg all seq det_dfs
        concat_seq_dfs = []
        for seq_name, det_df in self.seq_det_dfs.items():
            seq_det_df_ = det_df.copy()
            seq_det_df_['seq_name'] = seq_name
            concat_seq_dfs.append(seq_det_df_)

        concat_seq_dfs = pd.concat(concat_seq_dfs, sort=False)

        # Get all valid (seq_name, starting_frame) pairs
        seq_frame_pairs = concat_seq_dfs[['seq_name', 'frame']].drop_duplicates()
        last_frame_df = self._get_last_frame_df()
        index_df = seq_frame_pairs.merge(last_frame_df, on = 'seq_name')
        index_df = index_df[index_df['frame']<=index_df['last_frame']]

        # Create a tuples with pairs (scene, starting_frame), that will be used to know which pair corresponds to each ix
        seq_frame_ixs = list((tuple(seq_frame) for seq_frame in index_df[['seq_name', 'frame']].values))

        # Shuffle ixs to ensure that if we only sample a subset of the dataloader, we still sample different seqs
        #random.shuffle(seq_frame_ixs)
        return seq_frame_ixs

    def get_from_frame_and_seq(self, seq_name, start_frame, max_frame_dist, end_frame = None, ensure_end_is_in = False,
                               return_full_object = False, inference_mode =False):
        """
        Method behind __getitem__ method. We load a graph object of the given sequence name, starting at 'start_frame'.

        Args:
            seq_name: string indicating which scene to get the graph from
            start_frame: int indicating frame num at which the graph should start
            end_frame: int indicating frame num at which the graph should end (optional)
            ensure_end_is_in: bool indicating whether end_frame needs to be in the graph
            return_full_object: bool indicating whether we need the whole MOTGraph object or only its Graph object
                                (Graph Network's input)

        Returns:
            mot_graph: output MOTGraph object or Graph object, depending on whethter return full_object == True or not

        """
        seq_det_df = self.seq_det_dfs[seq_name]
        seq_info_dict= self.seq_info_dicts[seq_name]
        seq_step_size = self.seq_info_dicts[seq_name]['step_size']

        # If doing data augmentation, randomly change the fps rate at which the scene is processed
        if self.mode == 'train' and self.augment and seq_step_size > 1:
            if np.random.rand() < self.dataset_params['p_change_fps_step']:
                seq_step_size = np.round(seq_step_size*(0.5 + np.random.rand())).astype(int)

        mot_graph = MOTGraph(dataset_params=self.dataset_params,
                             seq_info_dict= seq_info_dict,
                             seq_det_df=seq_det_df,
                             step_size=seq_step_size,
                             start_frame=start_frame,
                             end_frame=end_frame,
                             ensure_end_is_in=ensure_end_is_in,
                             max_frame_dist = max_frame_dist,
                             cnn_model = self.cnn_model,
                             inference_mode=inference_mode)

        if self.mode == 'train' and self.augment:
            mot_graph.augment()

        # Construct the Graph Network's input
        mot_graph.construct_graph_object()
        if self.mode in ('train', 'val'):
            mot_graph.assign_edge_labels()

        if return_full_object:
            return mot_graph

        else:
            return mot_graph.graph_obj

    def __len__(self):
        return len(self.seq_frame_ixs) if hasattr(self, 'seq_names') and self.seq_names else 0

    def __getitem__(self, ix):
        seq_name, start_frame = self.seq_frame_ixs[ix]
        return self.get_from_frame_and_seq(seq_name= seq_name,
                                           start_frame = start_frame,
                                           end_frame=None,
                                           ensure_end_is_in=False,
                                           return_full_object=False,
                                           inference_mode=False,
                                           max_frame_dist=self.dataset_params['max_frame_dist'])
