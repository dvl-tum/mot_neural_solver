"""
This file contains MOTSeqProcessor, which does all the necessary work to prepare tracking data (that is, detections,
imgs for every frame, sequence metainfo, and possibly ground truth files) for training and evaluation.

MOT Sequences from different datasets (e.g. MOT15 and MOT17) might have different storage structure, this is why we
define different 'sequence types', and map different sequences to them in _SEQ_TYPES.

For each type in _SEQ_TYPES, we define a different function to load a pd.DataFrame with their detections, a dictionary
with sequence metainfo (frames per second, img resolution, static/moving camera, etc.), and another pd.DataFrame with
ground truth boxes information. See e.g. MOT17loader.py as an example.

Once these three objects have been loaded, the rest of the sequence processing (e.g. matching ground truth boxes to
detections, storing embeddings, etc.) is performed in common, by the methods in MOTSeqProcessor

If you want to add new/custom sequences:
    1) Store with the same structure as e.g. MOT challenge mot_seqs (one directo    ry per sequence):
    2) Add its sequences' names (dir names) and sequence type to the corresponding/new 'seq_type' in _SEQ_TYPES
    3) Modify / write a det_df loader function for the new 'seq_type' (see MOTCha_loader.py as an example)
    If you had to write a new loader function:
        4) Add the new (seq_type, det_df loader function) to SEQ_TYPE_DETS_DF_LOADER
    Make sure that 'fps' and other metadata is available in the scene_info_dict returned by your loader
"""
import pandas as pd
import numpy as np

from lapsolver import solve_dense

from mot_neural_solver.data.seq_processing.MOTCha_loader import get_mot_det_df, get_mot_det_df_from_gt
from mot_neural_solver.data.seq_processing.MOT15_loader import get_mot15_det_df, get_mot15_det_df_from_gt
from mot_neural_solver.utils.iou import iou
from mot_neural_solver.utils.rgb import BoundingBoxDataset

import os
import os.path as osp

import shutil

import torch
from torch.utils.data import DataLoader

##########################################################
# Definition of available Sequences
##########################################################

# We define 'sequence types' for different MOT sequences, depending on the kind of processing they require (e.g. file
# storage structure, etc.). Each different type requires a different loader function that returns a pandas DataFrame
# with the right format from its detection file (see e.g. MOTCha_loader.py).

# Assign a loader func to each Sequence Type
_SEQ_TYPE_DETS_DF_LOADER = {'MOT': get_mot_det_df,
                            'MOT_gt': get_mot_det_df_from_gt,
                            'MOT15': get_mot15_det_df,
                            'MOT15_gt': get_mot15_det_df_from_gt}

# Determines whether boxes are allowed to have some area outside the image (all GT annotations in MOT15 are inside img
# hence we crop its detections to also be inside it)
_ENSURE_BOX_IN_FRAME = {'MOT': False,
                        'MOT_gt': False,
                        'MOT15': True,
                        'MOT15_gt': False}


# We now map each sequence name to a sequence type in _SEQ_TYPES
_SEQ_TYPES = {}

# MOT20 Sequences
mot20_seqs = [f'MOT20-{seq_num:02}{det}' for seq_num in (1, 2, 3, 5) for det in ('', '-GT')]
mot20_seqs += [f'MOT20-{seq_num:02}' for seq_num in (4, 6, 7, 8)]
for seq_name in mot20_seqs:
    if 'GT' in seq_name:
        _SEQ_TYPES[seq_name] = 'MOT_gt'

    else:
        _SEQ_TYPES[seq_name] = 'MOT'


# MOT17 Sequences
mot17_seqs = [f'MOT17-{seq_num:02}-{det}' for seq_num in (2, 4, 5, 9, 10, 11, 13) for det in ('DPM', 'SDP', 'FRCNN', 'GT')]
mot17_seqs += [f'MOT17-{seq_num:02}-{det}' for seq_num in (1, 3, 6, 7, 8, 12, 14) for det in ('DPM', 'SDP', 'FRCNN')]
for seq_name in mot17_seqs:
    if 'GT' in seq_name:
        _SEQ_TYPES[seq_name] = 'MOT_gt'

    else:
        _SEQ_TYPES[seq_name] = 'MOT'

# MOT15 Sequences 
mot15_seqs = ['KITTI-17', 'KITTI-13', 'ETH-Sunnyday', 'ETH-Bahnhof', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte']
mot15_seqs += ['ADL-Rundle-6', 'ADL-Rundle-8', 'Venice-2', 'ETH-Pedcross2']
mot15_seqs += [seq_name + '-GT' for seq_name in mot15_seqs]
mot15_seqs += ['Venice-1', 'KITTI-16', 'KITTI-19', 'ADL-Rundle-1', 'ADL-Rundle-3', 'AVG-TownCentre']
mot15_seqs += ['ETH-Crossing', 'ETH-Linthescher', 'ETH-Jelmoli', 'PETS09-S2L2', 'TUD-Crossing']
for seq_name in mot15_seqs:
    if 'GT' in seq_name:
        _SEQ_TYPES[seq_name] = 'MOT15_gt'

    else:
        _SEQ_TYPES[seq_name] = 'MOT15'

##########################################################################################
# Classes used to store and process detections for every sequence
##########################################################################################

class DataFrameWSeqInfo(pd.DataFrame):
    """
    Class used to store each sequences's processed detections as a DataFrame. We just add a metadata atribute to
    pandas DataFrames it so that sequence metainfo such as fps, etc. can be stored in the attribute 'seq_info_dict'.
    This attribute survives serialization.
    This solution was adopted from:
    https://pandas.pydata.org/pandas-docs/stable/development/extending.html#define-original-properties
    """
    _metadata = ['seq_info_dict']

    @property
    def _constructor(self):
        return DataFrameWSeqInfo

class MOTSeqProcessor:
    """
    Class to process detections files coming from different mot_seqs.
    Main method is process_detections. It does the following:
    - Loads a DataFrameWSeqInfo (~pd.DataFrame) from a  detections file (self.det_df) via a the 'det_df_loader' func
    corresponding to the sequence type (mapped via _SEQ_TYPES)
    - Adds Sequence Info to the df (fps, img size, moving/static camera, etc.) as an additional attribute (_get_det_df)
    - If GT is available, assigns GT identities to the detected boxes via bipartite matching (_assign_gt)
    - Stores the df on disk (_store_det_df)
    - If required, precomputes CNN embeddings for every detected box and stores them on disk (_store_embeddings)

    The stored information assumes that each MOT sequence has its own directory. Inside it all processed data is
    stored as follows:
        +-- <Sequence name>
        |   +-- processed_data
        |       +-- det
        |           +-- <dataset_params['det_file_name']>.pkl # pd.DataFrame with processed detections and metainfo
        |       +-- embeddings
        |           +-- <dataset_params['det_file_name']> # Precomputed embeddings for a set of detections
        |               +-- <CNN Name >
        |                   +-- {frame1}.jpg
        |                   ...
        |                   +-- {frameN}.jpg
    """
    def __init__(self, dataset_path, seq_name, dataset_params, cnn_model = None, logger = None):
        self.seq_name = seq_name
        self.dataset_path = dataset_path
        self.seq_type = _SEQ_TYPES[seq_name]

        self.det_df_loader = _SEQ_TYPE_DETS_DF_LOADER[self.seq_type]
        self.dataset_params = dataset_params

        self.cnn_model = cnn_model

        self.logger = logger

    def _ensure_boxes_in_frame(self):
        """
        Determines whether boxes are allowed to have some area outside the image (all GT annotations in MOT15 are inside
        the frame hence we crop its detections to also be inside it)
        """
        initial_bb_top = self.det_df['bb_top'].values.copy()
        initial_bb_left = self.det_df['bb_left'].values.copy()

        self.det_df['bb_top'] = np.maximum(self.det_df['bb_top'].values, 0).astype(int)
        self.det_df['bb_left'] = np.maximum(self.det_df['bb_left'].values, 0).astype(int)

        bb_top_diff = self.det_df['bb_top'].values - initial_bb_top
        bb_left_diff = self.det_df['bb_left'].values - initial_bb_left

        self.det_df['bb_height'] -= bb_top_diff
        self.det_df['bb_width'] -= bb_left_diff

        img_height, img_width = self.det_df.seq_info_dict['frame_height'], self.det_df.seq_info_dict['frame_width']
        self.det_df['bb_height'] = np.minimum(img_height - self.det_df['bb_top'], self.det_df['bb_height']).astype(int)
        self.det_df['bb_width'] = np.minimum(img_width - self.det_df['bb_left'], self.det_df['bb_width']).astype(int)

    def _assign_gt(self):
        """
        Assigns a GT identity to every detection in self.det_df, based on the ground truth boxes in self.gt_df.
        The assignment is done frame by frame via bipartite matching.
        """
        if self.det_df.seq_info_dict['has_gt'] and not self.det_df.seq_info_dict['is_gt']:
            print(f"Assigning ground truth identities to detections to sequence {self.seq_name}")
            for frame in self.det_df['frame'].unique():
                frame_detects = self.det_df[self.det_df.frame == frame]
                frame_gt = self.gt_df[self.gt_df.frame == frame]

                # Compute IoU for each pair of detected / GT bounding box
                iou_matrix = iou(frame_detects[['bb_top', 'bb_left', 'bb_bot', 'bb_right']].values,
                                 frame_gt[['bb_top', 'bb_left', 'bb_bot', 'bb_right']].values)

                iou_matrix[iou_matrix < self.dataset_params['gt_assign_min_iou']] = np.nan
                dist_matrix = 1 - iou_matrix
                assigned_detect_ixs, assigned_detect_ixs_ped_ids = solve_dense(dist_matrix)
                unassigned_detect_ixs = np.array(list(set(range(frame_detects.shape[0])) - set(assigned_detect_ixs)))

                assigned_detect_ixs_index = frame_detects.iloc[assigned_detect_ixs].index
                assigned_detect_ixs_ped_ids = frame_gt.iloc[assigned_detect_ixs_ped_ids]['id'].values
                unassigned_detect_ixs_index = frame_detects.iloc[unassigned_detect_ixs].index

                self.det_df.loc[assigned_detect_ixs_index, 'id'] = assigned_detect_ixs_ped_ids
                self.det_df.loc[unassigned_detect_ixs_index, 'id'] = -1  # False Positives

    def _get_det_df(self):
        """
        Loads a pd.DataFrame where each row contains a detections bounding box' coordinates information (self.det_df),
        and, if available, a similarly structured pd.DataFrame with ground truth boxes.
        It also adds seq_info_dict as an attribute to self.det_df, containing sequence metainformation (img size,
        fps, whether it has ground truth annotations, etc.)
        """
        self.det_df, seq_info_dict, self.gt_df = self.det_df_loader(self.seq_name, self.dataset_path, self.dataset_params)

        self.det_df = DataFrameWSeqInfo(self.det_df)
        self.det_df.seq_info_dict = seq_info_dict

        # Some further processing
        if self.seq_type in _ENSURE_BOX_IN_FRAME and _ENSURE_BOX_IN_FRAME[self.seq_type]:
            self._ensure_boxes_in_frame()

        # Add some additional box measurements that might be used for graph construction
        self.det_df['bb_bot'] = (self.det_df['bb_top'] + self.det_df['bb_height']).values
        self.det_df['bb_right'] = (self.det_df['bb_left'] + self.det_df['bb_width']).values
        self.det_df['feet_x'] = self.det_df['bb_left'] + 0.5 * self.det_df['bb_width']
        self.det_df['feet_y'] = self.det_df['bb_top'] + self.det_df['bb_height']

        # Just a sanity check. Sometimes there are boxes that lay completely outside the frame
        frame_height, frame_width = self.det_df.seq_info_dict['frame_height'], self.det_df.seq_info_dict['frame_width']
        conds = (self.det_df['bb_width'] > 0) & (self.det_df['bb_height'] > 0)
        conds = conds & (self.det_df['bb_right'] > 0) & (self.det_df['bb_bot'] > 0)
        conds  =  conds & (self.det_df['bb_left'] < frame_width) & (self.det_df['bb_top'] < frame_height)
        self.det_df = self.det_df[conds].copy()

        self.det_df.sort_values(by = 'frame', inplace = True)
        self.det_df['detection_id'] = np.arange(self.det_df.shape[0]) # This id is used for future tastks

        return self.det_df

    def _store_df(self):
        """
        Stores processed detections DataFrame in disk.
        """
        processed_dets_path = osp.join(self.det_df.seq_info_dict['seq_path'], 'processed_data', 'det')
        os.makedirs(processed_dets_path, exist_ok = True)
        det_df_path = osp.join(processed_dets_path, self.det_df.seq_info_dict['det_file_name'] + '.pkl')
        self.det_df.to_pickle(det_df_path)
        print(f"Finished processing detections for seq {self.seq_name}. Result was stored at {det_df_path}")

    def _store_embeddings(self):
        """
        Stores node and reid embeddings corresponding for each detection in the given sequence.
        Embeddings are stored at:
        {seq_info_dict['seq_path']}/processed_data/embeddings/{seq_info_dict['det_file_name']}/dataset_params['node/reid_embeddings_dir'}/FRAME_NUM.pt
        Essentially, each set of processed detections (e.g. raw, prepr w. frcnn, prepr w. tracktor) has a storage path, corresponding
        to a detection file (det_file_name). Within this path, different CNNs, have different directories
        (specified in dataset_params['node_embeddings_dir'] and dataset_params['reid_embeddings_dir']), and within each
        directory, we store pytorch tensors corresponding to the embeddings in a given frame, with shape
        (N, EMBEDDING_SIZE), where N is the number of detections in the frame.
        """
        from time import time
        assert self.cnn_model is not None
        assert self.dataset_params['reid_embeddings_dir'] is not None and self.dataset_params['node_embeddings_dir'] is not None

        # Create dirs to store embeddings
        node_embeds_path = osp.join(self.det_df.seq_info_dict['seq_path'], 'processed_data/embeddings',
                                   self.det_df.seq_info_dict['det_file_name'], self.dataset_params['node_embeddings_dir'])

        reid_embeds_path = osp.join(self.det_df.seq_info_dict['seq_path'], 'processed_data/embeddings',
                                   self.det_df.seq_info_dict['det_file_name'], self.dataset_params['reid_embeddings_dir'])

        if osp.exists(node_embeds_path):
            print("Found existing stored node embeddings. Deleting them and replacing them for new ones")
            shutil.rmtree(node_embeds_path)

        if osp.exists(reid_embeds_path):
            print("Found existing stored reid embeddings. Deleting them and replacing them for new ones")
            shutil.rmtree(reid_embeds_path)

        os.makedirs(node_embeds_path)
        os.makedirs(reid_embeds_path)

        # Compute and store embeddings
        # If there are more than 100k detections, we split the df into smaller dfs avoid running out of RAM, as it
        # requires storing all embedding into RAM (~6 GB for 100k detections)

        print(f"Computing embeddings for {self.det_df.shape[0]} detections")

        num_dets = self.det_df.shape[0]
        max_dets_per_df = int(1e5) # Needs to be larger than the maximum amount of dets possible to have in one frame

        frame_cutpoints = [self.det_df.frame.iloc[i] for i in np.arange(0, num_dets , max_dets_per_df, dtype=int)]
        frame_cutpoints += [self.det_df.frame.iloc[-1] + 1]

        for frame_start, frame_end in zip(frame_cutpoints[:-1], frame_cutpoints[1:]):
            sub_df_mask = self.det_df.frame.between(frame_start, frame_end - 1)
            sub_df = self.det_df.loc[sub_df_mask]

            #print(sub_df.frame.min(), sub_df.frame.max())
            bbox_dataset = BoundingBoxDataset(sub_df, seq_info_dict=self.det_df.seq_info_dict,
                                              return_det_ids_and_frame = True)
            bbox_loader = DataLoader(bbox_dataset, batch_size=self.dataset_params['img_batch_size'], pin_memory=True,
                                     num_workers=4)

            # Feed all bboxes to the CNN to obtain node and reid embeddings
            self.cnn_model.eval()
            node_embeds, reid_embeds = [], []
            frame_nums, det_ids = [], []
            with torch.no_grad():
                for frame_num, det_id, bboxes in bbox_loader:
                    node_out, reid_out = self.cnn_model(bboxes.cuda())
                    node_embeds.append(node_out.cpu())
                    reid_embeds.append(reid_out.cpu())
                    frame_nums.append(frame_num)
                    det_ids.append(det_id)
            #print("IT TOOK ", time() - t)
            #print(f"Finished computing embeddings")

            det_ids = torch.cat(det_ids, dim=0)
            frame_nums = torch.cat(frame_nums, dim=0)

            node_embeds = torch.cat(node_embeds, dim=0)
            reid_embeds = torch.cat(reid_embeds, dim=0)

            # Add detection ids as first column of embeddings, to ensure that embeddings are loaded correctly
            node_embeds = torch.cat((det_ids.view(-1, 1).float(), node_embeds), dim=1)
            reid_embeds = torch.cat((det_ids.view(-1, 1).float(), reid_embeds), dim=1)

            # Save embeddings grouped by frame
            for frame in sub_df.frame.unique():
                mask = frame_nums == frame
                frame_node_embeds = node_embeds[mask]
                frame_reid_embeds = reid_embeds[mask]

                frame_node_embeds_path = osp.join(node_embeds_path, f"{frame}.pt")
                frame_reid_embeds_path = osp.join(reid_embeds_path, f"{frame}.pt")

                torch.save(frame_node_embeds, frame_node_embeds_path)
                torch.save(frame_reid_embeds, frame_reid_embeds_path)

            #print("Finished storing embeddings")
        print("Finished computing and storing embeddings")

    def process_detections(self):
        # See class header
        self._get_det_df()
        self._assign_gt()
        self._store_df()

        if self.dataset_params['precomputed_embeddings']:
            self._store_embeddings()

        return self.det_df

    def load_or_process_detections(self):
        """
        Tries to load a set of processed detections if it's safe to do so. otherwise, it processes them and stores the
        result
        """
        # Check if the processed detections file already exists.
        seq_path = osp.join(self.dataset_path, self.seq_name)
        det_file_to_use = self.dataset_params['det_file_name'] if not self.seq_name.endswith('GT') else 'gt'
        seq_det_df_path = osp.join(seq_path, 'processed_data/det', det_file_to_use + '.pkl')

        # If loading precomputed embeddings, check if embeddings have already been stored (otherwise, we need to process dets again)
        node_embeds_path = osp.join(seq_path, 'processed_data/embeddings', det_file_to_use, self.dataset_params['node_embeddings_dir'])
        reid_embeds_path = osp.join(seq_path, 'processed_data/embeddings', det_file_to_use, self.dataset_params['reid_embeddings_dir'])
        try:
            num_frames = len(pd.read_pickle(seq_det_df_path)['frame'].unique())
            processed_dets_exist = True
        except:
            num_frames = -1
            processed_dets_exist = False

        embeds_ok = osp.exists(node_embeds_path) and len(os.listdir(node_embeds_path)) ==num_frames
        embeds_ok = embeds_ok and osp.exists(reid_embeds_path) and len(os.listdir(reid_embeds_path)) == num_frames
        embeds_ok = embeds_ok or not self.dataset_params['precomputed_embeddings']

        if processed_dets_exist and embeds_ok and not self.dataset_params['overwrite_processed_data']:
            print(f"Loading processed dets for sequence {self.seq_name} from {seq_det_df_path}")
            seq_det_df = pd.read_pickle(seq_det_df_path).reset_index().sort_values(by=['frame', 'detection_id'])

        else:
            print(f'Detections for sequence {self.seq_name} need to be processed. Starting processing')
            seq_det_df = self.process_detections()

        seq_det_df.seq_info_dict['seq_path'] = seq_path

        return seq_det_df
