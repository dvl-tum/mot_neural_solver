from mot_neural_solver.path_cfg import DATA_PATH
import configparser

import os
import os.path as osp

import numpy as np
import pandas as pd

import shutil

MOV_CAMERA_DICT = { 'MOT17-02-GT': False,
                    'MOT17-02-SDP': False,
                    'MOT17-02-FRCNN': False,
                    'MOT17-02-DPM': False,

                    'MOT17-04-GT': False,
                    'MOT17-04-SDP': False,
                    'MOT17-04-FRCNN': False,
                    'MOT17-04-DPM': False,

                    'MOT17-05-GT': True,
                    'MOT17-05-SDP': True,
                    'MOT17-05-FRCNN': True,
                    'MOT17-05-DPM': True,

                    'MOT17-09-GT': False,
                    'MOT17-09-SDP': False,
                    'MOT17-09-FRCNN': False,
                    'MOT17-09-DPM': False,

                    'MOT17-10-GT': True,
                    'MOT17-10-SDP': True,
                    'MOT17-10-FRCNN': True,
                    'MOT17-10-DPM': True,

                    'MOT17-11-GT': True,
                    'MOT17-11-SDP': True,
                    'MOT17-11-FRCNN': True,
                    'MOT17-11-DPM': True,

                    'MOT17-13-GT': True,
                    'MOT17-13-SDP': True,
                    'MOT17-13-FRCNN': True,
                    'MOT17-13-DPM': True,

                    'MOT17-14-SDP': True,
                    'MOT17-14-FRCNN': True,
                    'MOT17-14-DPM': True,

                    'MOT17-12-SDP': True,
                    'MOT17-12-FRCNN': True,
                    'MOT17-12-DPM': True,

                    'MOT17-08-SDP': False,
                    'MOT17-08-FRCNN': False,
                    'MOT17-08-DPM': False,

                    'MOT17-07-SDP': True,
                    'MOT17-07-FRCNN': True,
                    'MOT17-07-DPM': True,

                    'MOT17-06-SDP': True,
                    'MOT17-06-FRCNN': True,
                    'MOT17-06-DPM': True,

                    'MOT17-03-SDP': False,
                    'MOT17-03-FRCNN': False,
                    'MOT17-03-DPM': False,

                    'MOT17-01-SDP': False,
                    'MOT17-01-FRCNN': False,
                    'MOT17-01-DPM': False
                    }
# MOT20 (there is small camera movement in some sequences, but it's ok)
MOV_CAMERA_DICT.update({f'MOT20-{seq_num:02}{gt}': False for seq_num in range(1, 9) for gt in ('', '-GT')})



DET_COL_NAMES = ('frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf')
GT_COL_NAMES = ('frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'label', 'vis')

def _add_frame_path_mot17(det_df, seq_name, data_root_path):
    # Add each image's path from  MOT17Det data dir
    seq_name_wo_dets = '-'.join(seq_name.split('-')[:-1])
    det_seq_path = osp.join(data_root_path.replace('Labels', 'Det'), seq_name_wo_dets)
    add_frame_path = lambda frame_num: osp.join(det_seq_path, det_seq_path, f'img1/{frame_num:06}.jpg')
    det_df['frame_path'] = det_df['frame'].apply(add_frame_path)

def _build_scene_info_dict_mot17(seq_name, data_root_path, dataset_params):
    info_file_path = osp.join(data_root_path, seq_name, 'seqinfo.ini')
    cp = configparser.ConfigParser()
    cp.read(info_file_path)

    seq_info_dict = {'seq': seq_name,
                     'seq_path': osp.join(data_root_path, seq_name),
                     'det_file_name': dataset_params['det_file_name'],

                     'frame_height': int(cp.get('Sequence', 'imHeight')),
                     'frame_width': int(cp.get('Sequence', 'imWidth')),

                     'seq_len': int(cp.get('Sequence', 'seqLength')),
                     'fps': int(cp.get('Sequence', 'frameRate')),
                     'mov_camera': MOV_CAMERA_DICT[seq_name],

                     'has_gt': osp.exists(osp.join(data_root_path, seq_name, 'gt'))}
    return seq_info_dict

def get_mot_det_df(seq_name, data_root_path, dataset_params):

    seq_path = osp.join(data_root_path, seq_name)
    detections_file_path = osp.join(seq_path, f"det/{dataset_params['det_file_name']}.txt")
    det_df = pd.read_csv(detections_file_path, header=None)

    # Number and order of columns is always assumed to be the same
    det_df = det_df[det_df.columns[:len(DET_COL_NAMES)]]
    det_df.columns = DET_COL_NAMES

    det_df['bb_left'] -= 1 # Coordinates are 1 based
    det_df['bb_top'] -= 1

    # If id already contains an ID assignment (e.g. using tracktor output), keep it
    if len(det_df['id'].unique()) > 1:
        det_df['tracktor_id'] = det_df['id']

    # Add each image's path (in MOT17Det data dir)
    if 'MOT17' in seq_name:
        _add_frame_path_mot17(det_df, seq_name, data_root_path)

    else:
        det_df['frame_path'] = det_df['frame'].apply(lambda frame_num: osp.join(seq_path, f'img1/{frame_num:06}.jpg'))

    assert osp.exists(det_df['frame_path'].iloc[0])

    seq_info_dict = _build_scene_info_dict_mot17(seq_name, data_root_path, dataset_params)
    seq_info_dict['is_gt'] = False
    if seq_info_dict['has_gt']: # Return the corresponding ground truth, if available, for the ground truth assignment
        gt_file_path = osp.join(seq_path, f"gt/gt.txt")
        gt_df = pd.read_csv(gt_file_path, header=None)
        gt_df = gt_df[gt_df.columns[:len(GT_COL_NAMES)]]
        gt_df.columns = GT_COL_NAMES
        gt_df['bb_left'] -= 1  # Coordinates are 1 based
        gt_df['bb_top'] -= 1
        gt_df = gt_df[gt_df['label'].isin([1, 2, 7, 8, 12])].copy() # Classes 7, 8, 12 are 'ambiguous' and tracking
                                                                    # them is not penalized, hence we keep them for the
                                                                    # GT Assignment
                                                                    # See https://arxiv.org/pdf/1603.00831.pdf
        gt_df['bb_bot'] = (gt_df['bb_top'] + gt_df['bb_height']).values
        gt_df['bb_right'] = (gt_df['bb_left'] + gt_df['bb_width']).values

        # Store the gt file in the common evaluation path
        gt_to_eval_path = osp.join(DATA_PATH, 'MOT_eval_gt', seq_name, 'gt')
        os.makedirs(gt_to_eval_path, exist_ok=True)
        shutil.copyfile(gt_file_path, osp.join(gt_to_eval_path, 'gt.txt'))

    else:
        gt_df = None

    return det_df, seq_info_dict, gt_df

def get_mot_det_df_from_gt(seq_name, data_root_path, dataset_params):

    # Create a dir to store Ground truth data in case it does not exist yet
    seq_path = osp.join(data_root_path, seq_name)
    if not osp.exists(seq_path):
        os.mkdir(seq_path)

        # Copy ground truth and seq info from a seq that has this ground truth.
        if 'MOT17' in seq_name: # For MOT17 we use e.g. the seq with DPM detections (any will do)
            src_seq_path = osp.join(data_root_path, seq_name[:-2] + 'DPM')

        else: # Otherwise just use the actual sequence
            src_seq_path = osp.join(data_root_path, seq_name[:-3])

        shutil.copy(osp.join(src_seq_path, 'seqinfo.ini'), osp.join(seq_path, 'seqinfo.ini'))
        shutil.copytree(osp.join(src_seq_path, 'gt'), osp.join(seq_path, 'gt'))

    detections_file_path = osp.join(data_root_path, seq_name, f"gt/gt.txt")
    det_df = pd.read_csv(detections_file_path, header=None)

    # Number and order of columns is always assumed to be the same
    det_df = det_df[det_df.columns[:len(GT_COL_NAMES)]]
    det_df.columns = GT_COL_NAMES

    det_df['bb_left'] -= 1 # Coordinates are 1 based
    det_df['bb_top'] -= 1

    # VERY IMPORTANT: Filter out non Target Classes (e.g. vehicles, occluderst, etc.) (see: https://arxiv.org/abs/1603.00831)
    det_df = det_df[det_df['label'].isin([1, 2])].copy()


    if 'MOT17' in seq_name:
        _add_frame_path_mot17(det_df, seq_name, data_root_path)

    else:
        det_df['frame_path'] = det_df['frame'].apply(lambda frame_num: osp.join(seq_path[:-3], f'img1/{frame_num:06}.jpg'))
    assert osp.exists(det_df['frame_path'].iloc[0])

    seq_info_dict = _build_scene_info_dict_mot17(seq_name, data_root_path, dataset_params)

    # Correct the detections file name to contain the 'gt' as name
    seq_info_dict['det_file_name'] = 'gt'
    seq_info_dict['is_gt'] = True

    # Store the gt file in the common evaluation path
    gt_file_path = osp.join(seq_path, f"gt/gt.txt")
    gt_to_eval_path = osp.join(DATA_PATH, 'MOT_eval_gt', seq_name, 'gt')
    os.makedirs(gt_to_eval_path, exist_ok=True)
    shutil.copyfile(gt_file_path, osp.join(gt_to_eval_path, 'gt.txt'))

    return det_df, seq_info_dict, None