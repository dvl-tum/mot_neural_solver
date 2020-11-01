import time
from os import path as osp

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import sacred
from sacred import Experiment
from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.datasets.factory import Datasets
from tracktor.tracker import Tracker

from mot_neural_solver.path_cfg import OUTPUT_PATH

from mot_neural_solver.data.seq_processing.MOTCha_loader import MOV_CAMERA_DICT as MOT17_MOV_CAMERA_DICT
from mot_neural_solver.data.seq_processing.MOT15_loader import MOV_CAMERA_DICT as MOT15_MOV_CAMERA_DICT
from mot_neural_solver.data.preprocessing import FRCNNPreprocessor

from mot_neural_solver.utils.misc import make_deterministic

ex = Experiment()
ex.add_config('configs/preprocessing_cfg.yaml')

@ex.automain
def main(dataset_names,  prepr_w_tracktor, frcnn_prepr_params,  tracktor_params, frcnn_weights,  _config, _log, _run):
    sacred.commands.print_config(_run)

    if prepr_w_tracktor:
        prepr_params = tracktor_params

    else:
        prepr_params = frcnn_prepr_params

    make_deterministic(prepr_params['seed'])
    MOV_CAMERA_DICT = {**MOT15_MOV_CAMERA_DICT, **MOT17_MOV_CAMERA_DICT}

    # object detection
    _log.info("Initializing object detector.")
    obj_detect = FRCNN_FPN(num_classes=2)
    obj_detect.load_state_dict(torch.load(osp.join(OUTPUT_PATH, frcnn_weights),
                                          map_location=lambda storage, loc: storage))
    obj_detect.eval()
    obj_detect.cuda()

    if prepr_w_tracktor:
        preprocessor = Tracker(obj_detect, None, prepr_params['tracker'])
    else:
        preprocessor = FRCNNPreprocessor(obj_detect, prepr_params)

    _log.info(f"Starting  preprocessing of datasets {dataset_names} with {'Tracktor' if prepr_w_tracktor else 'FRCNN'} \n")

    for dataset_name in dataset_names:
        dataset = Datasets(dataset_name)
        _log.info(f"Preprocessing {len(dataset)} sequences from dataset {dataset_name} \n")

        time_total = 0
        num_frames = 0
        for seq in dataset:
            preprocessor.reset()

            start = time.time()
            _log.info(f"Preprocessing : {seq}")
            if prepr_w_tracktor:
                preprocessor.do_align = tracktor_params['tracker']['do_align'] and (MOV_CAMERA_DICT[str(seq)])

            data_loader = DataLoader(seq, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)
            for i, frame in enumerate(tqdm(data_loader)):
                with torch.no_grad():
                    preprocessor.step(frame)
                num_frames += 1

            time_total += time.time() - start
            _log.info(f"Runtime for {seq}: {time.time() - start :.1f} s.")

            output_file_path = osp.join(seq.seq_path, 'det', prepr_params['det_file_name'])
            if prepr_w_tracktor:
                results = preprocessor.get_results()
                #seq.write_results(results, output_file_path)
            else:
                _log.info(f"Writing predictions in: {output_file_path}")
                #preprocessor.save_results(output_file_path)

        _log.info(f"Tracking runtime for all sequences (without evaluation or image writing): "
                  f"{time_total:.1f} s ({num_frames / time_total:.1f} Hz)")
