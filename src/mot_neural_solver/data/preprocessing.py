import pandas as pd

import torch
from torchvision.ops import nms

class FRCNNPreprocessor:
    """
    Class used to preprocess datasets without tracktor. It filters out false detections, refines bounding box
    coordinates, and applines nms.
    """
    def __init__(self, obj_detect, prepr_params):
        self.obj_detect = obj_detect
        self.detect_score_thresh= prepr_params['detect_score_thresh']
        self.nms_thresh = prepr_params['nms_thresh']

        self.results_dfs = []
        self.curr_frame = 1

    @torch.no_grad()
    def step(self, blob):
        if blob['dets'].shape[1] != 0:
            self.obj_detect.load_image(blob['img'])
            boxes, scores = self.obj_detect.predict_boxes(blob['dets'].squeeze(dim=0).cuda())

            # Filter out detections with low score
            keep = scores >= self.detect_score_thresh
            boxes, scores = boxes[keep], scores[keep]

            # Apply NMS
            keep = nms(boxes, scores, self.nms_thresh)
            boxes, scores = boxes[keep], scores[keep]

            # Replace bottom right coordinates for height and width (MOTChallenge format)
            boxes[:, 2] -= boxes[:, 0]
            boxes[:, 3] -= boxes[:, 1]

            frame_df = pd.DataFrame(columns = ['bb_left', 'bb_top', 'bb_width', 'bb_height'], data = boxes.cpu().numpy())
            frame_df['frame'] = self.curr_frame
            frame_df['conf'] = scores.cpu().numpy()
            self.results_dfs.append(frame_df)

        self.curr_frame += 1

    def reset(self):
        self.results_dfs = []
        self.curr_frame  = 1

    def save_results(self, file_path):
        final_results = pd.concat(self.results_dfs)
        final_results['bb_left'] += 1 # MOT bbox annotations are 1 -based
        final_results['bb_top'] += 1 # MOT bbox annotations are 1 -based
        final_results['id'] = -1
        final_results[['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf']].to_csv(file_path,
                                                                                                    header=False,
                                                                                                    index=False)