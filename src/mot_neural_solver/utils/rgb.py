import os.path as osp

import numpy as np

from PIL import Image
from skimage.io import imread
#from skimage.util import pad
from numpy import pad

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

class BoundingBoxDataset(Dataset):
    """
    Class used to process detections. Given a DataFrame (det_df) with detections of a MOT sequence, it returns
    the image patch corresponding to the detection's bounding box coordinates
    """
    def __init__(self, det_df, seq_info_dict, pad_ = True, pad_mode = 'mean', output_size = (128, 64),
                 return_det_ids_and_frame = False):
        self.det_df = det_df
        self.seq_info_dict = seq_info_dict
        self.pad = pad_
        self.pad_mode = pad_mode
        self.transforms= Compose((Resize(output_size), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406],
                                                                             std=[0.229, 0.224, 0.225])))

        # Initialize two variables containing the path and img of the frame that is being loaded to avoid loading multiple
        # times for boxes in the same image
        self.curr_img = None
        self.curr_img_path = None

        self.return_det_ids_and_frame = return_det_ids_and_frame

    def __len__(self):
        return self.det_df.shape[0]

    def __getitem__(self, ix):
        row = self.det_df.iloc[ix]

        # Load this bounding box' frame img, in case we haven't done it yet
        if row['frame_path'] != self.curr_img_path:
            self.curr_img = imread(row['frame_path'])
            self.curr_img_path = row['frame_path']

        frame_img = self.curr_img

        # Crop the bounding box, and pad it if necessary to
        bb_img = frame_img[int(max(0, row['bb_top'])): int(max(0, row['bb_bot'])),
                   int(max(0, row['bb_left'])): int(max(0, row['bb_right']))]
        if self.pad:
            x_height_pad = np.abs(row['bb_top'] - max(row['bb_top'], 0)).astype(int)
            y_height_pad = np.abs(row['bb_bot'] - min(row['bb_bot'], self.seq_info_dict['frame_height'])).astype(int)

            x_width_pad = np.abs(row['bb_left'] - max(row['bb_left'], 0)).astype(int)
            y_width_pad = np.abs(row['bb_right'] - min(row['bb_right'], self.seq_info_dict['frame_width'])).astype(int)

            bb_img = pad(bb_img, ((x_height_pad, y_height_pad), (x_width_pad, y_width_pad), (0, 0)), mode=self.pad_mode)

        bb_img = Image.fromarray(bb_img)
        if self.transforms is not None:
            bb_img = self.transforms(bb_img)

        if self.return_det_ids_and_frame:
            return row['frame'], row['detection_id'], bb_img
        else:
            return bb_img

def load_embeddings_from_imgs(det_df, dataset_params, seq_info_dict, cnn_model, return_imgs = False, use_cuda=True):
    """
    Computes embeddings for each detection in det_df with a CNN.
    Args:
        det_df: pd.DataFrame with detection coordinates
        seq_info_dict: dict with sequence meta info (we need frame dims)
        cnn_model: CNN to compute embeddings with. It needs to return BOTH node embeddings and reid embeddings
        return_imgs: bool, determines whether RGB images must also be returned

    Returns:
        (bb_imgs for each det or [], torch.Tensor with shape (num_detects, node_embeddings_dim), torch.Tensor with shape (num_detects, reidembeddings_dim))

    """
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    ds = BoundingBoxDataset(det_df, seq_info_dict = seq_info_dict, output_size=dataset_params['img_size'],
                            return_det_ids_and_frame=False)
    bb_loader = DataLoader(ds, batch_size=dataset_params['img_batch_size'], pin_memory=True, num_workers=6)
    cnn_model = cnn_model.eval()

    bb_imgs = []
    node_embeds = []
    reid_embeds = []
    with torch.no_grad():
        for bboxes in bb_loader:
            node_out, reid_out = cnn_model(bboxes.cuda())
            node_embeds.append(node_out.to(device))
            reid_embeds.append(reid_out.to(device))

            if return_imgs:
                bb_imgs.append(bboxes)

    node_embeds = torch.cat(node_embeds, dim=0)
    reid_embeds = torch.cat(reid_embeds, dim=0)

    return bb_imgs, node_embeds, reid_embeds

def load_precomputed_embeddings(det_df, seq_info_dict, embeddings_dir, use_cuda):
    """
    Given a sequence's detections, it loads from disk embeddings that have already been computed and stored for its
    detections
    Args:
        det_df: pd.DataFrame with detection coordinates
        seq_info_dict: dict with sequence meta info (we need frame dims)
        embeddings_dir: name of the directory where embeddings are stored

    Returns:
        torch.Tensor with shape (num_detects, embeddings_dim)

    """
    # Retrieve the embeddings we need from their corresponding locations
    embeddings_path = osp.join(seq_info_dict['seq_path'], 'processed_data', 'embeddings', seq_info_dict['det_file_name'],
                               embeddings_dir)
    #print("EMBEDDINGS PATH IS ", embeddings_path)
    frames_to_retrieve = sorted(det_df.frame.unique())
    embeddings_list = [torch.load(osp.join(embeddings_path, f"{frame_num}.pt")) for frame_num in frames_to_retrieve]
    embeddings = torch.cat(embeddings_list, dim=0)

    # First column in embeddings is the index. Drop the rows of those that are not present in det_df
    ixs_to_drop = list(set(embeddings[:, 0].int().numpy()) - set(det_df['detection_id']))
    embeddings = embeddings[~np.isin(embeddings[:, 0], ixs_to_drop)]  # Not so clean, but faster than a join
    assert_str = "Problems loading embeddings. Indices between query and stored embeddings do not match. BOTH SHOULD BE SORTED!"
    assert (embeddings[:, 0].numpy() == det_df['detection_id'].values).all(), assert_str

    embeddings = embeddings[:, 1:]  # Get rid of the detection index

    return embeddings.to(torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu"))
