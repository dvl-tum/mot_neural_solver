"""
All the code here is exctracted with minor modifications from this Colab Notebook:
https://colab.research.google.com/drive/1_arNo-81SnqfbdtAhb3TBSU5H0JXQ0_1
which was made public in the following repo: https://github.com/phil-bergmann/tracking_wo_bnw
It corresponds to the paper 'Tracking without bells and whistles' (ICCV2019), by Bergmann, Meinhardt and Leal-Taixé.
"""
import os
import os.path as osp

import torch

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from obj_detect.engine import train_one_epoch
import obj_detect.utils as utils
import obj_detect.transforms as T
from obj_detect.dataset import MOT17ObjDetect

from mot_neural_solver.path_cfg import OUTPUT_PATH, DATA_PATH

from sacred import Experiment

ex = Experiment()
ex.add_config('configs/mot20/obj_detect_cfg.yaml')


def get_detection_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.nms_thresh = 0.3

    return model

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


@ex.automain
def main(_config):

    torch.manual_seed(1)

    # use our dataset and defined transformations
    dataset_path = osp.join(DATA_PATH, _config['dataset_dir'])
    dataset = MOT17ObjDetect(dataset_path, get_transform(train=True))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=_config['train_params']['batch_size'], shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # get the model using our helper function
    model = get_detection_model(dataset.num_classes)
    model.to(device)

    model_state_dict = torch.load(osp.join(OUTPUT_PATH, _config['train_params']['start_ckpt']))
    model.load_state_dict(model_state_dict)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, **_config['optimizer_params'])


    os.makedirs(osp.join(OUTPUT_PATH, 'trained_models/frcnn'),  exist_ok=True)

    for epoch in range(1, _config['train_params']['num_epochs']+ 1):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        if not _config['train_params']['save_only_last_ckpt']:
            torch.save(model.state_dict(), osp.join(OUTPUT_PATH, 'trained_models/frcnn/mot20', f"mot20_frcnn_epoch_{epoch}.pt.tar"))

    if _config['train_params']['save_only_last_ckpt']:
        torch.save(model.state_dict(), osp.join(OUTPUT_PATH, 'trained_models/frcnn', f"mot20_frcnn_epoch_{epoch}.pt.tar"))

