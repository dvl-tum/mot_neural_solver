#!/usr/bin/env bash

# Download MOT20 dataset
DATA_PATH=$(python -c "from mot_neural_solver.path_cfg import DATA_PATH; print(DATA_PATH)")
wget -P $DATA_PATH https://motchallenge.net/data/MOT20.zip
unzip -d $DATA_PATH/MOT17Labels $DATA_PATH/MOT20.zip
rm $DATA_PATH/MOT20.zip

# Download tracktor preprocessed detections for MOT20
wget -P $DATA_PATH https://vision.in.tum.de/webshare/u/brasoand/mot_neural_solver/mot20_prepr_det_files.zip
unzip -d $DATA_PATH $DATA_PATH/mot20_prepr_det_files.zip
rm $DATA_PATH/mot20_prepr_det_files.zip

# Download MOT20 detection (tracktor) and tracking (MOTNeuralSolver) models
OUTPUT_PATH=$(python -c "from mot_neural_solver.path_cfg import OUTPUT_PATH; print(OUTPUT_PATH)")
wget -P $OUTPUT_PATH/trained_models/frcnn https://vision.in.tum.de/webshare/u/brasoand/mot_neural_solver/mot20_frcnn_epoch_27.pt.tar
wget -P $OUTPUT_PATH/trained_models/graph_nets https://vision.in.tum.de/webshare/u/brasoand/mot_neural_solver/mot20_mot_mpnet_epoch_021.ckpt


