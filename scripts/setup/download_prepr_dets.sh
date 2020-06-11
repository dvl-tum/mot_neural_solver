#!/usr/bin/env bash

DATA_PATH=$(python -c "from mot_neural_solver.path_cfg import DATA_PATH; print(DATA_PATH)")
wget -P $DATA_PATH https://vision.in.tum.de/webshare/u/brasoand/mot_neural_solver/prepr_det_files.zip
unzip -d $DATA_PATH $DATA_PATH/prepr_det_files.zip
rm $DATA_PATH/prepr_det_files.zip