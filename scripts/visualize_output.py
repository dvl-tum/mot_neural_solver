import os
import random
import argparse

import cv2

from mot_neural_solver.path_cfg import OUTPUT_PATH, DATA_PATH


COLOR_DICT = {}


def get_detections(file_lines):
    detections = {}
    for line in file_lines:
        frame_id, instance_id, top_x, top_y, w, h, _, _, _, _ = list(map(float, line.strip().split(",")))
        frame_id, instance_id, top_x, top_y, w, h = int(frame_id), int(instance_id), int(top_x), int(top_y), int(w), int(h)
        if not(frame_id in detections):
            detections[frame_id] = {}        
        detections[frame_id][instance_id] = (top_x, top_y, w, h)
    
    return detections


def main(args):
    out_txt_file = os.path.join(OUTPUT_PATH, args.input_file)
    seq_name = out_txt_file.split("/")[-1].replace(".txt","")
    with open(out_txt_file, "r") as file:
        lines = file.readlines()
    detections = get_detections(lines)
    
    for frame_num in detections:
        img_path = os.path.join(DATA_PATH, args.image_folder, f'{frame_num:06}.jpg')
        img = cv2.imread(img_path)
        frame_detections = detections[frame_num]
        
        for instance_id in frame_detections:
            if not (instance_id in COLOR_DICT):
                COLOR_DICT[instance_id] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            color = COLOR_DICT[instance_id]

            top_x, top_y, w, h = frame_detections[instance_id]
            img = cv2.rectangle(img, (top_x, top_y), (top_x+w, top_y+h), color, 2) 
        
        cv2.imshow("Tracking", img)
        cv2.waitKey(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='MOT Neural Solver Visualizer')
    parser.add_argument('--input_file', type=str, help="Path to experiment file relative to the OUTPUT_PATH")
    parser.add_argument('--image_folder', type=str, help="Path to image folder relative to the DATA_PATH")
    args = parser.parse_args()

    main(args) 