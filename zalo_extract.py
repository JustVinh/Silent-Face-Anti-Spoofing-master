import os
import cv2
import numpy as np
import time
import json
import csv

if __name__ == '__main__':
    VIDEO_PATH = '/home/vinhnt/work/DATN/FAS/data/zalo/zalo_video/archive/train/train/videos'
    CSV_PATH = '/home/vinhnt/work/DATN/FAS/data/zalo/zalo_video/archive/train/train/label.csv'
    SAVE_PATH = '/home/vinhnt/work/DATN/FAS/data/zalo/zalo_image'
    with open(CSV_PATH, 'r') as file:
        csvreader = csv.reader(file)
        skip_first_line = True
        for row in csvreader:
            if skip_first_line :
                skip_first_line = False
                continue
            # video_path = os.path.join(VIDEO_PATH, row[1])
            video_path = os.path.join(VIDEO_PATH, row[0])
            base_name = row[0].replace(".mp4", "")
            cap = cv2.VideoCapture(video_path)

            if (cap.isOpened() == False):
                print("Error opening video stream or file ", video_path)
            frame_count = 0
            count = 0
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    if frame_count % 2 == 0:
                        frame_path = os.path.join(SAVE_PATH, row[1])
                        # frame_path = os.path.join(frame_path, base_name)
                        os.makedirs(frame_path, exist_ok=True)
                        frame_path = os.path.join(frame_path, base_name +'_' + str(frame_count) + ".png")
                        cv2.imwrite(frame_path, frame)
                        count += 1
                    frame_count += 1
                else:
                    break
                print("Processed images: " + str(count))

            cap.release()