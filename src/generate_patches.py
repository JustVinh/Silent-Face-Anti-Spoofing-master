# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm
"""
Create patch from original input image by using bbox coordinate
"""

import cv2
import numpy as np
import os

from src.anti_spoof_predict import Detection


class CropImage:
    @staticmethod
    def _get_new_box(src_w, src_h, bbox, scale):
        x = bbox[0]
        y = bbox[1]
        box_w = bbox[2]
        box_h = bbox[3]

        scale = min((src_h-1)/box_h, min((src_w-1)/box_w, scale))

        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w/2+x, box_h/2+y

        left_top_x = center_x-new_width/2
        left_top_y = center_y-new_height/2
        right_bottom_x = center_x+new_width/2
        right_bottom_y = center_y+new_height/2

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0

        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0

        if right_bottom_x > src_w-1:
            left_top_x -= right_bottom_x-src_w+1
            right_bottom_x = src_w-1

        if right_bottom_y > src_h-1:
            left_top_y -= right_bottom_y-src_h+1
            right_bottom_y = src_h-1

        return int(left_top_x), int(left_top_y),\
               int(right_bottom_x), int(right_bottom_y)

    def crop(self, org_img, bbox, scale, out_w, out_h, crop=True):

        if not crop:
            dst_img = cv2.resize(org_img, (out_w, out_h))
        else:
            src_h, src_w, _ = np.shape(org_img)
            left_top_x, left_top_y, \
                right_bottom_x, right_bottom_y = self._get_new_box(src_w, src_h, bbox, scale)

            img = org_img[left_top_y: right_bottom_y+1,
                          left_top_x: right_bottom_x+1]
            dst_img = cv2.resize(img, (out_w, out_h))
        return dst_img

if __name__ == "__main__":
    DATA_PATH = '/home/vinhnt/work/DATN/FAS/data/mydata'
    PATCH_PATH = '/home/vinhnt/work/DATN/FAS/projects/Silent-Face-Anti-Spoofing-master/datasets/rgb_image/2.7_224x224'
    image_cropper = CropImage()
    detector = Detection()
    no_img = 0


    for label in range(2):
        directory = os.fsencode(DATA_PATH + '/' + str(label))
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg"):
                file_path = os.path.join(DATA_PATH + '/' + str(label), filename)
                img = cv2.imread(file_path)

                img_bbox = detector.get_bbox(img)
                new_img = image_cropper.crop(img, img_bbox, 4, 224, 224, True)

                my_label = 1
                if label == 0: my_label = 0


                save_path = os.path.join(PATCH_PATH + '/' + str(my_label), filename)
                cv2.imwrite(save_path, new_img)
                no_img += 1
                print("Done ", no_img)