# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time
import torch

from src.anti_spoof_predict import AntiSpoofPredict
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--image_name",
        type=str,
        default="image_F1.jpg",
        help="image used to test")
    args = parser.parse_args()

    model_1 = AntiSpoofPredict(args.device_id)
    # model_2 = AntiSpoofPredict(args.device_id)

    model_1.custom_load_model('/home/vinhnt/work/DATN/FAS/projects/Silent-Face-Anti-Spoofing-master/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth')
    # model_2.custom_load_model('/home/vinhnt/work/DATN/FAS/projects/Silent-Face-Anti-Spoofing-master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth')
    model_1.model.cuda()
    dummy_input = torch.randn(1, 3, 80, 80).cuda()
    input_names = ['input']
    output_names = ['output']
    dynamic_axes = {'input': {0: 'batch'},
                    'output': {0: 'batch'}}
    path_onnx = f'/home/vinhnt/work/DATN/FAS/projects/Silent-Face-Anti-Spoofing-master/resources/onnx_converted/mobilenet_convnext.onnx'
    torch.onnx.export(model_1.model, dummy_input, path_onnx, verbose=False, input_names=input_names,
                      output_names=output_names, opset_version=17,
                      export_params=True, do_constant_folding=True,
                      dynamic_axes=dynamic_axes)
