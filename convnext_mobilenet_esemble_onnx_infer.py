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
from PIL import Image
import torch
import onnxruntime
import timm
import uuid
import multiprocessing
import concurrent.futures
from src.data_io import transform as trans

# import torchvision.transforms as transforms

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


SAMPLE_IMAGE_PATH = "./images/sample/"
FALSE_NEGATIVE_IMAGE_PATH = '/home/vinhnt/Downloads/create_data/false_negative/'
FALSE_POSITIVE_IMAGE_PATH= '/home/vinhnt/Downloads/create_data/new1/fake/fake_image/'


# 因为安卓端APK获取的视频流宽高比为3:4,为了与之一致，所以将宽高比限制为3:4
def check_image(image):
    # height, width, channel = image.shape
    # if width/height != 3/4:
    #     print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
    #     return False
    # else:
    #     return True
    return True
def test_cam(model_1, image_cropper, session, transform, input_name, session_mbfnet, input_name_mbfnet):
    vid = cv2.VideoCapture(0)
    while (True):

        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        frame_clone = frame.copy()
        test_speed = 0
        start = time.time()
        # #ver1
        # label2, image_bbox = test_image_mobilenet(frame_clone, model_1, image_cropper)
        # if(label2 == 1):
        #     label1, image_bbox = test_image(frame, model_1, image_cropper, session, transform, input_name)
        # else:
        #     label2 = 0
        image_bbox = model_1.get_bbox(frame)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit function1 to the executor
            future1 = executor.submit(test_image_mobilenet, frame_clone, model_1, image_bbox, session_mbfnet, input_name_mbfnet)

            # Submit function2 to the executor
            future2 = executor.submit(test_image, frame, model_1, image_bbox, session, transform, input_name)

            # Retrieve the returned values from both futures
            label2, image_bbox = future1.result()
            label1, image_bbox= future2.result()

        test_speed += time.time() - start
        if(label1 != 1) or (label2 != 1):
            label = 0
        else:
            label = 1

        if label == 1:
            print("Image '{}' is Real Face. Score: {:.2f}.".format("image_name", 0))
            result_text = "RealFace Score: {:.2f}".format(0)
            color = (255, 0, 0)
        else:
            print("Image '{}' is Fake Face. Score: {:.2f}.".format("image_name", 0))
            result_text = "FakeFace Score: {:.2f}".format(0)
            color = (0, 0, 255)
        print("Prediction cost {:.2f} s".format(test_speed))
        cv2.rectangle(
            frame,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 2)
        cv2.putText(
            frame,
            result_text,
            (image_bbox[0], image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX, 0.5 * frame.shape[0] / 1024, color)
        # Display the resulting frame
        cv2.imshow('frame', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def test_image(image, model_1, image_bbox, session, transform, input_name):
    image_clone = image.copy()
    image_cropper = CropImage()
    # result = check_image(image)
    # if result is False:
    #     return

    test_speed = 0
    # sum the prediction from single model's result

    param = {
        "org_img": image,
        "bbox": image_bbox,
        "scale": 2.7,
        "out_w": 224,
        "out_h": 224,
        "crop": True,
    }

    face_img = image_cropper.crop(**param)
    color_coverted = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_coverted)

    im = transform(pil_image).unsqueeze(0).cpu().numpy().astype(np.float32)

    start = time.time()
    a = session.run(None, {input_name: im.astype(np.float32)})[0]
    test_speed += time.time()-start
    probabilities = softmax(a[0])
    label = probabilities.argmax()
    # draw result of prediction
    return label, image_bbox
    # # # # #todo: comment this if not necessary
    # if label == 1:
    #     cv2.imwrite(FALSE_POSITIVE_IMAGE_PATH + str(uuid.uuid4()) + '.jpg', image_clone)

    # format_ = os.path.splitext(image_name)[-1]
    # result_image_name = image_name.replace(format_, "_result" + format_)
    # cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)
    # cv2.imshow("Result", image)

def test_image_mobilenet(image, model_1, image_bbox, session_mbfnet, input_name_mbfnet):
    result = check_image(image)
    if result is False:
        return
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result

    h_input, w_input, model_type, scale = parse_model_name('4_0_0_80x80_MiniFASNetV1SE.pth')

    param = {
        "org_img": image,
        "bbox": image_bbox,
        "scale": scale,
        "out_w": w_input,
        "out_h": h_input,
        "crop": True,
    }
    if scale is None:
        param["crop"] = False
    img = image_cropper.crop(**param)

    test_transform = trans.Compose([
        trans.ToTensor(),
    ])
    img = test_transform(img)
    img = img.unsqueeze(0).cpu().numpy().astype(np.float32)

    start = time.time()
    a = session_mbfnet.run(None, {input_name_mbfnet: img.astype(np.float32)})[0]
    test_speed += time.time() - start
    probabilities = softmax(a[0])
    label = probabilities.argmax()

    return label, image_bbox

    # format_ = os.path.splitext(image_name)[-1]
    # result_image_name = image_name.replace(format_, "_result" + format_)
    # cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)
    # cv2.imshow("Result", image)

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
    image_cropper = CropImage()

    providers = ['CUDAExecutionProvider']
    session = onnxruntime.InferenceSession('/home/vinhnt/work/DATN/FAS/projects/Silent-Face-Anti-Spoofing-master/resources/convnext_quantized/convnext.fp16.simplified.ver2.onnx',
                                           providers=providers)
    input_name = session.get_inputs()[0].name

    session_mbfnet = onnxruntime.InferenceSession(
        '/home/vinhnt/work/DATN/FAS/projects/Silent-Face-Anti-Spoofing-master/resources/onnx_converted/mobilenet_convnext.onnx',
        providers=providers)

    input_name_mbfnet = session.get_inputs()[0].name

    model = timm.create_model(
        "convnext_tiny_in22ft1k", pretrained=False, num_classes=2
    )
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg)

    test_cam(model_1, image_cropper, session, transform, input_name, session_mbfnet, input_name_mbfnet)
