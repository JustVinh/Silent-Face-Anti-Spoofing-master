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
import onnxruntime

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
from src.data_io import transform as trans
warnings.filterwarnings('ignore')


SAMPLE_IMAGE_PATH = "./images/sample/"


# 因为安卓端APK获取的视频流宽高比为3:4,为了与之一致，所以将宽高比限制为3:4
def check_image(image):
    # height, width, channel = image.shape
    # if width/height != 3/4:
    #     print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
    #     return False
    # else:
    #     return True
    return True
def test_cam(model_1, model_2, image_cropper, input_name):
    vid = cv2.VideoCapture(0)
    while (True):

        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        test_image(frame, model_1, model_2, image_cropper, input_name)
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

def test_image(image, model_1, model_2, image_cropper, input_name):

    image_cropper = CropImage()
    result = check_image(image)
    if result is False:
        return
    image_bbox = model_1.get_bbox(image)
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
    a = session.run(None, {input_name: img.astype(np.float32)})[0]
    test_speed += time.time() - start
    probabilities = softmax(a[0])
    label = probabilities.argmax()

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
        image,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    cv2.putText(
        image,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)

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
    model_2 = AntiSpoofPredict(args.device_id)

    model_1.custom_load_model('/home/vinhnt/work/DATN/FAS/projects/Silent-Face-Anti-Spoofing-master/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth')
    # model_2.custom_load_model('/home/vinhnt/work/DATN/FAS/projects/Silent-Face-Anti-Spoofing-master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth')

    providers = ['CUDAExecutionProvider']
    session = onnxruntime.InferenceSession(
        '/home/vinhnt/work/DATN/FAS/projects/Silent-Face-Anti-Spoofing-master/resources/onnx_converted/mobilenet_convnext.onnx',
        providers=providers)

    input_name = session.get_inputs()[0].name
    image_cropper = CropImage()

    test_cam(model_1, model_2, image_cropper, input_name)