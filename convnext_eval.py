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
import torchvision.transforms

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


SAMPLE_IMAGE_PATH = "./images/sample/"
DATA_PATH = '/home/vinhnt/Downloads/dataset-20230605T134927Z-001/dataset/val'

# 因为安卓端APK获取的视频流宽高比为3:4,为了与之一致，所以将宽高比限制为3:4
def check_image(image):
    # height, width, channel = image.shape
    # if width/height != 3/4:
    #     print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
    #     return False
    # else:
    #     return True
    return True
def test_cam(model_1, image_cropper, session, transform, input_name):
    n_sample = 0
    n_0 = 0
    n_1 = 1

    n_true = 0
    n_fr = 0
    n_fa = 0

    for label in range(2):
        directory = os.fsencode(DATA_PATH + '/' + str(label))
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg"):
                n_sample += 1

                file_path = os.path.join(DATA_PATH + '/' + str(label), filename)

                frame = cv2.imread(file_path)
                pred = test_image(frame, model_1, image_cropper, session, transform, input_name)

                if pred == 2: pred = 0

                if label == pred:
                    n_true += 1

                if label == 0:
                    n_0 += 1
                    if pred == 1:
                        n_fa += 1
                else:
                    n_1 += 1
                    if pred == 0:
                        n_fr += 1

                # Display the resulting frame
                cv2.imshow('frame', frame)

                # the 'q' button is set as the
                # quitting button you may use any
                # desired button of your choice
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    print("ACC: ", n_true / n_sample)
    print("FAR: ", n_fa / n_0)
    print("FRR: ", n_fr / n_1)

    # After the loop release the cap object
    # Destroy all the windows
    cv2.destroyAllWindows()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def test_image(image, model_1, image_cropper, session, transform, input_name):

    image_cropper = CropImage()
    result = check_image(image)
    if result is False:
        return
    image_bbox = model_1.get_bbox(image)

    test_speed = 0
    # sum the prediction from single model's result

    # param = {
    #     "org_img": image,
    #     "bbox": image_bbox,
    #     "scale": 4,
    #     "out_w": 224,
    #     "out_h": 224,
    #     "crop": True,
    # }

    # face_img = image_cropper.crop(**param)

    #eval without preprocessing (transform images)
    color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #shape 480,720,3
    pil_image = Image.fromarray(color_coverted)

    # im = transform(pil_image).unsqueeze(0).cpu().numpy().astype(np.float32)
    # tran_img = cv2.resize(image, (224, 224))
    # tran_img = np.moveaxis(tran_img, -1, 0)
    pil_image = pil_image.resize((224,224))
    MEAN = 255 * np.array([0.4850, 0.4560, 0.4060])
    STD = 255 * np.array([0.2290, 0.2240, 0.2250])
    # img_pil = Image.open("ty.jpg")
    x = np.array(pil_image) #shape 224,224,3
    x = x.transpose(-1, 0, 1) #shape 3,224,224
    x = (x - MEAN[:, None, None]) / STD[:, None, None]

    im = torch.from_numpy(x).unsqueeze(0).cpu().numpy().astype(np.float32) #shape


    start = time.time()
    a = session.run(None, {input_name: im.astype(np.float32)})[0]
    test_speed += time.time()-start
    probabilities = softmax(a[0])
    label = probabilities.argmax()
    # draw result of prediction
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
    return label


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

    model_1.model.eval()
    model_1.model.cuda()
    #
    # model_2.model.eval()
    # model_2.model.cuda()

    providers = ['CUDAExecutionProvider']
    session = onnxruntime.InferenceSession('/home/vinhnt/work/DATN/FAS/projects/Silent-Face-Anti-Spoofing-master/resources/convnext_quantized/convnext.fp16.simplified.onnx',
                                           providers=providers)
    input_name = session.get_inputs()[0].name

    model = timm.create_model(
        "convnext_tiny_in22ft1k", pretrained=False, num_classes=2
    )
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg)

    test_cam(model_1, image_cropper, session, transform, input_name)
