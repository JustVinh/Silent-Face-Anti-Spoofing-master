import cv2
from PIL import Image

# start streaming video from webcam
video_stream()
# label for video
label_html = 'Capturing...'
# initialze bounding box to empty
bbox = ''
count = 0
while True:
    js_reply = video_frame(label_html, bbox)
    if not js_reply:
        break

    # convert JS response to OpenCV Image
    img = js_to_image(js_reply["img"])

    # create transparent overlay for bounding box
    bbox_array = np.zeros([480,640,4], dtype=np.uint8)

    img = cv2.imread('/content/sample_data/reall.png')
    image_cropper = CropImage()

    detector = Detection()
    image_bbox = detector.get_bbox(img)

    param = {
        "org_img": img,
        "bbox": image_bbox,
        "scale": 4,
        "out_w": 224,
        "out_h": 224,
        "crop": True,
    }

    face_img = image_cropper.crop(**param)
    color_coverted = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_coverted)

    start_time = time.time()
    image_tensor = transform(pil_image)
    output = model(image_tensor.unsqueeze(0))
    probabilities = torch.nn.functional.softmax(output[0], dim=0).cpu().detach().numpy()
    # print("--- %s seconds ---" % (time.time() - start_time))
    res = probabilities.argmax()

    if res == 1 :
      color = (255, 0, 0)
    else:
      color = (0, 0, 255)

    bbox_array = cv2.rectangle(
        img,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)

    bbox_array[:,:,3] = (bbox_array.max(axis = 2) > 0 ).astype(int) * 255
    # convert overlay of bbox into bytes
    bbox_bytes = bbox_to_bytes(bbox_array)
    # update bbox so next frame gets new overlay
    bbox = bbox_bytes