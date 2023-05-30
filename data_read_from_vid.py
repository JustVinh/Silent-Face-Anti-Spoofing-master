import cv2
import time

label = 1
name = 69

DATA_PATH = '/home/vinhnt/work/DATN/FAS/data/mydata/Mydata-20230523T151458Z-001/Mydata'
REAL_DATA_PATH = '/home/vinhnt/work/DATN/FAS/data/mydata'

import os

directory = os.fsencode(DATA_PATH)
frame_count = 0
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".MOV"):
        file_path = os.path.join(DATA_PATH, filename)
        vid = cv2.VideoCapture(file_path)
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        while (vid.isOpened()):

            # Capture the video frame
            # by frame
            ret, frame = vid.read()
            if (frame is None): break
            frame_count += 1
            # Display the resulting frame
            if frame_count % (fps*1) == 0:
                # cv2.imshow('frame' + str(name), frame)
                cv2.imwrite(REAL_DATA_PATH + '/' + str(label) + '/' + str(name) + '.jpg', frame)
                name += 1
                # the 'q' button is set as the
                # quitting button you may use any
                # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        continue
    else:
        continue

vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
    # time.sleep(1)

