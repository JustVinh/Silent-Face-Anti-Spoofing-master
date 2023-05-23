import cv2
import time

label = 1
name = 0

DATA_PATH = '/home/vinhnt/work/DATN/FAS/data/mydata'

# define a video capture object
vid = cv2.VideoCapture(0)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imwrite(DATA_PATH + '/' + str(label) + '/' + str(name) + '.jpg', frame)
    time.sleep(1)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
    # time.sleep(1)

