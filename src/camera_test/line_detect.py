import time
import cv2





# select the connected webcam as opposed to the integrated webcam
camera_port = 1

# throw away initial frames while camera adjusts to light
ramp_frames = 30

# intitialize camera
camera = cv2.VideoCapture(camera_port)


# capture a single image
def get_image():
    retval, im = camera.read()
    return im

# throw some frames away
for i in xrange(ramp_frames):
    temp = get_image()

camera_capture = get_image()
file = "test.png"
cv2.imwrite(file, camera_capture)

del(camera)

