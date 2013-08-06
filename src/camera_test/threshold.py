import cv2
import numpy as np
import os

def thresh(img_name):
    img = cv2.imread(img_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    # sort bar by order of values in foo
    flatrank = np.argsort(img.ravel())
    # get the top n% of pixels
    n = 0.35
    thresh_index = flatrank[int(len(flatrank) * (1 - 0.01*n))]
    thresh_value = np.ravel(img)[thresh_index]
    print thresh_value

    img = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY)[1]
    return img

if __name__=="__main__":
    img_name = 'Picture 4.jpg'
    img = thresh(img_name)
    cv2.imshow("%s_thresh.png" % os.path.splitext(img_name)[0], img)
    #cv2.waitKey(0)

