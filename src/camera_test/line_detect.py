import time
import cv2


im_gray = cv2.imread('Picture 5.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
cv2.imwrite('im_gray_5.png', im_gray)

# this code picks out the brightest points in the image
# we make a normalized histogram and pick off the highest bin
im_normalized = cv2.equalizeHist(im_gray)
thresh = 254
cv2.imwrite('im_thresh_5.png', cv2.threshold(im_normalized, thresh, 255, cv2.THRESH_BINARY)[1])

# this code simply sets a threshold and throws away pixels below it
#thresh = 150
#im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
#cv2.imwrite('bw_image5.png', im_bw)
