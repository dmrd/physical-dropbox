import cv2
import numpy as np

def thresh(color_img):
    ''' Threshold the image so that the most intense pixels are white '''
    bw_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    # sort bar by order of values in foo
    flatrank = np.argsort(bw_img.ravel())
    # get the top n% of pixels
    n = 0.35
    thresh_index = flatrank[int(len(flatrank) * (1 - 0.01*n))]
    thresh_value = np.ravel(bw_img)[thresh_index]

    bw_img = cv2.threshold(bw_img, thresh_value, 255, cv2.THRESH_BINARY)[1]
    return bw_img


def line_coords(thresholded):
    '''
    Return [a list of (x,y)] tuples representing the middle white pixel of each
        line for which one exists
    '''
    pixels = []
    for y, line in enumerate(thresholded):
        white_pix = []
        for x,pixel in enumerate(line):
            if pixel == 0:
                white_pix.append(x)
        if white_pix:
            # Add x,y tuple
            pixels.append((white_pix[int(len(white_pix) / 2)], y))
    return pixels


def process_line(line_coords, angle):
    '''
    Return [a list of (x,y,z)] tuples representing the point cloud calculated
        from line_coords and the given angle
    '''
    pass


class Processor:
    def __init__(self):
        self.point_cloud = []


    def process_picture(self, picture, angle):
        ''' Takes picture and angle (in degrees).  Adds to point cloud '''
        thresholded = thresh(picture)                 # Do a hard threshold of the image
        pixels = line_coords(thresholded)             # Get line coords from image
        self.point_cloud.extend(process_line(pixels)) # Add new points to cloud


    def process_pictures(self, pictures):
        for picture in pictures:
            self.process_picture(picture)

