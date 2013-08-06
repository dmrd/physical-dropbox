import cv2
import numpy as np
import math
import csv
import sys
from math import radians

from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def thresh(color_img):
    ''' Threshold the image so that the most intense pixels are white '''
    n = 0.35 # use top n% of pixels
    bw_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    flatrank = np.argsort(bw_img.ravel())
    thresh_index = flatrank[int(len(flatrank) * (1 - 0.01*n))]
    thresh_value = np.ravel(bw_img)[thresh_index]

    bw_img = cv2.threshold(bw_img, thresh_value, 255, cv2.THRESH_BINARY)[1]
    return bw_img


BUFFER = 5 # Ignore pixels within this distance of the edge

def line_coords(thresholded):
    '''
    Return [a list of (x,y)] tuples representing the middle white pixel of each
        line for which one exists
    '''
    pixels = []
    for y, line in enumerate(thresholded[BUFFER:-BUFFER]):
        white_pix = np.where(line[BUFFER:-BUFFER])
        if len(white_pix[0]):
            # Add x,y tuple
            pixels.append((white_pix[int(len(white_pix) / 2)] + BUFFER, y + BUFFER))
    return np.array(pixels)


def process_line(line_coords, angle, distance):
    '''
    Return [a list of (x,y,z)] tuples representing thepoint cloud calculated
        from line_coords and the given angle (in degrees)
    '''
    angle = radians(angle)
    coords = []
    for x, y in line_coords:
        x = x * math.cos(angle)
        z = -x * math.sin(angle)
        coords.append((x,y,z))
    return coords


def points_to_mesh(points, fname):
    '''write a mesh file equivalent of points, which is a numpy array of [x,y,z] points'''
    pass


def visualize_points(points):
    '''3d scatter plot for testing; takes a numpy array of [x,y,z] points'''
    fig = figure()
    ax = fig.gca(projection='3d')
    ax.plot(points[:,0],points[:,1],points[:,2],'o')
    plt.show()


class Processor:
    def __init__(self, laser_camera_distance = 1, laser_angle = 30.0, path=None):
        self.point_cloud = []
        self.distance = laser_camera_distance
        self.angle = laser_angle
        if path:
            self.load_cloud(path)


    def process_picture(self, picture, angle):
        ''' Takes picture and angle (in degrees).  Adds to point cloud '''
        thresholded = thresh(picture)                 # Do a hard threshold of the image
        pixels = line_coords(thresholded)             # Get line coords from image
        self.point_cloud.extend(process_line(pixels)) # Add new points to cloud


    def process_pictures(self, pictures):
        for picture in pictures:
            self.process_picture(picture)
            picture = preprocess(picture)
            pixels = extract_pixels(picture)
            self.point_cloud.extend(pixels)


    def load_cloud(self, path):
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for point in reader:
                self.point_cloud.append(tuple(int(p) for p in point))


    def save_cloud(self, path):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            for point in self.point_cloud:
                writer.writerow(point)


    def visualize(self):
        visualize_points(np.array(self.point_cloud))


if __name__=="__main__":
    proc = Processor()
    #img = cv2.imread(sys.argv[1])

    # test preprocess
    #img = thresh(img)
    #cv2.imshow('', img)
    #cv2.waitKey(0)

    points = 0.6 * np.random.standard_normal((200,3))
    visualize_points(points)
