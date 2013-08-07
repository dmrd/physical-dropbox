import cv2
import numpy as np
import math
import csv
import sys
from math import radians

import pylab

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.mlab import griddata

from mpl_toolkits.mplot3d import Axes3D

def thresh(color_img):
    ''' Threshold the image so that the most intense pixels are white '''
    n = 0.35  # use top n% of pixels
    bw_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    flatrank = np.argsort(bw_img.ravel())
    thresh_index = flatrank[int(len(flatrank) * (1 - 0.01*n))]
    thresh_value = np.ravel(bw_img)[thresh_index]

    bw_img = cv2.threshold(bw_img, thresh_value, 255, cv2.THRESH_BINARY)[1]
    return bw_img


BUFFER = 5  # Ignore pixels within this distance of the edge


def line_coords(thresholded, x_center):
    '''
    Return [a list of (x,y)] tuples representing the middle white pixel of each
        line for which one exists
    '''
    pixels = []
    for y, line in enumerate(thresholded[BUFFER:-BUFFER]):
        white_pix = np.nonzero(line[BUFFER:-BUFFER])[0]
        if len(white_pix):
            # Add x,y tuple
            pixels.append((
                white_pix[int(len(white_pix) / 2)] + BUFFER - x_center,
                -(y + BUFFER)
                ))
    return np.array(pixels)


def process_line(line_coords, angle, distance):
    '''
    Return [a list of (x,y,z)] tuples representing the point cloud calculated
        from line_coords and the given angle (in degrees)
    '''
    angle = radians(angle)
    coords = []
    for x, y in line_coords:
        nx = x * math.cos(angle)
        nz = y #swap y,z for delaunay tetrahedralization
        ny = -x * math.sin(angle)
        coords.append((nx,ny,nz))
    return coords


def points_to_mesh(points, fname):
    '''write a mesh file equivalent of a numpy array of [x,y,z] points'''
    pass


def visualize_points(points):
    '''3d scatter plot for testing; takes a numpy array of [x y z] points'''
    fig = pylab.figure()
    ax = fig.gca(projection='3d')
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'o')
    plt.show()


# http://stackoverflow.com/questions/4363857/matplotlib-color-in-3d-plotting-from-an-x-y-z-data-set-without-using-contour
def visualize_mesh(points):
    '''
    generate and visualize a mesh
    delaunay triangulation on numpy array of [x y z] points
    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    data = points
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    xi = np.linspace(min(x), max(x))
    yi = np.linspace(min(y), max(y))

    X, Y = np.meshgrid(xi, yi)
    Z = griddata(x, y, z, xi, yi)

    surf = ax.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap=cm.jet,
                           linewidth=1, antialiased=True)

    ax.set_zlim3d(np.min(Z), np.max(Z))
    fig.colorbar(surf)

    plt.show()


def resize_image(image, new_x=None):
    '''return scaled down image (aspect ratio is preserved)'''
    new_x = new_x or 600
    x, y = image.shape[1], image.shape[0]
    new_y = y * new_x/x
    return cv2.resize(image, (new_x, new_y))


class Processor:
    def __init__(self, laser_camera_distance=1, laser_angle=30.0, path=None):
        self.point_cloud = []
        self.distance = laser_camera_distance
        self.angle = laser_angle
        if path:
            self.load_cloud(path)

    def process_picture(self, picture, angle):
        ''' Takes picture and angle (in degrees).  Adds to point cloud '''
        x_center = picture.shape[1]/2 # for now, let's say axis of rotation is the
                                      # center of the image
        thresholded = thresh(picture)                   # Do a hard threshold of the image
        pixels = line_coords(thresholded, x_center)     # Get line coords from image
        self.point_cloud.extend(
            process_line(pixels, angle, self.distance)) # Add new points to cloud

    def process_pictures(self, pictures):
        if filter(lambda x:x==None, pictures):
            raise Exception('some pictures are null')
        for i, picture in enumerate(pictures):
            picture = resize_image(picture)
            self.process_picture(picture, i * 360.0 / len(pictures))
            print "processed %d; angle %f" % (i, i*360.0/len(pictures))

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
        #visualize_points(np.array(self.point_cloud))
        visualize_mesh(np.array(self.point_cloud))


if __name__ == "__main__":
    proc = Processor()
    img = cv2.imread(sys.argv[1])

    # test same image, many revolutions
    proc.process_pictures([img]*20)
    proc.visualize()

    # test preprocess
    #img = thresh(img)
    #cv2.imshow('', img)
    #cv2.waitKey(0)

    # test visualize points / mesh
    #points = 0.6 * np.random.standard_normal((200,3))
    #visualize_points(points)
    #visualize_mesh(points)
