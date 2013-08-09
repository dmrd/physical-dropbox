import cv2
import numpy as np
import os
import math
import csv
from math import radians

import pylab

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay

import util


#CENTER = 0.51  # current center
#CENTER = 0.53  # for old scans
#CENTER = 0.45  # for old scans
L_CENTER = 0.53
R_CENTER = 0.45
#CENTER = -0.53  # for old scans
HARD_THRESHOLD = 40       # always ignore pixels below this value
BACK_WALL_MARGIN = 15
LINE_COORDS_BUFFER = 5    # Ignore pixels within this distance of the edge
PERCENT_TOP_PIXELS = 0.2  # max percent of brightness-ranked pixels to select


def ply_write(path, cloud):
    print(path + ": " + str(len(cloud)))
    with open(path + '.ply', 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex {0}\n".format(len(cloud)))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("element face 0\n")
        f.write("element edge 0\n")
        f.write("end_header\n")
        for point in cloud:
            f.write("{0} {1} {2}\n".format(point[0], point[1], point[2]))


def find_back_wall(calibration_img):
    '''pick x-coordinate for laser line falling on back wall
       (we ignore any light to the right of this)'''
    x = line_coords(thresh(calibration_img, 1, 10))
    if len(x) == 0:
        return calibration_img.shape[1]
    x = x[:, 0]
    return np.bincount(x).argmax() - BACK_WALL_MARGIN  # mode minus margin


def thresh(color_img, percent=PERCENT_TOP_PIXELS,
           hard_threshold=HARD_THRESHOLD):
    ''' Threshold the image so that the most intense pixels are white '''
    percent *= 0.01
    bw_img = cv2.split(color_img)[1]  # just extract green channel
    #bw_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    flatrank = np.argsort(bw_img.ravel())
    thresh_index = flatrank[int(len(flatrank) * (1 - percent))]
    thresh_value = min(np.ravel(bw_img)[thresh_index], 254)
    bw_img = cv2.threshold(bw_img, max(thresh_value, HARD_THRESHOLD),
                           255, cv2.THRESH_BINARY)[1]

    #normalized = cv2.equalizeHist(bw_img)
    #bw_img = cv2.threshold(normalized, 254, 255, cv2.THRESH_BINARY)[1]

    return bw_img


def line_coords(thresholded, x_center=None):
    '''
    Return [a list of (x,y)] tuples representing middle white pixel per line
    If x_center given, transforms coordinates so that
        the axis of rotation is x=0.
    '''
    pixels = []
    for y, line in enumerate(thresholded[LINE_COORDS_BUFFER:-LINE_COORDS_BUFFER]):
        white_pix = np.nonzero(line[LINE_COORDS_BUFFER:-LINE_COORDS_BUFFER])[0]
        if len(white_pix):
            # Add x,y tuple
            pixels.append((
                white_pix[int(len(white_pix) / 2)] + LINE_COORDS_BUFFER,
                y + LINE_COORDS_BUFFER))
    if x_center is not None:
        pixels = [(x - x_center, -y) for (x, y) in pixels]
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
        nz = y  # swap y,z for delaunay tetrahedralization
        ny = -x * math.sin(angle)
        coords.append((nx, ny, nz))
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


def visualize_mesh(points):
    '''
    generate and visualize a mesh
    delaunay triangulation on numpy array of [x y z] points
    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2])
    plt.show()


def resize_image(image, new_x=None):
    '''return scaled down image (aspect ratio is preserved)'''
    new_x = new_x or 600

    x, y = image.shape[1], image.shape[0]
    new_y = y * new_x / x
    return cv2.resize(image, (new_x, new_y))


class Processor:
    def __init__(self, calibration_img, laser_camera_distance=1, laser_angle=30.0, path=None):
        self.back_wall_x = find_back_wall(calibration_img)
        print "back wall x: %d pixels" % self.back_wall_x
        self.point_cloud_l = []
        self.point_cloud_r = []
        self.distance = laser_camera_distance
        self.angle = laser_angle
        if path:
            self.load_cloud(path)

    def process_picture(self, picture, angle, right=False):
        ''' Takes picture and angle (in degrees).  Adds to point cloud '''
        center_val = R_CENTER if right else L_CENTER
        x_center = picture.shape[1] * center_val
        thresholded = thresh(picture)                   # Do a hard threshold of the image
        #cv2.imwrite('thresh_%d.jpg'%angle, thresholded) # for debugging
        pixels = line_coords(thresholded, x_center)     # Get line coords from image

        # filter out any pixels to right of back wall line
        pixels = filter(lambda p: p[0] < self.back_wall_x - x_center, pixels)

        # add to point cloud
        if right:
            points = process_line(pixels, angle, self.distance)
            self.point_cloud_r.extend(points)
        else:
            points = process_line(pixels, angle + 120, self.distance)
            self.point_cloud_l.extend(points)
        return thresholded

    def process_pictures(self, pictures, prefix=None):
        # process pics
        if filter(lambda x: x is None, pictures):
            raise Exception('some pictures are null')

        processed = []
        for i, picture in enumerate(pictures):
            #picture = resize_image(picture) #if we turn resize back on
            #  don't forget to adjust back_wall_x
            processed.append(self.process_picture(picture, i * 360.0 / len(pictures)))
            print "processed %d; angle %f" % (i, i * 360.0 / len(pictures))

        if prefix:
            util.save_images(processed,
                             prefix,
                             dir_name=os.path.join("img", prefix, "processed"))

        # save to wrl
        #points_to_mesh(self.point_cloud, 'OMG.wrl')

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

    def save_ply(self, path, left=False, right=False, dual=False):
        if left:
            ply_write(path + '_l', self.point_cloud_l)
        if right:
            ply_write(path + '_r', self.point_cloud_r)
        if dual:
            ply_write(path + '_d', self.point_cloud_l + self.point_cloud_r)

    def visualize(self):
        visualize_points(np.array(self.point_cloud))
        #visualize_mesh(np.array(self.point_cloud))

    def process_continuous(self, images, num_rotations, prefix=None, right=False):
        processed = []
        for i, img in enumerate(images):
            angle = 360.0 * i * num_rotations / len(images)
            processed.append(self.process_picture(img, angle, right))
            print("Processing image {0} of {1}".format(i + 1, len(images)))

        if prefix:
            raw_dir = 'processed_r' if right else 'processed_l'
            util.save_images(processed,
                             prefix,
                             dir_name=os.path.join("img", prefix, raw_dir))


    def process_scan(self, rotations, prefix,
                     calibration_name="calibration/calibration.jpg",
                     right=False):

        images = []

        raw_dir = 'raw_r' if right else 'raw_l'
        if right:
            path = os.path.join('img', prefix, raw_dir)
        else:
            path = os.path.join('img', prefix, raw_dir)

        for f in os.listdir(path):
            f = os.path.join(path, f)
            images.append(cv2.imread(f))

        #proc.process_pictures(images)
        self.process_continuous(images, rotations, prefix=prefix, right=right)
        #proc.visualize()
