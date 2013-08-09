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
CENTER = 0.53  # for old scans
HARD_THRESHOLD = 40       # always ignore pixels below this value
BACK_WALL_MARGIN = 15
LINE_COORDS_BUFFER = 5
PERCENT_TOP_PIXELS = 0.2  # max percent of brightness-ranked pixels to select


def ignore_half(img, right=True):
    '''return a copy of img with part of the image <right> of axis blacked out'''
    x_center = int(img.shape[1] * CENTER)
    if right:
        cv2.rectangle(img, (x_center,0), (img.shape[1],img.shape[0]), 0, thickness=cv2.cv.CV_FILLED)
    else:
        cv2.rectangle(img, (0,0), (x_center,img.shape[0]), 0, thickness=cv2.cv.CV_FILLED)
    return img


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

def process_line(line_coords, angle, distance, right=False):
    '''
    Return [a list of (x,y,z)] tuples representing the point cloud calculated
        from line_coords and the given angle (in degrees)
    '''
    angle = radians(angle)
    coords = []
    if right:
        x = -x
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
        self.find_back_wall(calibration_img)
        #print "back wall: left=%d, right=%d" % (self.wall_left, self.wall_right)
        print "back wall: %d" % self.find_back_wall(calibration_img)
        self.point_cloud = []
        self.distance = laser_camera_distance
        self.angle = laser_angle
        if path:
            self.load_cloud(path)

    def find_back_wall(self, calibration_img):
        '''pick x-coordinate for laser lines falling on back wall
           (we ignore any light outside of this)'''
        left_laser_img = ignore_half(calibration_img, True) #right blacked out
        right_laser_img = ignore_half(calibration_img, False) #left blacked out
   
        x = line_coords(thresh(calibration_img, 1, 10))
        if len(x) == 0:
            return calibration_img.shape[1]
        x = x[:, 0]
        return np.bincount(x).argmax() - BACK_WALL_MARGIN  # mode minus margin
 
#        left = line_coords(thresh(left_laser_img, 1, 10))
#        if len(left) == 0:
#            left = 0
#        else:
#            left = np.bincount( left[:,0] ).argmax() + BACK_WALL_MARGIN # mode minus margin
#        self.wall_left = left
#    
#        right = line_coords(thresh(right_laser_img, 1, 10))
#        if len(right)==0:
#            right = calibration_img.shape[1]
#        else:
#            right = np.bincount( right[:,0] ).argmax() - BACK_WALL_MARGIN
#        self.wall_right = right

    def process_picture(self, picture, angle, right=False):
        ''' Takes picture and angle (in degrees).  Adds to point cloud '''
        x_center = picture.shape[1] * CENTER

        left_img = ignore_half(picture, True)
        right_img = ignore_half(picture, False)

        lthresh = thresh(left_img)  # Do a hard threshold of the image
        rthresh = thresh(right_img)  # Do a hard threshold of the image

        #cv2.imwrite('thresh_%d.jpg'%angle, thresholded) # for debugging
        pixels = line_coords(lthresh, x_center)     # Get line coords from image
        # filter out any pixels outside back wall lines
        pixels = filter(lambda p: p[0] < self.wall_right - x_center and p[0] > self.wall_left - x_center, pixels)
        # add to point cloud
        self.point_cloud.extend(
            process_line(pixels, angle, self.distance, right=right))

        pixels = line_coords(rthresh, x_center)     # Get line coords from image
        # filter out any pixels outside back wall lines
        pixels = filter(lambda p: p[0] < self.wall_right - x_center and p[0] > self.wall_left - x_center, pixels)
        # add to point cloud
        self.point_cloud.extend(
            process_line(pixels, angle, self.distance, right=right))

        return (left_img, right_img)

    def process_pictures(self, pictures, prefix=None):
        # process pics - Does not support dual lasers yet
        if filter(lambda x: x is None, pictures):
            raise Exception('some pictures are null')

        processed = []
        for i, picture in enumerate(pictures):
            #picture = resize_image(picture)  # if we turn resize back on
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

    def save_ply(self, path):
        with open(path, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write("element vertex {0}\n".format(len(self.point_cloud)))
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("element face 0\n")
            f.write("element edge 0\n")
            f.write("end_header\n")
            for point in self.point_cloud:
                f.write("{0} {1} {2}\n".format(point[0], point[1], point[2]))

    def visualize(self):
        visualize_points(np.array(self.point_cloud))
        #visualize_mesh(np.array(self.point_cloud))

    def process_continuous(self, images, num_rotations, prefix=None, right=False):
        processed_l = []
        processed_r = []
        for i, img in enumerate(images):
            angle = 360.0 * i * num_rotations / len(images)
            l, r = self.process_picture(img, angle)
            processed_l.append(l)
            if right:
                processed_r.append(r)
            print("Processing image {0} of {1}".format(i + 1, len(images)))

        if prefix:
            util.save_images(processed_l,
                             prefix,
                             dir_name=os.path.join("img", prefix + "_l", "processed"))
            if right:
                util.save_images(processed_r,
                                 prefix,
                                 dir_name=os.path.join("img", prefix + "_r", "processed"))


def process_scan(rotations, prefix,
                 calibration_name="calibration/calibration.jpg",
                 right=False):
    calibration_img = cv2.imread(calibration_name)
    proc = Processor(calibration_img)

    images = []
    path = os.path.join('img', prefix, 'raw')
    for f in os.listdir(path):
        f = os.path.join(path, f)
        images.append(cv2.imread(f))

    #proc.process_pictures(images)
    proc.process_continuous(images, rotations, prefix=prefix, right=right)
    #proc.visualize()
    proc.save_ply("ply/" + prefix + '.ply')
