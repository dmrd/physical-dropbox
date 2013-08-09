import os
import time
import util
import serial
import serial.tools.list_ports
import cv2

# Use to try finding the arduino serial port
PORT_SIGNAL = "usbmodem"


def wait_arduino(com):
        com.read()  # Block until success byte received
        com.flushInput()
        com.flushOutput()


class Turntable:
    def __init__(self, serial_object):
        self.com = serial_object
        # Number of steps for the turntable to do a full rotation
        self.STEPS_PER_ROTATION = 3200

    def async_step(self, steps=1):
        '''
        Accepts number of steps to take. Does not wait for it to finish turning
        '''
        self.com.write(str(steps))

    def step(self, steps=1):
        '''
        Accepts number of steps to take
        '''
        self.com.write(str(steps))
        wait_arduino(self.com)

    def step_interval(self, rotation_intervals):
        '''
        Takes number of steps for one interval based on number of intervals in
            a rotation.
        36 would mean there are 36 turns in a rotation.
        '''
        steps = self.get_step_interval(rotation_intervals)
        self.step(steps)

    def get_step_interval(self, rotation_intervals):
        return self.STEPS_PER_ROTATION / rotation_intervals


class Laser:
    def __init__(self, serial_object):
        self.com = serial_object

    def on(self, right=False):
        signal = '-4' if right else '-2'
        self.com.write(signal)
        wait_arduino(self.com)
        time.sleep(1)  # Wait for laser to warm up

    def off(self, right=False):
        signal = '-3' if right else '-1'
        self.com.write(signal)
        wait_arduino(self.com)


class Camera:
    def __init__(self):
        # detect which camera to use
        for i in xrange(0, 3):
            cam = cv2.VideoCapture(i)
            # hacky hardcode to look for our camera
            if cam.get(3) == 1920.0 and cam.get(4) == 1080:
                break
        if not cam.isOpened():
            raise Exception("Could not find the camera.")
        self.c = cam
        print "selected camera %d" % i

    def take_picture(self):
        return self.c.read()

    def set_exposure(self):
        pass


class Scanner:
    """
    Wrapper class for getting the Laser, Camera, and Turntable classes to work
    together
    """
    def __init__(self, serial_port=None):
        # Try to find the arduino - tested for macs
        if not serial_port:
            # comports returns [list of (name, name, name) tuples]
            ports = [p[0] for p in serial.tools.list_ports.comports()
                     if 'usb' in p[0]]
            serial_port = ports[0]
        self.com = serial.Serial(serial_port, 9600)
        self.turntable = Turntable(self.com)
        self.laser = Laser(self.com)
        self.camera = Camera()

    def step(self, rotation_intervals=36, progress=None):
        '''
        Take a step and return an image.
        Step size calculated to correspond to num_steps_per_rotation
        Returns resulting image
            '''
        if progress:
            print(progress)
        self.turntable.step_interval(rotation_intervals)
        _, img = self.camera.take_picture()
        return img

    def do_rotation(self, rotation_intervals=36):
        '''
        Do an entire rotation
        Returns [array of images] for the entire rotation
        '''
        self.laser.on()
        result = [self.step(rotation_intervals, progress=i)
                  for i in range(rotation_intervals)]
        self.laser.off()
        return result

    def continuous(self, rotations=1, right=False):
        self.laser.on(right)

        self.turntable.async_step(rotations
                                  * self.turntable.STEPS_PER_ROTATION)
        self.com.setTimeout(0)  # Timeout immediately on
        images = []
        while not self.com.read():
            _, im = self.camera.take_picture()
            images.append(im)
        self.com.setTimeout(None)
        self.laser.off(right)

        return images


def run_scan(rotations, prefix, right=False):
    s = Scanner()
    print("Scanner initialized")
    print("Current laser is: " + "right" if right else "left")
    result = s.continuous(rotations, right=right)
    print("Images taken")
    raw_dir = 'raw_r' if right else 'raw_l'
    util.save_images(result,
                     prefix,
                     dir_name=os.path.join("img", prefix, raw_dir))
    print("Images saved")
