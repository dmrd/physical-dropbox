import serial
import serial.tools.list_ports
import cv2

# Use to try finding the arduino serial port
PORT_SIGNAL = "usbmodem"


class Turntable:
    def __init__(self, serial_object):
        self.com = serial_object
        # Number of steps for the turntable to do a full rotation
        self.STEPS_PER_ROTATION = 3200

    def step(self, steps=1):
        '''
        Accepts number of steps to take
        '''
        self.com.write(str(steps))

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

    def on(self):
        self.com.write('-1')

    def off(self):
        self.com.write('-2')


class Camera:
    def __init__(self, camera_id=1):
        self.c = cv2.VideoCapture(camera_id)

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

    def step(self, rotation_intervals=36):
        '''
        Take a step and return an image.
        Step size calculated to correspond to num_steps_per_rotation
        Returns resulting image
        '''
        self.turntable.step_interval(rotation_intervals)
        self.laser.on()
        img = self.camera.take_picture()
        self.laser.off()
        return img

    def do_rotation(self, rotation_intervals=36):
        '''
        Do an entire rotation
        Returns [array of images] for the entire rotation
        '''
        return [self.step(rotation_intervals)
                for i in range(rotation_intervals)]
