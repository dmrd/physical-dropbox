import serial

s = serial.Serial('/dev/cu.usbmodem1411') # Need to specify port to arduino

class Turntable:
    def __init__(self, serial_object):
        pass

    def step(self, angle):
        '''Excepts angle to turn to in degrees'''
        s.write(str(angle))



class Laser:
    def __init__(self, serial_object):
        pass

    def on(self):
        s.write('-1')


    def off(self):
        s.write('-2')


class Camera:
    def __init__(self):
        pass

    def take_picture(self):
        pass

    def transfer_picture(self):
        pass

    def transfer_pictures(self):
        pass

    def set_exposure(self):
        pass
