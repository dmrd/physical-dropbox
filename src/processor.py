class Processor:
    def __init__(self):
        pass

    def process_picture(self, picture, angle):
        ''' Takes picture and angle (in radians).  Adds to point cloud '''
        pass

    def process_pictures(self, pictures):
        for picture in pictures:
            self.process_picture(picture)

