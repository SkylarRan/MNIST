from picamera import PiCamera
from time import sleep
import os

'''
bool take_picture(filename)
'''
def take_picture(filename=None):
    '''Input:  filename and path of picture
       Output: True or False
    ''' 
    if filename == None:
        return False
    if os.path.isfile( filename ):
        os.remove(filename)

    camera = PiCamera()
    if camera:
        camera.resolution = (2592,1944)
        camera.framerate = 15
        camera.start_preview()
        sleep(2)
        camera.capture(filename)
        camera.stop_preview()
        return True
    
    else:
        return False

