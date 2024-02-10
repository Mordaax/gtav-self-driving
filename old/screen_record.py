import numpy as np
import cv2
from mss import mss
from PIL import ImageGrab
import time

bounding_box = {'top':50, 'left': 0, 'width': 800, 'height': 600}

sct = mss()
last_time = time.time()

def process_img(image):
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    return processed_img

while True:
    #sct_img = sct.grab(bounding_box)
    screen = np.array(ImageGrab.grab(bbox=(0,50,800,600)))
    last_time=time.time()
    cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    
    #print('fps: {}'.format(1/(time.time() - last_time)))  
    #cv2.imshow('screen', np.array(sct_img))

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
          
        cv2.destroyAllWindows()
        break