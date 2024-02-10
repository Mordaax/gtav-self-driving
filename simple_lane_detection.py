import numpy as np
import cv2
from mss import mss
from PIL import ImageGrab
import time
from utils.directkeys import PressKey, W, A, S, D, ReleaseKey
from utils.draw_lane import draw_lanes
from utils.move import straight, left, right, slow
from utils.getkeys import key_check
import os
from utils.grabscreen import grab_screen


bounding_box = {'top':50, 'left': 0, 'width': 800, 'height': 600}
sct = mss()
last_time = time.time()

def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])
    y2 = int(y1*3.0/5)      
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]
 
def average_slope_intercept(image, lines):
    left_fit    = []
    right_fit   = []
    
    try:
        if len(lines) < 2:
            return None
        for line in lines:
            for x1, y1, x2, y2 in line:
                fit = np.polyfit((x1,x2), (y1,y2), 1)
                slope = fit[0]
                intercept = fit[1]
                if slope < 0: 
                    left_fit.append((slope, intercept))
                else:
                    right_fit.append((slope, intercept))
        
        left_fit_average  = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line  = make_points(image, left_fit_average)
        right_line = make_points(image, right_fit_average)
        averaged_lines = [left_line, right_line]
        return averaged_lines
    except:
        return None

def add_blur(img):
    return cv2.GaussianBlur(img, (5,5), 0)

def draw_lines(img, lines):
    try: 
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
    except:
        return img    
    return img

def get_hough_lines(cropped_canny_image):
    return cv2.HoughLinesP(cropped_canny_image, 1, np.pi/180, 180, 
         minLineLength=80, maxLineGap=15)

def region_of_interest(canny_img, vertices):
    mask = np.zeros_like(canny_img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(canny_img, mask)
    return masked_image



def process_img(image):
    original_image = image
    
    
            

    processed_img = cv2.Canny(image, threshold1=200, threshold2=300) #edge detection

    vertices = np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500]])
    processed_img = region_of_interest(processed_img, [vertices]) 

    blured_img = add_blur(processed_img) #Add GaussianBlur

    hough_lines = get_hough_lines(blured_img) #HoughLines detection

    #averaged_lines = average_slope_intercept(blured_img, hough_lines)

    #lined_image = draw_lines(blured_img, averaged_lines)

    m1 = 0
    m2 = 0
    try:
        l1, l2, m1,m2 = draw_lanes(original_image,hough_lines)
        cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), [102,190,208], 10)
        cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [102,190,208], 10)
    except Exception as e:
        #print(str(e))
        pass
    try:
        for coords in hough_lines:
            coords = coords[0]
            try:
                cv2.line(processed_img, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 3)
                
                
            except Exception as e:
                #print(str(e))
                pass
        
    except Exception as e:
        pass

    return processed_img,original_image, m1, m2, training_data

""" time.sleep(3)
PressKey(W)
time.sleep(3)
ReleaseKey(W) """
time.sleep(3)


    

while True:
    screen = grab_screen(region=(0,40,800,640))
    #screen = np.array(ImageGrab.grab(bbox=(0,50,800,600)))
    #new_screen, blured_img = process_img(screen)
    last_time=time.time()
    new_screen,original_image, m1, m2 = process_img(screen)
    """ if m1 < 0 and m2 < 0:
        right()
    elif m1 > 0 and m2 > 0:  
        left()
    else:
        straight() """
    
    cv2.imshow('window', cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))


    if (cv2.waitKey(1) & 0xFF) == ord('q'):
          
        cv2.destroyAllWindows()
        break