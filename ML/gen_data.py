import os
import sys
import cv2
import time
import argparse
import numpy as np
import matplotlib as plt

np.set_printoptions(threshold=sys.maxsize)

SPEED = 10
IMAGESIZE = 80
OUTDIR = "./data/train"
CANNYMIN = 100
CANNYMAX = 200

CATEGORIES = [
    'Angery', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
]

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", help="specify final image size", type=int, default=IMAGESIZE)
parser.add_argument("-o", "--output", help="path to save csv to", action="store", default=OUTDIR)
parser.add_argument("--min", help="specify canny min value", type=int, default=CANNYMIN)
parser.add_argument("--max", help="specify canny max value", type=int, default=CANNYMAX)
parser.add_argument("--speed", help="specify capture speed (smaller is faster)", type=int, default=SPEED)
parser.add_argument("--nohist", help="skip background subtraction", action="store_true")
args = parser.parse_args()

cap = cv2.VideoCapture(0)
img = None

rectangles = None
rect_offset = 10
face_hist = None
is_hist_created = False
selected = False
counter = 1

def draw_rect(frame):
    global rectangles
    rows, cols, _ = frame.shape
    
    rect_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20],dtype=np.uint32
    )
    rect_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20],dtype=np.uint32
    )
    rectangles = list(zip(rect_x, rect_y))
    for rect in rectangles:
        cv2.rectangle(frame, (rect[1], rect[0]),
                      (rect[1] + rect_offset, rect[0] + rect_offset),
                      (0, 255, 0), 1
        )
    
    return frame

def face_histogram(frame):
    global rectangles
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)
    for i, rect in enumerate(rectangles):
        roi[i * rect_offset: i * rect_offset + rect_offset, 0: rect_offset] = hsv_frame[rect[0]:rect[0] + rect_offset, rect[1]:rect[1] + rect_offset]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)

def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 115, 255, cv2.THRESH_BINARY)

    thresh = cv2.merge((thresh, thresh, thresh))

    return cv2.cvtColor(cv2.bitwise_and(frame, thresh), cv2.COLOR_BGR2GRAY)

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def edge_masking(grame, hist):
    if (not args.nohist):
        # converting BGR to HSV 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

        # define range of red color in HSV 
        lower_red = np.array([30,150,50]) 
        upper_red = np.array([255,255,180]) 

        # create a red HSV colour boundary and  
        # threshold HSV image 
        mask = cv2.inRange(hsv, lower_red, upper_red) 

        # Bitwise-AND mask and original image 
        res = cv2.bitwise_and(frame,frame, mask= mask) 

    # finds edges in the input image image and 
    # marks them in the output map edges 
    return cv2.Canny(frame,args.min,args.max) 

while cap.isOpened():
    # Capture frame-by-frame
    pressed_key = cv2.waitKey(1)
    ret, frame = cap.read()
    
    frame = adjust_gamma(frame, 1)
    
    # Display the resulting frame
    if not is_hist_created and not args.nohist:
        cv2.imshow('frame', draw_rect(frame))
    else:
        if (not args.nohist):
            hist_mask = hist_masking(frame, face_hist)
            edges = edge_masking(frame, hist_mask)
            
            cv2.imshow('frame',cv2.bitwise_and(hist_mask, edges))
        else:
            edges = edge_masking(frame, False)
            cv2.imshow('frame',edges)

        counter = counter % args.speed
        if counter == 0 and selected:
            path = os.path.join(args.output, selected, str(time.time()) + ".png")
            print(f"{selected}: Saving image to {path}{args.size}")
            img = cv2.resize(edges, (args.size, args.size))
            cv2.imwrite(path, img)
            
    #Controls
    if pressed_key == ord('+'):
        break
    elif pressed_key == ord('[') and not args.nohist:
        is_hist_created = True
        face_hist = face_histogram(frame)
    elif pressed_key == ord(']'):
        is_hist_created = False
        selected = False
        
    if (is_hist_created or args.nohist) and not selected:
        if pressed_key == ord('0'):
            selected = 'Angery'
        elif pressed_key == ord('1'):
            selected = 'Disgust'
        elif pressed_key == ord('2'):
            selected = 'Fear'
        elif pressed_key == ord('3'):
            selected = 'Happy'
        elif pressed_key == ord('4'):
            selected = 'Sad'
        elif pressed_key == ord('5'):
            selected = 'Suprise'
        elif pressed_key == ord('6'):
            selected = 'Neutral'
    
    counter += 1
    
cap.release()
cv2.destroyAllWindows()
        