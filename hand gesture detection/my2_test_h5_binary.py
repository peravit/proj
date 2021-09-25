from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt



detection_graph, sess = detector_utils.load_inference_graph()


parser = argparse.ArgumentParser()
parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.2,
        help='Score threshold for displaying bounding boxes')
parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        default=0,
        help='Device index of the camera.')
parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=320,
        help='Width of the frames in the video stream.')
parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=180,
        help='Height of the frames in the video stream.')
parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
args = parser.parse_args()

#cap = cv2.VideoCapture(args.video_source)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

cap = cv2.VideoCapture(0)

im_width, im_height = (cap.get(3), cap.get(4))

font = cv2.FONT_HERSHEY_PLAIN
fontscale = 2
thickness = 2

filename = 'model_7.h5'
model = load_model(filename)
width = 28
height = 28
dsize = (width, height)

#path = 'C:/Users/S/Desktop/handdetect/handtracking/handtracking/data_img/'

#n = 0

while True:
    
    ret, image_np = cap.read()

    try:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    except:
        print("Error converting to RGB")
        
#    skip every 4 frame
#    if n%5 != 0:
#        n += 1
#        continue
    
#    now = datetime.datetime.now()
#    stamp = str(int(time.mktime(now.timetuple())))
    
    boxes, scores = detector_utils.detect_objects(image_np,
                                                      detection_graph, sess)
        
    
    left, right, top, bottom = (boxes[0][1] * im_width, boxes[0][3] * im_width,
                                          boxes[0][0] * im_height, boxes[0][2] * im_height)
    
    
    top2 = int(top - 0.4*(bottom - top))
    if top2 < 0:
        top2 = 0
    bottom2 = int(bottom)
    left2 = int(left - 0.2*(right - left))
    if left2 < 0:
        left2 = 0
    right2 = int(right + 0.2*(right - left))
    
    crop = image_np[top2:bottom2,left2:right2]
    
#    filename = (path + stamp + '_a' + '.jpg')
#    cv2.imwrite(filename, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    
#    filename = (path + stamp + '_b' + '.jpg')
#    cv2.imwrite(filename, crop_gray)
    
    top3 = int(top + 0.5*(bottom - top))
    bottom3 = int(bottom - 0.3*(bottom - top))
    left3 = int(left + 0.4*(right - left))
    right3 = int(right - 0.4*(right - left))

    
    crop2 = image_np[top3:bottom3,left3:right3]
    
    hsv2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2HSV)
    
    hist = cv2.calcHist( [hsv2], 
                         [0],         # index of channel(s) used to create histogram
                         None,        # mask image
                         [256],       # the number of bins
                         [0,256] )
    
    
    blur = cv2.GaussianBlur(crop, (15, 15), 0)
    
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    color_skin = np.argwhere(hist > 0)
    
    lower = np.array([color_skin[0][0],20,50])
    upper = np.array([color_skin[-1][0],255,255])

    threshold = cv2.inRange(hsv, lower, upper)
    
    kernel = np.ones((5,5),np.uint8)

    closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    
    contours,hierarchy = cv2.findContours( opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    
    if len(contours) > 0 :

        crop = cv2.drawContours ( crop, contours, -1, (0,255,255), thickness=2 )
    
        mask = np.zeros_like( threshold )
        cv2.drawContours ( mask, [ max(contours, key=cv2.contourArea) ], 0, (255,255,255), thickness=-1 )
        
        cv2.imshow('Mask',mask)
        
        mask2 = cv2.resize(mask, dsize)
#        flatten = mask2.flatten()
        mask_pred = mask2/255
        mask_pred = mask_pred.reshape(width,height,-1)
        mask_pred = np.expand_dims(mask_pred,axis=0)
        p = model.predict_classes(mask_pred)
        text = str(p[0])
#        filename = (path + stamp + '_c' + '.jpg')
#        cv2.imwrite(filename, mask)
    
    cv2.imshow('Crop',cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    cv2.imshow('Threshold',threshold)
    
    
    cv2.rectangle(image_np,(int(left), int(top)),(int(right), int(bottom)),(0,255,0),thickness = 2)
    
    cv2.rectangle(image_np,(left2,top2),(right2,bottom2),(255,0,0),thickness = 2)
    
    cv2.rectangle(image_np,(left3,top3),(right3,bottom3),(0,0,255),thickness = 2)
    
    cv2.putText(image_np,text,(15,40), font, fontscale,(255,255,255),thickness,cv2.LINE_AA)
        
    cv2.imshow('Single-Threaded Detection',cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        
    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()


