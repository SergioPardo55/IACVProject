import os
import cv2
import numpy as np
from CourtTracker import courtTracker as cT
import matplotlib.pyplot as plt
import pandas as pd

def read_video(path_video):
    """ Read video file    
    :params
        path_video: path to video file
    :return
        frames: list of video frames
        fps: frames per second
    """
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps

def applyColorFilter(frame: np.ndarray)->np.ndarray:
    """ Apply color filter to frame
    :params
        frame: frame to apply filter
    :return
        frame: frame with color filter
    Applies a color filter over the frame to detect the trace of the 
    already detected ball
    """

    result = frame.copy()

    result = cv2.medianBlur(result,5) 

    # Convert to cv color space
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    
    # define range of red color in HSV
    lower_red = np.array([0,200,200])
    upper_red = np.array([20,255,255])
     
    # Threshold the HSV image using inRange function to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Bitwise-AND mask and original image 
    result = cv2.bitwise_and(result, result, mask=mask)

    return mask

def findCircles(img: np.ndarray)->np.ndarray:
    # img = cv2.medianBlur(img,5) 
    # cv2.imshow("detected circles", img)
    # cv2.waitKey(0)

    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    minDist = 60
    # higher threshold of Canny Edge detector, lower threshold is twice smaller
    p1UpperThreshold = 1000
    # the smaller it is, the more false circles may be detected
    p2AccumulatorThreshold = 5
    mR = 0
    mRa = 0

    # use gray image, not edge detected
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,minDist,
                                param1= p1UpperThreshold,param2=p2AccumulatorThreshold,minRadius=mR,maxRadius=mRa)
    circles = np.uint16(np.around(circles))
    return circles

def getDeltaPosition(f1: np.ndarray, f2: np.ndarray)->np.ndarray:
    """ Get delta position between two frames
    :params
        f1: first frame
        f2: second frame
    :return
        delta: delta position between f1 and f2
    """
    #Find coordinate of the tennis ball in each frame
    circles1 = findCircles(f1)
    circles2 = findCircles(f2)

    x1 = circles1[0][0][0]
    y1 = circles1[0][0][1]
    x2 = circles2[0][0][0]
    y2 = circles2[0][0][1]

    dx = x2-x1
    dy = y2-y1

    #Find the difference in position of the tennis ball
    delta = dy/dx

    return delta

if __name__ == '__main__':
    file_path = './video_tennis_output_2.mp4'
    frames, fps = read_video(file_path)
    i=1
    while i<len(frames)-2:
        frame1 = frames[i]
        frame2 = frames[i+1]
        f1 = applyColorFilter(frame1)
        f2 = applyColorFilter(frame2)
        m = getDeltaPosition(f1, f2)
        t = 1/fps
        v = i*t

        i+=1
    #     cv2.imshow('frame', f1)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # edges = cT.filterBackGround(frames[0])
    # cv2.imshow('edges', edges)
