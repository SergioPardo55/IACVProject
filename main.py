import os
import cv2
import numpy as np
from CourtTracker import courtTracker as cT

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
    """

    result = frame.copy()

    result = cv2.medianBlur(result,5) 

    # Convert to cv color space
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    
    # define range of red color in HSV
    lower_red = np.array([0,100,100])
    upper_red = np.array([20,255,255])
     
    # Threshold the HSV image using inRange function to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Bitwise-AND mask and original image 
    result = cv2.bitwise_and(result, result, mask=mask)

    return result

def findCircles(img: np.ndarray)->np.ndarray:
    img = cv2.medianBlur(img,5) 
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        cv2.imshow('detected circles',cimg)

if __name__ == '__main__':
    file_path = './video_tennis_output_2.mp4'
    frames, fps = read_video(file_path)
    for frame in frames:
        f1 = applyColorFilter(frame)
        cv2.imshow('frame', f1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #edges = cT.filterBackGround(frames[0])
    #cv2.imshow('edges', edges)
