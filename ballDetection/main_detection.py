from model import BallTrackerNet
import torch
import cv2 as cv
from general import postprocess
from tqdm import tqdm
import numpy as np
import argparse
from itertools import groupby
from scipy.spatial import distance


def read_video(path_video):
    """ Read video file    
    :params
        path_video: path to video file
    :return
        frames: list of video frames
        fps: frames per second
    """
    cap = cv.VideoCapture(path_video)
    fps = int(cap.get(cv.CAP_PROP_FPS))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps




parser = argparse.ArgumentParser()
parser.add_argument('--video_path', type=str, default = './video_tennis_output_2.mp4' ,help='path to input video')
args = parser.parse_args()



frames, fps = read_video(args.video_path)
print(frames[50].shape)
window_name = 'image'
  
# Using cv2.imshow() method 
# Displaying the image 
# cv2.imshow(window_name, frames[1])
# cv2.waitKey(0) 


gray = cv.cvtColor( frames[50], cv.COLOR_BGR2GRAY)

src = frames[50]
gray = cv.medianBlur(gray, 5)


rows = gray.shape[0]
circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                            param1=100, param2=30,
                            minRadius=1, maxRadius=30)


if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv.circle(src, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv.circle(src, center, radius, (255, 0, 255), 3)


cv.imshow("detected circles", src)
cv.waitKey(0)