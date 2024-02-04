import cv2
import numpy as np
from CourtTracker import courtTracker as cT
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import argrelextrema

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

    #result = cv2.medianBlur(result,5) 

    # Convert to cv color space
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    
    # define range of red color in HSV
    lower_red = np.array([0,200,200])
    upper_red = np.array([14,255,255])

    # define range of red color in HSV
    lower_red_ub = np.array([160,200,200])
    upper_red_ub = np.array([180,255,255])
    mask_ub = cv2.inRange(hsv, lower_red_ub, upper_red_ub)

    # Threshold the HSV image using inRange function to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)
    def_mask = mask + mask_ub

    # Bitwise-AND mask and original image 
    result = cv2.bitwise_and(result, result, mask=mask)

    return def_mask

def findCircles(img: np.ndarray)->np.ndarray:
    minDist = 60
    # higher threshold of Canny Edge detector, lower threshold is twice smaller
    p1UpperThreshold = 1000
    # the smaller it is, the more false circles may be detected
    p2AccumulatorThreshold = 5
    mR = 0
    mRa = 15
    # use gray image, not edge detected
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,minDist,
                                param1= p1UpperThreshold,param2=p2AccumulatorThreshold,minRadius=mR,maxRadius=mRa)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(img, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(img, center, radius, (255, 0, 255), 3)
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

def getCircPos(f1: np.ndarray)->np.ndarray:
    """ Get delta position between two frames
    :params
        f1: first frame
    :return
        delta: delta position between f1 and f2
    """
    #Find coordinate of the tennis ball in each frame
    circles1 = findCircles(f1)
    if circles1 is None:
        return None
    x1 = circles1[0][0][0]
    y1 = circles1[0][0][1]
    return (x1, y1)


def write_video(frames, path_output_video, fps):
    """ Write .avi file with detected ball tracks
    :params
        frames: list of original video frames
        path_output_video: path to output video
        fps: frames per second
    """
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'), 
                          fps, (width, height))
    for num in range(len(frames)):
        frame = frames[num]

        out.write(frame) 
    out.release()

if __name__ == '__main__':
    file_path = './tennis2.mp4'
    frames, fps = read_video(file_path)
    times = []
    posx = []
    posy = []
    i=0
    lastPosx = 0
    lastPosy = 0
    deltax = 1
    deltay = 1
    north = False
    while i<len(frames)-1:
        frame1 = frames[i]
        f1 = applyColorFilter(frame1)
        circ_f1 = getCircPos(f1)
        #Calculate the time of the frame
        t = 1/fps
        v = i*t
        times.append(v)
        if circ_f1 is not None:
            if lastPosy < circ_f1[1]:
                north = True
            posx.append(circ_f1[0])
            if not (circ_f1[0] ==lastPosx):
                deltax = abs(circ_f1[0] - lastPosx)
            lastPosx = circ_f1[0]
            posy.append(circ_f1[1])
            if not (circ_f1[1] ==lastPosy):
                deltay = abs(circ_f1[1] - lastPosy)
            lastPosy = circ_f1[1]
        else:
            if north:
                lastPosx -= deltax
                lastPosy -= deltay
                posx.append(lastPosx)
                posy.append(lastPosy)
            else:
                lastPosx += deltax
                lastPosy += deltay
                posx.append(lastPosx)
                posy.append(lastPosy)
        i+=1

    # Find local maxima
    data = np.array(posy)
    radius = 4 # number of elements to the left and right to compare to
    mins = argrelextrema(data, np.less, order=radius)[0]
    maxs = argrelextrema(data, np.greater, order=radius)[0]

    # Join the arrays
    crits = np.concatenate((mins, maxs))

    # Transform slope and time to a DataFrame
    data = {'Time': times, 'X': posx, 'Y': posy}
    df = pd.DataFrame(data)
    df = df.dropna()
    #compute data for plotting
    x = df['Time']
    y = df['Y']

    #plot the chart
    fig, ax = plt.subplots()
    bar_plot = ax.plot(x, y)
    
    #plot the critical points
    bar_plot = ax.plot(x[crits], y[crits], 'ro')

    #set chart title
    ax.set_title('Position of the ball throughout time')

    #set y-axis label
    ax.set_ylabel('Height of the ball on the screen')
    plt.show()
    
    print(crits)
    # For each point in the critical points, write the frame to a file
    for i in crits:
        frame = cv2.circle(frames[i], (posx[i],posy[i]), radius=0, color=(23, 252, 3), thickness=12)
        frames[i] = frame

    write_video(frames, 'output.mp4', fps)    
    # edges = cT.filterBackGround(frames[0])
    # cv2.imshow('edges', edges)