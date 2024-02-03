import numpy
import cv2 


#edges = edge(rgb2gray(background), 'canny', .2);

def filterBackGround(frame: numpy.ndarray)->numpy.ndarray:
    """ Apply color filter to frame
    :params
        frame: frame to apply filter
    :return
        frame: frame with color filter
    """
    # Convert to gray scale
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Find the median of the frames
    background = numpy.median(frame, 4)

    # Convert to gray scale
    hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)

    #Find edges in the image
    edges = cv2.Canny(hsv, 100, 200)

    return edges