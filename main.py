import cv2
import numpy as np
from CourtTracker import courtTracker as cT
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, lfilter, freqz, filtfilt, find_peaks, argrelextrema



def remove_close_numbers(lst, y, rate, invert):
    np.argsort(lst)
    if invert:
        lst.reverse()
        #print(lst)
    result = [lst[0]]
    min_distance = np.mean(np.diff(lst)) * rate
    for num in lst[1:]:
        if abs(num - result[-1]) >= min_distance:
            result.append(num)
        else:
            result[-1] = num


    return result



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def min_max_finder(data, radius = 1):
    mins = argrelextrema(data, np.less, order=radius)[0]
    maxs = argrelextrema(data, np.greater, order=radius)[0]


    # Join the arrays
    crits = []
    for m in mins:
        crits.append(mappingKeys[m])
    for m in maxs:
        crits.append(mappingKeys[m])

    # Transform slope and time to a DataFrame
    data = {'Time': times, 'X': list(posx.values()), 'Y': list(posy.values()), 'Frame': n_frame}
    df = pd.DataFrame(data)
    df.set_index('Frame', inplace=True)
    df = df.dropna()
    #compute data for plotting
    x = df['Time']
    crits = np.sort(crits)
    y = df['Y']
    return crits
  

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
    file_path = './tennis_1_output.mp4'
    frames, fps = read_video(file_path)
    times = []
    n_frame = []
    posx = {}
    posy = {}
    i=0
    while i<len(frames)-1:
        frame1 = frames[i]
        f1 = applyColorFilter(frame1)
        circ_f1 = getCircPos(f1)
        if circ_f1 is not None:
            #Calculate the time of the frame
            t = 1/fps
            v = i*t
            times.append(v)
            n_frame.append(i)
            posx[i] = (circ_f1[0])
            posy[i] = (circ_f1[1])
        i+=1

    # Create a mapping of the keys
    i=0
    mappingKeys = {}
    ykeys = list(posy.keys())
    while i<len(ykeys):
        mappingKeys[i] = ykeys[i]
        i+=1

    # Find local maxima
    data = np.array(list(posy.values()))
    data_original = data
    radius = 1 # number of elements to the left and right to compare to 1 is the best
    mins = argrelextrema(data, np.less, order=radius)[0]
    maxs = argrelextrema(data, np.greater, order=radius)[0]

    # Join the arrays
    crits = []
    for m in mins:
        crits.append(mappingKeys[m])
    for m in maxs:
        crits.append(mappingKeys[m])

    # Transform slope and time to a DataFrame
    data = {'Time': times, 'X': list(posx.values()), 'Y': list(posy.values()), 'Frame': n_frame}
    df = pd.DataFrame(data)
    df.set_index('Frame', inplace=True)
    df = df.dropna()
    #compute data for plotting
    x = df['Time']
    y = df['Y']




    ### Proccesing ####

    crits =  min_max_finder(data_original)
    crits_process = remove_close_numbers(crits, y, 0.4, 0)

    #Normalize
    y = y/np.max(y)

    # Kernel size
    window_size = 2 # 2 is normal
    window_size = 3 # 2 is normal

    # Mean filter kernel
    kernel = np.ones(window_size) / window_size

    # Filter the signal
    filtered_signal = np.convolve(y, kernel, mode='same')
    critics_3 = min_max_finder(filtered_signal)


    # Setting standard filter requirements.
    order = 6
    fs = 30
    cutoff = 1.5 # Normally 2


    signal = y
    signal = butter_lowpass_filter(signal, cutoff, fs, order)

    # FFT of the signal
    # fft_result = np.fft.fft(signal)
    # fft_freq = np.fft.fftfreq(len(fft_result), 1/fs)


    
    bounce_player_1 = [value for value in critics_3 if y[value] <= 0.5]
    bounce_player_1 = remove_close_numbers(bounce_player_1, y, 0.2, invert = 0)


    crits_2 = min_max_finder(signal)
    crits_2 = [value for value in crits_2 if value <= len(signal)]


     ## analysis of critic points ##

    #Player 1 is gonna be the upper one
    player_1_critics =  [value for value in crits_2 if  y[value] <= 0.5]
    player_1_critics = remove_close_numbers(player_1_critics, y, 0.8, invert = 0)

    #Player 2
    player_2_critics_aux = [value for value in crits_2 if  y[value] >= 0.5]
    #print('value,  y[value] ', (player_2_critics_aux,  y[player_2_critics_aux] ))
    #print('player_2_critics_aux', player_2_critics_aux)

    player_2_critics = []
    for element in player_2_critics_aux:
        #print('element', element)
        player_2_critics_aux_2 = [value for value in crits if abs(value - element) < 7]
        if  len(player_2_critics_aux_2) == 0:
            player_2_critics_aux_2 = [value for value in critics_3 if abs(value - element) < 7]

        if len(player_2_critics_aux_2) > 1:
            player_2_critics_aux_2 = [np.max(player_2_critics_aux_2)]
        player_2_critics.append(player_2_critics_aux_2[0])





    #Bounce_player_1
    bounce_player_1_aux = []
    for element in player_1_critics:
        bounce_player_1 = [value for value in bounce_player_1 if abs(value - element) > 7]
    
    for element in player_2_critics:
        bouncer_1 = [value for value in bounce_player_1 if value > element]
        bouncer_1_original = [value for value in crits_process if value > element]
        if len(bouncer_1) > 0:
            if abs(bouncer_1[0] - element) > 60:
                if  abs(bouncer_1_original[0] - element) < 30:
                    bounce_player_1.append(bouncer_1_original[0])
                    bounce_player_1.sort()


    #Bounce_player_2
    crits_bounce_player_2 = min_max_finder(data_original, 3) #should be 5
    crits_bounce_player_2_original = crits_bounce_player_2
    crits_bounce_player_2 = [value for value in crits if y[value] >= 0.75]
    crits_bounce_player_2 = remove_close_numbers(crits_bounce_player_2, y, 0.1, 0)

    peaks, _ = find_peaks(y, distance=2)  # 'distance' define distance between peaks
    peaks = [value for value in peaks if value <= len(x)]
    peaks = np.array(peaks)
    peaks = [value for value in peaks if value  in x.keys()]

    #print(peaks)
    #print(len(x), len(y))
    #print('x keys', x.keys())
    #print(353 in x.keys())







    #Graph the signal and its frequency spectrum
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(x, y)
    plt.plot(x[crits], y[crits], 'ro')
    plt.title('Signal through time')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    #print(type(crits))
    #print(type(peaks))
   
    plt.subplot(3, 1, 1)
    # plt.plot(x, signal)
    # #print(len(crits_2))
    # #print('critics', crits_2)
    plt.plot(x, signal)
    plt.plot(x[player_2_critics], y[player_2_critics], 'yo')
    plt.plot(x[player_1_critics], y[player_1_critics], 'o', color='green', markersize = 6)
    plt.plot(x[crits_process], y[crits_process], 'o', color='yellow', markersize = 4)

    plt.plot(x[crits_2], y[crits_2], 'bo')

    # plt.ylim(100, 650)
    plt.title('Filtered signal through time')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 2)
    plt.plot(x, y)
    plt.plot(x[crits_bounce_player_2], y[crits_bounce_player_2], 'o', color='violet', markersize = 4)
    plt.plot(x[bounce_player_1], y[bounce_player_1], 'o', color = 'red')
    plt.plot(x[player_1_critics], y[player_1_critics], 'o',color= 'blue', markersize= 4)
    plt.plot(x[player_2_critics], y[player_2_critics], 'o', color='green', markersize = 6)
    plt.title('Frequency spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Spectrum Amplitude')
    plt.savefig('frequency.png')

    plt.subplot(3, 1, 3)
    plt.plot(x, y)
    plt.plot(x[crits], y[crits], 'o', color='green', markersize = 6)
    plt.plot(x[crits_bounce_player_2_original], y[crits_bounce_player_2_original], 'o', color='blue', markersize = 4)
    plt.plot(x[peaks], y[peaks], 'o', color='red', markersize = 5)
    plt.title('Frequency spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Spectrum Amplitude')


    plt.tight_layout()
    plt.savefig('frequency.png')

    # plt.show()


# Drawing the player_1 bounce
    for i in crits_bounce_player_2:
            xpos = posx[i]
            ypos = posy[i]
            frame = cv2.circle(frames[i], (xpos,ypos), radius=0, color=(236,30,250), thickness=15)
            frames[i] = frame

# Drawing the player_1 bounce
    for i in bounce_player_1:
            xpos = posx[i]
            ypos = posy[i]
            frame = cv2.circle(frames[i], (xpos,ypos), radius=0, color=(236,30,250), thickness=15)
            frames[i] = frame

# Drawing the player_1_critics
    for i in player_1_critics:
            xpos = posx[i]
            ypos = posy[i]
            frame = cv2.circle(frames[i], (xpos,ypos), radius=0, color=(0, 255, 0), thickness=15)
            frames[i] = frame

# Drawing the player_2_critics
    for i in player_2_critics:
            xpos = posx[i]
            ypos = posy[i]
            frame = cv2.circle(frames[i], (xpos,ypos), radius=0, color=(0, 255, 255), thickness=15)
            frames[i] = frame
    

    write_video(frames, 'output.mp4', fps)