import cv2
import numpy as np
from CourtTracker import courtTracker as cT
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, lfilter, freqz, filtfilt, find_peaks, argrelextrema



def remove_close_numbers(lst, y, rate, invert):
    '''
    Remove close numbers in the graph to avoid multiple peaks, e.g. multiple 'detected bounces or hits'
    '''
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
    '''
    Create a butterworth filter
    '''
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    '''
    Filter the signal using a butterworth filter
    '''
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def min_max_finder(data, radius = 1):
    '''
    Finds the local minima and maxima of a signal (critical points)
    '''
    mins = argrelextrema(data, np.less, order=radius)[0]
    maxs = argrelextrema(data, np.greater, order=radius)[0]

    # Join the arrays
    crits = []
    for m in mins:
        crits.append(mappingKeys[m])
    for m in maxs:
        crits.append(mappingKeys[m])

    crits = np.sort(crits)
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
    '''
    Find circles in the image through HoughCircles algorithm
    '''
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

def getCircPos(f1: np.ndarray)->np.ndarray:
    """ Calls findCircles and returns the position of the ball
    :params
        f1: first frame
    :return
        (x1, y1): position of the ball in the frame
    """
    #Find coordinate of the tennis ball in each frame
    circles1 = findCircles(f1)
    if circles1 is None:
        return None
    x1 = circles1[0][0][0]
    y1 = circles1[0][0][1]
    return (x1, y1)

def computeError(truth: list, detected: list)->float:
    '''
    Compute the error between the detected critical points and the ground truth
    '''
    error = 0
    count = 0
    not_detected = 0
    i = 0
    j = 0
    while i < len(truth):
        t = truth[i]
        d = detected[j]
        if abs(t-d)<5:
            error += abs((truth[i] - detected[j])/truth[i])
            count += 1
            j+=1
        else:
            not_detected += 1
        i+=1
    error = error/count
    return error, not_detected

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
    vid = 1
    file_path = './tennis_'+str(vid)+'_tracked.mp4'
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

    # Transform slope and time to a DataFrame
    data = {'Time': times, 'X': list(posx.values()), 'Y': list(posy.values()), 'Frame': n_frame}
    df = pd.DataFrame(data)
    df.set_index('Frame', inplace=True)
    df = df.dropna()
    #compute data for plotting
    x = df['Time']
    y = df['Y']

    data = np.array(list(posy.values()))
    data_original = data
    radius = 1 # number of elements to the left and right to compare to 1 is the best

    ### Proccesing ####    

    #Normalize
    y = y/np.max(y)

    # Kernel size for mean filter
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

     ## analysis of critic points ##

    # Find critical points with original signal
    crits = min_max_finder(data, radius)
    crits_process = remove_close_numbers(crits, y, 0.4, 0)
    
    # Find critical points with only values present
    crits_2 = min_max_finder(signal)
    crits_2 = [value for value in crits_2 if value <= len(signal)]

    # Find the critic points for player 1. All the points that are below 0.5 mean 
    # that the ball is in the upper part of the court
    player_1_critics =  [value for value in crits_2 if  y[value] <= 0.5]
    player_1_critics = remove_close_numbers(player_1_critics, y, 0.8, invert = 0)

    # Variable to find the bounce in the court of player 1
    bounce_player_1 = [value for value in critics_3 if y[value] <= 0.5]
    bounce_player_1 = remove_close_numbers(bounce_player_1, y, 0.2, invert = 0)

    #Find the critical points for player 2. All the points that are above 0.5 mean
    player_2_critics_aux = [value for value in crits_2 if  y[value] >= 0.5]


    player_2_critics = []
    for element in player_2_critics_aux:
        #Take the max index of the frame of the neighborhood that corresponds to the hit of the ball
        player_2_critics_aux_2 = [value for value in crits if abs(value - element) < 7]
        if  len(player_2_critics_aux_2) == 0:
            player_2_critics_aux_2 = [value for value in critics_3 if abs(value - element) < 7]

        if len(player_2_critics_aux_2) > 1:
            player_2_critics_aux_2 = [np.max(player_2_critics_aux_2)]
        player_2_critics.append(player_2_critics_aux_2[0])

    #Bounce_player_1
    bounce_player_1_aux = []
    for element in player_1_critics:
        #Preserve the bounces of the ball that happen in player 1 side of the court
        bounce_player_1 = [value for value in bounce_player_1 if abs(value - element) > 7]
    
    #If the bounce is not in the bounce_player_1 list, then we look for it in the bounces
    #  right after the hit of player 2 in crits_process
    for element in player_2_critics:
        bouncer_1 = [value for value in bounce_player_1 if value > element]
        bouncer_1_original = [value for value in crits_process if value > element]
        if len(bouncer_1) > 0:
            if abs(bouncer_1[0] - element) > 60:
                if  abs(bouncer_1_original[0] - element) < 30:
                    bounce_player_1.append(bouncer_1_original[0])
                    bounce_player_1.sort()

    #Detect the bounces on player 2 side of the court
    crits_bounce_player_2 = min_max_finder(data_original, 3) #should be 5
    crits_bounce_player_2_original = crits_bounce_player_2
    crits_bounce_player_2 = [value for value in crits if (y[value] >= 0.75 and value not in player_2_critics)]
    crits_bounce_player_2 = remove_close_numbers(crits_bounce_player_2, y, 0.1, 0)

    ### Plotting ###

    plt.plot(x, y)
    plt.plot(x[crits_bounce_player_2], y[crits_bounce_player_2], 'o', color='violet', markersize = 4, label='Player 2 bounce')
    plt.plot(x[bounce_player_1], y[bounce_player_1], 'o', color = 'red', label='Player 1 bounce')
    plt.plot(x[player_1_critics], y[player_1_critics], 'o',color= 'blue', markersize= 4, label='Player 1 hits the ball')
    plt.plot(x[player_2_critics], y[player_2_critics], 'o', color='green', markersize = 6,label='Player 2 hits the ball')
    plt.title('Y axis position vs time')
    plt.xlabel('Time [s]')
    plt.ylabel('Y position')
    plt.legend()

    plt.tight_layout()
    plt.savefig('yPosition'+str(vid)+'.png')
    
    #Store all critical points in a list
    all_crits = []
    # Drawing the player_1 bounce 
    for i in crits_bounce_player_2:
            xpos = posx[i]
            ypos = posy[i]
            frame = cv2.circle(frames[i], (xpos,ypos), radius=0, color=(236,30,250), thickness=15)
            frames[i] = frame
            all_crits.append(i)

# Drawing the player_1 bounce
    for i in bounce_player_1:
            xpos = posx[i]
            ypos = posy[i]
            frame = cv2.circle(frames[i], (xpos,ypos), radius=0, color=(236,30,250), thickness=15)
            frames[i] = frame
            all_crits.append(i)

# Drawing the player_1_critics
    for i in player_1_critics:
            xpos = posx[i]
            ypos = posy[i]
            frame = cv2.circle(frames[i], (xpos,ypos), radius=0, color=(0, 255, 0), thickness=15)
            frames[i] = frame
            all_crits.append(i)

# Drawing the player_2_critics
    for i in player_2_critics:
            xpos = posx[i]
            ypos = posy[i]
            frame = cv2.circle(frames[i], (xpos,ypos), radius=0, color=(0, 255, 255), thickness=15)
            frames[i] = frame
            all_crits.append(i)
    
    all_crits = np.sort(all_crits)
    # Write a text file with the critical points
    with open('critical_points.txt', 'w') as f:
        last = -1
        for item in all_crits:
            if item != last:
                if item in player_1_critics:
                    f.write("Player 1 hits the ball at %s seconds\n" % str(item*1/fps))
                if item in player_2_critics:
                    f.write("Player 2 hits the ball at %s seconds\n" % str(item*1/fps))
                if item in crits_bounce_player_2:
                    f.write("The ball bounces in player's 2 field at %s seconds\n" % str(item*1/fps))
                if item in bounce_player_1:
                    f.write("The ball bounces in player's 1 field at %s seconds\n" % str(item*1/fps))
    f.close()
    
    # Write a file per each frame of the video and store it in a folder with the name of the video
    for i in range(len(frames)):
        frame = frames[i]
        cv2.imwrite('./frames1/frame'+str(i)+'.jpg', frame)
    
    ## Computing errors ##
        
    # Establish ground truths
    if vid == 1:
        player_1_crits_truth = [60, 132, 210, 280, 348]
        player_2_crits_truth = [26, 96, 173, 244, 316]
        bounce_player_2_truth = [87, 162, 235, 307, 371]
        bounce_player_1_truth = [49, 121, 200, 271, 338, 382]
    elif vid == 2:
        player_1_crits_truth = [38,105, 173, 245, 231, 392, 464]
        bounce_player_2_truth = [61, 133, 20, 275, 346, 419, 492, 563]
        player_2_crits_truth = [12,71, 139, 201, 283, 355, 427, 498, 570]
        bounce_player_1_truth = [25,95, 166, 236, 308, 378, 455,528]
    elif vid == 3:
        player_1_crits_truth = []
        bounce_player_2_truth = []
        player_2_crits_truth = []
        bounce_player_1_truth = []
    elif vid == 4:
        player_1_crits_truth = [46, 101, 163, 225, 295, 356, 417, 475, 539, 607, 665, 729, 797, 867, 927, 988, 1051, 1109, 1169]
        bounce_player_2_truth = [58, 122, 184, 246, 313,379, 435, 497, 563, 623, 684, 751, 829,887, 948, 1013, 1070, 1129, 1191]
        player_2_crits_truth = [69, 133, 193, 258, 327,386, 445, 506, 569, 635, 694, 759, 837, 897, 957, 1021,1079, 1138]
        bounce_player_1_truth = [95, 150, 213, 284, 348, 405, 466, 530, 595, 655, 722, 781, 857, 919, 979,1041,1102, 1162]
    elif vid == 5:
        player_1_crits_truth = [27, 97, 169]
        bounce_player_2_truth = [50, 121]
        player_2_crits_truth = [61, 132]
        bounce_player_1_truth = [17, 85, 160]

    # Write errors and ground truths to a file
    all_truths = player_1_crits_truth + player_2_crits_truth + bounce_player_2_truth + bounce_player_1_truth
    all_truths.sort()
    with open('ground_truths'+str(vid)+'.txt', 'w') as f:
        last = -1
        for item in all_truths:
            if item != last:
                if item in player_1_crits_truth:
                    # Write file with 2 items using format()
                    f.write("Player 1 hits the ball on frame {} or at {} seconds\n".format(str(item), str(item*1/fps)))
                if item in player_2_crits_truth:
                    f.write("Player 2 hits the ball on frame {} or at {} seconds\n".format(str(item), str(item*1/fps)))
                if item in bounce_player_2_truth:
                    f.write("The ball bounces in player's 2 field on frame {} or at {} seconds\n".format(str(item), str(item*1/fps)))
                if item in bounce_player_1_truth:
                    f.write("The ball bounces in player's 1 field on frame {} or at {} seconds\n".format(str(item), str(item*1/fps)))

    # Compute errors
    error_player_1, not_detected_player_1 = computeError(player_1_crits_truth, player_1_critics)
    error_player_2, not_detected_player_2 = computeError(player_2_crits_truth, player_2_critics)
    error_bounce_1, not_detected_bounce_1 = computeError(bounce_player_1_truth, bounce_player_1)
    error_bounce_2, not_detected_bounce_2 = computeError(bounce_player_2_truth, crits_bounce_player_2)
    overall_error = (error_player_1 + error_player_2 + error_bounce_1 + error_bounce_2)/4
    events_not_detected = not_detected_player_1 + not_detected_player_2 + not_detected_bounce_1 + not_detected_bounce_2

    # Write errors to a file
    with open('errors'+str(vid)+'.txt', 'w') as f:
        f.write("Error for player's 1 hits: %s\n" % str(error_player_1))
        f.write("Player's 1 hits not detected: %s\n" % str(not_detected_player_1))
        f.write("Error for player's 2 hits: %s\n" % str(error_player_2))
        f.write("Player's 2 hits not detected: %s\n" % str(not_detected_player_2))
        f.write("Error for bounces in player's 1 side of the court: %s\n" % str(error_bounce_1))
        f.write("Bounces not detected in player's 1 side: %s\n" % str(not_detected_bounce_1))
        f.write("Error for bounces in player's 2 side of the court: %s\n" % str(error_bounce_2))
        f.write("Bounces not detected in player's 2 side: %s\n" % str(not_detected_bounce_2))
        f.write("Overall error: %s\n" % str(overall_error))
        f.write("Events not detected: %s\n" % str(events_not_detected))
    f.close()


    write_video(frames, 'tennis_'+str(vid)+'_Output.mp4', fps)