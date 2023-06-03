import sys
import cv2
import numpy as np
import os, shutil
from datetime import datetime
from la_velocita_function import *
from detection_1 import detect_new

### --- SETTING --- ###

# OPEN VIDEO
# ubi_video = "/home/cidferran/Documents/PEF2/ALGAS/Alexandrium+Parasito_10min.MP4"  # Ferran
name_video = '\Alexandrium sano 6x.MP4'
ubi_video = r"C:\Users\gerir\OneDrive\Escriptori\Enginyeria_Física\3B\PEF2\pef2projecte\.venv" + name_video
video = cv2.VideoCapture(ubi_video)

if not video.isOpened():        # Exit if video not opened.
    print("Could not open video")
    sys.exit()

ok, frame = video.read()    # Read first frame.
if not ok:
    print('Cannot read video file')
    sys.exit()


# DEFINE DATA VECTORS
num_trackers=4
frames_not_moving=np.zeros([num_trackers])

time = 0
vec_time = np.array([time])
vec_x = np.zeros([1,num_trackers])
vec_y = np.zeros([1,num_trackers])



# SELECT BOXES
bboxes = []
trackers = []
for ii in range(4):
    center = detect_new(frame, ii)
    bbox = (center[0]-20,center[1]-20,40,40)
    vec_x[0,ii] = center[0]
    vec_y[0,ii] = center[1]

    tracker = cv2.legacy.TrackerCSRT_create()
    ok = tracker.init(frame, bbox)
    
    bboxes.append(bbox)
    trackers.append(tracker)


### --- TRACK LOOP --- ###
while True:
    ok, frame = video.read()    #get new frame
    if not ok:
        break
    
    time += 1
    vec_time = np.vstack([vec_time,time])   #Store new time


    vec_x = np.vstack([vec_x, np.zeros([1,num_trackers])])  #create new row for x's
    vec_y = np.vstack([vec_y, np.zeros([1,num_trackers])])  #create new row for y's     

    for i, tracker in enumerate(trackers):  #iterate each tracker

        ok, bbox = tracker.update(frame)    #update tracker

        if ok and frames_not_moving[i]<20:
            #store new x and y
            coordinates = np.array([bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2])
            vec_x[-1,i] = coordinates[0]
            vec_y[-1,i] = coordinates[1]

            # check if it has stopped
            if abs(coordinates[0]-vec_x[-2,i])<5 and abs(coordinates[1]-vec_y[-2,i])<5:
                frames_not_moving[i] += 1
            else:
                frames_not_moving[i] = 0

        else:       # if tracker lost or cell not moving
            # find new cell and set bbox            
            center = detect_new(frame, i)
            bbox = (center[0]-20,center[1]-20,40,40)
            new_tracker = cv2.legacy.TrackerCSRT_create()
            ok = new_tracker.init(frame, bbox)
            trackers[i] = new_tracker

            # store x and y
            coordinates = np.array([bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2])
            vec_x[-1,i] = coordinates[0]
            vec_y[-1,i] = coordinates[1]

            frames_not_moving[i] = 0

        # draw tracked objects
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
          
    # show frame    
    cv2.putText(frame, "Frame: " + str(int(time)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    cv2.imshow('MultiTracker', frame)

    # exit
    if cv2.waitKey(1) & 0xFF == 27:  # when Esc pressed
        break


### --- SAVE DATA --- ###

# SELECT RESULTS FOLDER

# results_folder = "/home/cidferran/Documents/PEF2/ALGAS/results/"
results_folder = "C:\\Users\\gerir\\OneDrive\\Escriptori\\Enginyeria_Física\\3B\\PEF2\\pef2projecte\\.venv\\Results\\"

# clear the folder beforehand
for filename in os.listdir(results_folder):         
    file_path = os.path.join(results_folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))


# SAVE INFO
width  = video.get(3)  # float `width`
height = video.get(4) 
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

info=np.vstack(( ['Frame Width', width],\
                 ['Frame Height', height],\
                 ['Total Frames', total_frames],\
                 ['Date', str(datetime.now())],\
                 ['Video', ubi_video]))

np.savetxt(results_folder + "INFO.csv", info, delimiter=',', header = "Information, Value", fmt='%s', comments = '')


# SAVE DATA
x_coordinates = np.column_stack([vec_time,vec_x])
y_coordinates = np.column_stack([vec_time,vec_y])
head = "TimeStamp"
for kk in range(num_trackers):
    head = head + ",Box"+str(kk)  

np.savetxt(results_folder + "x_coordinates.csv", x_coordinates, delimiter=",", header=head, fmt = "%i", comments = '')
np.savetxt(results_folder + "y_coordinates.csv", y_coordinates, delimiter=",", header=head, fmt = "%i", comments = '')
