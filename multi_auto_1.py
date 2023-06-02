import sys
import cv2
import numpy as np
from random import randint
from datetime import datetime
from matplotlib import pyplot as plt
from la_velocita_function import *
from detection import detect_new

# Empty results folder beforehand

import os, shutil
# folder = "/home/cidferran/Documents/PEF2/ALGAS/results/"
folder = "C:\\Users\\gerir\\OneDrive\\Escriptori\\Enginyeria_Física\\3B\\PEF2\\pef2projecte\\.venv\\Results"
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))


# OPEN VIDEO

name_video = '\Alexandrium sano 6x.MP4'
# ubi_video = "/home/cidferran/Documents/PEF2/ALGAS/Alexandrium+Parasito_10min.MP4"  # Ferran
ubi_video = r"C:\Users\gerir\OneDrive\Escriptori\Enginyeria_Física\3B\PEF2\pef2projecte\.venv" + name_video


video = cv2.VideoCapture(ubi_video)


if not video.isOpened():        # Exit if video not opened.
    print("Could not open video")
    sys.exit()


ok, frame = video.read()    # Read first frame.
if not ok:
    print('Cannot read video file')
    sys.exit()

# SELECT BOXES

bboxes = []
trackers = []



for ii in range(4):
    tracker = cv2.legacy.TrackerCSRT_create()
    center = detect_new(frame, ii)
    bbox = (center[0]-20,center[1]-20,40,40)
    ok = tracker.init(frame, bbox)
    bboxes.append(bbox)
    trackers.append(tracker)



num_trackers = len(bboxes)

initial_bboxes = bboxes

trackerType = "CSRT"   # Specify the tracker type
 

# DEFINE DATA VECTORS


time = 0
vec_time = np.array([time])
vec_x = np.zeros([1,num_trackers])
vec_y = np.zeros([1,num_trackers])


for ii in range(len(bboxes)):
    bbox = bboxes[ii]
    coordinates = np.array([bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2])
    vec_x[0,ii] = coordinates[0]
    vec_y[0,ii] = coordinates[1]


print(vec_x[0,1])



# TRACK LOOP

while True:
    ok, frame = video.read()
    if not ok:
        break

    time += 1
    vec_time = np.vstack([vec_time,time])    

    vec_x = np.vstack([vec_x, np.zeros([1,num_trackers])])
    vec_y = np.vstack([vec_y, np.zeros([1,num_trackers])])   

    # get updated location of objects in subsequent frames

    for i, tracker in enumerate(trackers):

        ok, bbox = tracker.update(frame)

        if ok:
            coordinates = np.array([bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2])
            vec_x[-1,i] = coordinates[0]
            vec_y[-1,i] = coordinates[1]

            print(coordinates)

            # draw tracked objects
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

        else:
            center = detect_new(frame, i)
            bbox = (center[0]-20,center[1]-20,40,40)
            new_tracker = cv2.legacy.TrackerCSRT_create()
            ok =new_tracker.init(frame, bbox)
            trackers[i] = new_tracker


           
    # show frame
    
    cv2.putText(frame, "Frame: " + str(int(time)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    cv2.imshow('MultiTracker', frame)
    # quit on ESC button
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break


width  = video.get(3)  # float `width`
height = video.get(4) 
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print('Frame width:', width)
print('Frame height: ', height)

info=np.vstack(( ['Frame Width', width],\
                 ['Frame Height', height],\
                 ['Total Frames', total_frames],\
                 ['Date', str(datetime.now())],\
                 ['Video', ubi_video]))

# results_folder = "/home/cidferran/Documents/PEF2/ALGAS/results/" # Ferran
results_folder = "C:\\Users\\gerir\\OneDrive\\Escriptori\\Enginyeria_Física\\3B\\PEF2\\pef2projecte\\.venv\\Results\\"

np.savetxt(results_folder + "INFO.csv", info, delimiter=',', header = "Information, Value", fmt='%s', comments = '')

x_coordinates = np.column_stack([vec_time,vec_x])
y_coordinates = np.column_stack([vec_time,vec_y])

head = "TimeStamp"
for kk in range(num_trackers):
    head = head + ",Box"+str(kk)
    
    

np.savetxt(results_folder + "x_coordinates.csv", x_coordinates, delimiter=",", header=head, fmt = "%i", comments = '')
np.savetxt(results_folder + "y_coordinates.csv", y_coordinates, delimiter=",", header=head, fmt = "%i", comments = '')
