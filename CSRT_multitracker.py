import sys
import cv2
import numpy as np
from random import randint
from matplotlib import pyplot as plt
from la_velocita_function import *

# Empty results folder beforehand
import os, shutil
folder = "/home/cidferran/Documents/PEF2/ALGAS/results/"
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
# ubi_video = r"C:\Users\gerir\OneDrive\Escriptori\Enginyeria_Física\3B\PEF2\pef2projecte\.venv" + name_video
ubi_video = "/home/cidferran/Documents/PEF2/ALGAS/Alexandrium+Parasito_10min.MP4"  # Ferran

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
colors = [] 

while True:
    # draw bounding boxes over objects
    cv2.putText(frame, "Press ENTER TWICE to add bbox", (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
    cv2.putText(frame, "Press ENTER and ESC to add the last", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

    bbox = cv2.selectROI('MultiTracker', frame, False)
    bboxes.append(bbox)
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))

    k = cv2.waitKey(0) & 0xFF
    
    if (k == 27):  # ESC is pressed
        break

num_trackers = len(bboxes)

initial_bboxes = bboxes

trackerType = "CSRT"   # Specify the tracker type
 

multiTracker = cv2.legacy.MultiTracker_create() # Create MultiTracker object
 

for bbox in bboxes:            # Initialize MultiTracker
  tracker = cv2.legacy.TrackerCSRT_create()
  multiTracker.add(tracker, frame, bbox)


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
    
    # get updated location of objects in subsequent frames
    ok, boxes = multiTracker.update(frame)
    
    time += 1
    vec_time = np.vstack([vec_time,time])

    # draw tracked objects
    
    vec_x = np.vstack([vec_x, np.zeros([1,num_trackers])])
    vec_y = np.vstack([vec_y, np.zeros([1,num_trackers])])

    print(vec_x)

    for i, bbox in enumerate(boxes):
        coordinates = np.array([bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2])
        vec_x[-1,i] = coordinates[0]
        vec_y[-1,i] = coordinates[1]

        if bbox.all() == 0:
            bboxes[i] = initial_bboxes[i]
       
        
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

    
    
    # show frame
    cv2.imshow('MultiTracker', frame)
    
    # quit on ESC button
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break


    print(vec_time)
    print(vec_x)

width  = video.get(3)  # float `width`
height = video.get(4) 
print('Frame width:', width)
print('Frame height: ', height)

info=np.vstack(( ['Frame Width', width], ['Frame Height', height]))

results_folder = "/home/cidferran/Documents/PEF2/ALGAS/results/" # Ferran

np.savetxt(results_folder + "INFO.csv", info, delimiter=',', header = "Information, Value", fmt='%s', comments = '')

x_coordinates = np.column_stack([vec_time,vec_x])
y_coordinates = np.column_stack([vec_time,vec_y])

head = "TimeStamp"
for kk in range(num_trackers):
    head = head + ",Box"+str(kk)
    
    

np.savetxt(results_folder + "x_coordinates.csv", x_coordinates, delimiter=",", header=head, fmt = "%i", comments = '')
np.savetxt(results_folder + "y_coordinates.csv", y_coordinates, delimiter=",", header=head, fmt = "%i", comments = '')

#Plot
plt.plot(vec_x[:,0],vec_y[:,0])
plt.gca().invert_yaxis()
plt.show()