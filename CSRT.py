import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt

tracker = cv2.legacy.TrackerCSRT_create()

# Read video
# ubi_video = "/home/cidferran/Documents/PEF2/ALGAS/Alexandrium+Parasito_10min.MP4"
ubi_video = "/home/cidferran/Documents/PEF2/ALGAS/Alexandrium+Parasito_30sec.mp4"

# ubi_video = r"C:\Users\gerir\OneDrive\Escriptori\Enginyeria_Física\3B\PEF2\pef2projecte\.venv\Alexandrium sano 6x.MP4"
video = cv2.VideoCapture(ubi_video)

# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

# Read first frame.
ok, frame = video.read()
if not ok:
    print('Cannot read video file')
    sys.exit()

# # Define an initial bounding box
# bbox = (287, 23, 86, 320)

# # Uncomment the line below to select a different bounding box
bbox = cv2.selectROI(frame, False)
coordinates0 = np.array([bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2])
print(bbox) # prints initial coordinates of left lower corner, width, height.

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)


deg = 2

v_time = np.zeros([deg+1,1])
vec_time = np.array([[0]])
time = 0

v_coo = np.zeros([deg,2])
v_coo = np.vstack([v_coo,coordinates0])
vec_coo = coordinates0

velocity=np.array([[0,0]])
speed = 0
vec_vel = velocity
vec_speed = np.array([[0]])

print('time:',time)
print("coordinates: ", coordinates0)
print("v_coo=",v_coo,'\n ---------- ')


while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)
 
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    

    #Time
    time +=1
    v_time = np.vstack([v_time,time])
    v_time = v_time[deg+1-deg:deg+1,:]
    vec_time= np.vstack([vec_time,time])
    print('time=',time)
    
    # Position
    coordinates = np.array([bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2])
    
    v_coo = np.vstack([v_coo,coordinates])
    v_coo = np.delete(v_coo,0,0)  
    vec_coo = np.vstack([vec_coo,coordinates])
    print("coordinates: ", coordinates)


    #Velocity pixels/framestep
    if deg == 1:
        #Backward difference    
        velocity = v_coo[1,:]-v_coo[0,:]
        speed = np.sqrt(velocity.dot(velocity)) 
        vec_vel = np.vstack([vec_vel, velocity])
        vec_speed = np.vstack([vec_speed, speed])

    elif deg == 2:
        #Centered difference (Parabolic)
        # it calculates velocity for n-1 !!!
        if time == 1:
            pass
        else:
            velocity = 0.5*(v_coo[2,:]-v_coo[0,:])
            speed = np.sqrt(velocity.dot(velocity)) 
            vec_vel = np.vstack([vec_vel, velocity])
            vec_speed = np.vstack([vec_speed, speed])
    
    print('velocity=',velocity)
    print('speed=',speed,'\n ---------- ')

    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else :
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Display tracker type on frame
    cv2.putText(frame, " Tracker: CSRT", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break



vec_time = vec_time[0:-1,:]
vec_coo = vec_coo[0:-1,:] # when track is lost [[0, 0]] is addedd so remove
vec_vel = vec_vel[0:-1,:]

if deg == 2:
    vec_vel = np.vstack([vec_vel, [0,0]]) # deg 2 cant calculate last v


width  = video.get(3)  # float `width`
height = video.get(4) 
print('Frame width:', width)
print('Frame height: ', height)

#Save data
header = "Time, x, y, Vx, Vy, Speed"
data = np.column_stack([vec_time, vec_coo, vec_vel, vec_speed])
# np.savetxt('output_data.csv', data, header=header, delimiter=",")
np.savetxt("/home/cidferran/Documents/PEF2/output_data.csv", data, header=header, delimiter=",")


#Plot
plt.plot(vec_coo[:,0],vec_coo[:,1])
plt.gca().invert_yaxis()
plt.show()