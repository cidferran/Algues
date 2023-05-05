import cv2
import sys
import os
import glob
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

tracker = cv2.legacy.TrackerCSRT_create()

# Read video
# ubi_video = /home/cidferran/Documents/PEF2/ALGAS/Alexandrium+Parasito_10min.MP4 #Ferran
name_video = '\Alexandrium sano 6x.MP4'
ubi_video = r"C:\Users\gerir\OneDrive\Escriptori\Enginyeria_Física\3B\PEF2\pef2projecte\.venv" + name_video


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


deg = 1

v_time = np.zeros([deg+1,1])
vec_time = np.array([0])
time = 0

v_coo = np.zeros([deg,2])
v_coo = np.vstack([v_coo,coordinates0])
vec_coo = coordinates0

velocity=np.array([[0,0]])
speed = 0
angle = 0

vec_vel = velocity
vec_speed = np.array([[0]])
vec_angle = np.array([[0]])

print('time:',time)
print("coordinates: ", coordinates0)
print("v_coo=",v_coo,'\n ---------- ')

failure = np.array([[0]])

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
        try:
                tg = velocity[1]/velocity[0]
                angle = np.arctan(tg)

        except:
            angle = pi/2

        vec_vel = np.vstack([vec_vel, velocity])
        vec_speed = np.vstack([vec_speed, speed])
        vec_angle = np.vstack([vec_angle, angle])

    elif deg == 2:
        #Centered difference (Parabolic)
        # it calculates velocity for n-1 frames!!!
        if time == 1:
            pass
        else:
            velocity = 0.5*(v_coo[2,:]-v_coo[0,:])
            speed = np.sqrt(velocity.dot(velocity))
            try:
                tg = velocity[1]/velocity[0]
                angle = np.arctan(tg)

            except:
                angle = pi/2

            vec_vel = np.vstack([vec_vel, velocity])
            vec_speed = np.vstack([vec_speed, speed])
            vec_angle = np.vstack([vec_angle, angle])

    else:
        print("Incorrect velocity degree, deg = 1, 2")
    
    print('velocity=',velocity)
    print('speed=',speed)
    print('angle=',angle,'\n ---------- ')


    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else :
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        failure = np.vstack([failure, time])


    # Display tracker type on frame
    cv2.putText(frame, " Tracker: CSRT", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)

    # Display FPS on frame
    # cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
    cv2.putText(frame, "Frame: " + str(int(time)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break


if deg == 2:
    #To match time and coord vel and speed are 1 row shorter
    vec_time = vec_time[0:-1,:]
    vec_coo = vec_coo[0:-1,:] 



# Separate trajectories

failure_short=failure[0:-1]
steps=failure[1:]-failure[0:-1]  #steps between failure i and i-1
lenght=steps>20                 #chcek if there exist a trajectory between failures
length=lenght.astype(int)  #convert true/false to 0/1
length=np.nonzero(length)     
start=failure_short[length]     #trajectory start indices
finish=steps[length]+start      #trajectory finish indices



results_folder = "C:\\Users\\gerir\\OneDrive\\Escriptori\\Enginyeria_Física\\3B\\PEF2\\pef2projecte\\.venv\\Results"

## Delete folder contents before !!!
for ii in range(len(start)):
    head = "Time, x, y, Vx, Vy, Speed"
    t = vec_time[start[ii]:finish[ii]]
    coo = vec_coo[start[ii]:finish[ii]]
    vel = vec_vel[start[ii]:finish[ii]]
    spd = vec_speed[start[ii]:finish[ii]]
    angl = vec_angle[start[ii]:finish[ii]]


    data = np.column_stack([t, coo, vel, spd, angl])
    np.savetxt(results_folder+'\\Trajectory '+str(ii+1)+'.csv', data, header=head, delimiter=",")
    







# Option 2 to save data

"""
df_time = pd.DataFrame() #Create DataFrame(s)
df_x = pd.DataFrame()
df_y = pd.DataFrame()
df_vx = pd.DataFrame()
df_vy = pd.DataFrame()
df_speed = pd.DataFrame()
df_angle = pd.DataFrame()


    for ii in range(len(start)):
    dt=pd.DataFrame({'t'+str(ii) : vec_time[start[ii]:finish[ii],0]})
    df_time=pd.concat([df_time, dt], axis=1)

    #df_x['x'+str(ii)] = vec_coo[start[ii]:finish[ii],0]
    #df_y['y'+str(ii)] = vec_coo[start[ii]:finish[ii],1]
    #df_vx['vx'+str(ii)] = vec_vel[start[ii]:finish[ii],0]
    #df_vx['vy'+str(ii)] = vec_vel[start[ii]:finish[ii],1]
    #df_speed['speed'+str(ii)] = vec_speed[start[ii]:finish[ii]]
    #df_angle['angle'+str(ii)] = vec_angle[start[ii]:finish[ii]]


df_time.to_csv('t')

#df_x.to_csv('x')
#df_y.to_csv('y')
#df_vx.to_csv('vx')
#df_vy.to_csv('vy')
#df_speed.to_csv('speed')
#df_angle.to_csv('angle') """



width  = video.get(3)  # float `width`
height = video.get(4) 
print('Frame width:', width)
print('Frame height: ', height)

info=np.array([['Video', name_video],
          ['Frame width', str(width)],
          ['Frame height', str(height)],
          ['Number trajectories', str(len(start))]])


np.savetxt(results_folder + '\\INFO.csv', info, delimiter=',',fmt='%s')

#Save data

head = "Time, x, y, Vx, Vy, Speed"
data = np.column_stack([vec_time, vec_coo, vec_vel, vec_speed])
np.savetxt('output_data.csv', data, header=head, delimiter=",")
np.savetxt('failure.csv', failure, delimiter=',')


#Plot
plt.plot(vec_coo[:,0],vec_coo[:,1])
plt.gca().invert_yaxis()
plt.show()



        

