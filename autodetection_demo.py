import cv2
import numpy as np

#OPEN IMAGE
name_video = '\Alexandrium sano 6x.MP4'
ubi_video = r"C:\Users\gerir\OneDrive\Escriptori\Enginyeria_Física\3B\PEF2\pef2projecte\.venv" + name_video
# ubi_video = "/home/cidferran/Documents/PEF2/ALGAS/Alexandrium+Parasito_10min.MP4"  # Ferran
video = cv2.VideoCapture(ubi_video)
ok, image = video.read() 


# THRESHOLD
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
blur = cv2.blur(gray,(10,10))
_, threshold = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY) # Apply thresholding to create a binary image


#DETECTION
params = cv2.SimpleBlobDetector_Params() # Set up the detector with default parameters.
params.filterByArea = True
params.minArea = 70
detector = cv2.SimpleBlobDetector.create(params)
keypoints = detector.detect(threshold)  # Detect blobs


# REGION BOUNDARIES
region = 0
height, width,_ = image.shape
if region == 0: 
    reg_min_x=1
    reg_max_x=width/2
    reg_min_y=1
    reg_max_y=height/2
elif region == 1:
    reg_min_x=width/2
    reg_max_x=width
    reg_min_y=1
    reg_max_y=height/2
elif region == 2: 
    reg_min_x=1
    reg_max_x=width/2
    reg_min_y=height/2
    reg_max_y=height
else:
    reg_min_x=width/2
    reg_max_x=width
    reg_min_y=height/2
    reg_max_y=height


# FIND LARGEST IN REGION
largest_blob_size=0
for kp in keypoints:
    if reg_min_x <= kp.pt[0] <= reg_max_x and reg_min_y <= kp.pt[1] <= reg_max_y: #blob in target region?
        if kp.size>largest_blob_size:
            largest_blob=kp
            x = int(kp.pt[0])
            y = int(kp.pt[1])
    else: pass


# DISPLAY IMAGES
radius_largest = round(largest_blob.size/2)
print(largest_blob.pt)
center_largest=(round(largest_blob.pt[0]),round(largest_blob.pt[1]))
im_with_largest = cv2.circle(image, center_largest, radius_largest, (255,0,0))
cv2.imshow("largest", im_with_largest)

im_with_keypoints = cv2.drawKeypoints(threshold, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
cv2.imshow("All blobs", im_with_keypoints)

cv2.imshow("Threshold", threshold)

cv2.imshow("Blurred", blur)

cv2.imshow("Gray", gray)

while True:
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        cv2.destroyAllWindows()
        break





