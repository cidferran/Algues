import cv2
import numpy as np




def detect_new(image, region):
    # THRESHOLD
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    _, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY) # Apply thresholding to create a binary image

    #DETECTION
    params = cv2.SimpleBlobDetector_Params() # Set up the detector with default parameters.
    params.filterByArea = True
    params.minArea = 70
    detector = cv2.SimpleBlobDetector.create(params)
    keypoints = detector.detect(threshold)  # Detect blobs

    # REGION BOUNDARIES
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


    return largest_blob.pt




