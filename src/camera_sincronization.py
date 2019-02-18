# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:09:56 2019

@author: Alf
"""
"""
from cv2 import *

namedWindow("webcam")
vc = VideoCapture(0);

while True:
    next, frame = vc.read()

    gray = cvtColor(frame, COLOR_BGR2GRAY)
    gauss = GaussianBlur(gray, (7,7), 1.5, 1.5)
    can = Canny(gauss, 0, 30, 3)

    imshow("webcam", can)
    
    if waitKey(50) >= 0:
        break
"""

import routes
import cv2
import time
import random
import datetime

def capture_from_webcam():
    cap = cv2.VideoCapture(0)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret==True:
            #frame = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7,7), 1.5, 1.5), 0, 30, 3)
            
            # write the flipped frame
            out.write(frame)
    
            cv2.imshow('Realtime Processing',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
def capture_from_a_video(route):
    
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(route)
     
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")
     
    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True:
     
        # Display the resulting frame
        cv2.imshow('Frame',frame)
     
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
     
      # Break the loop
      else: 
        break
     
    # When everything done, release the video capture object
    cap.release()
     
    # Closes all the frames
    cv2.destroyAllWindows()
        
def sintectic_generation(size):
    file        = open(routes.data_sintectic, 'w')
    actual_size = 0    
    
    file.write("sensor_value1,sensor_value2,sensor_value3,sensor_value4,"+
               "sensor_value5,sensor_value6,sensor_value7,sensor_value8,"+
               "sensor_value9,sensor_value10,sensor_value11,sensor_value12,"+
               "sensor_value13,sensor_value14,sensor_value15,sensor_value16,"+
               "sensor_value17,sensor_value18,sensor_value19,sensor_value20,"+
               "sensor_value21,sensor_value22,sensor_value23,sensor_value24,"+
               "sensor_value25,sensor_value26,sensor_value27,sensor_value28,"+
               "sensor_value29,sensor_value30,sensor_value31,sensor_value32,"+
               "timestamp,date_format_timestamp\n")
    
    while actual_size < size:
        
        if actual_size % 100 == 0:
            time.sleep(1)
            
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        
        for i in range(32):
            if i != 31:
                file.write("{},".format(random.randint(0,1024)))
            else:
                file.write("{}".format(random.randint(0,1024)))
                
        file.write("{},{}\n".format(ts,st))
        
        actual_size += 1

capture_from_webcam()


"""
import bluetooth
def bluetooth_finder(target_name):
    target_address = None
    
    nearby_devices = bluetooth.discover_devices()
    
    for bdaddr in nearby_devices:
        if target_name == bluetooth.lookup_name( bdaddr ):
            target_address = bdaddr
            break
    
    if target_address is not None:
        print ("found target bluetooth device with address "+ target_address)
    else:
        print ("could not find target bluetooth device nearby")
"""


        
    
        
