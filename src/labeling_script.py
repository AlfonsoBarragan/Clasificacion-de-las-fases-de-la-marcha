import cv2
import utils
import random
import routes
import os, os.path


from time import sleep
from datetime import datetime, timedelta
from utils import printProgressBar

def split_in_frames(route_in, frame_rate, route_out):
    vidcap = cv2.VideoCapture(route_in)
    success,image = vidcap.read()
    count = 0
    
    while success:
        if count % frame_rate == 0:
            cv2.imwrite("{}/frame{}.jpg".format(route_out,count), image)     # save frame as JPEG file      
        
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

def labeling_by_syncro_frame(framerate, dataset_route):
    
    dataset                 = utils.read_dataset(dataset_route)

    dataset['frames']           = 0
    increment                   = 0

    # In order to calculate, the number of frames that every sample
    # has assigned, we should compute the sample rate of capture

    for i in range(dataset['timestamp'].count() - 1):
        if i != 0 and i != dataset.count:
            """
            dt_before = datetime.strptime(  dataset.iloc[i, 32],
                                            '%Y-%m-%d %H:%M:%S;%f')
            
            dt_after  = datetime.strptime(  dataset.iloc[i+1, 32],
                                            '%Y-%m-%d %H:%M:%S;%f')
            
            dt_before_mili = dt_before.timestamp() * 1000000
            dt_after_mili = dt_before.timestamp() * 1000000
            """
            
            dt_before       = dataset.iloc[i, 32].split(';')
            dt_before_mu    = dt_before[1]

            dt_after        = dataset.iloc[i+1, 32].split(';')
            dt_after_mu     = dt_after[1]

            increment += (float(dt_after_mu) - float(dt_before_mu))/1000
    
    final_sample_rate       = increment / (dataset['timestamp'].count())
    print(final_sample_rate)

    frame_by_sample_rate    = final_sample_rate * framerate

    print(int(frame_by_sample_rate)) 

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

    increment   = 0
    total_size  = 0

    for i in range(size):
        
        if total_size % 100 == 0:
            increment += 1
        
        ts = datetime.timestamp(datetime.now() + timedelta(seconds=increment))
        
        st = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

        mu_seconds = datetime.fromtimestamp(ts).strftime('%f')
        
        for i in range(32):
            if i != 31:
                file.write("{},".format(random.randint(0,1024)))
            else:
                file.write("{}".format(random.randint(0,1024)))
                
        file.write("{},{},{}\n".format(ts,st,mu_seconds))
        total_size += 1

def converse_to_grayscale(image_route, image_folder):

    number_of_files = len([name for name in os.listdir(image_folder)])
    for i in range(number_of_files):
        img = cv2.imread(image_route.format(i),cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(image_route.format('_gray'+str(i)), img)

if __name__ == '__main__':
    labeling_by_syncro_frame(120, routes.data_sintectic)


# Initial call to print 0% progress
"""
printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
for i, item in enumerate(items):
    # Do stuff...
    sleep(0.1)
    # Update Progress Bar
    printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
"""