import cv2
import utils
import random
import routes
import os, os.path

from SignalInput import SignalInput

from time import sleep
from datetime import datetime, timedelta
from utils import printProgressBar, binarySearch
from math import pi, log, log2, cos, sin

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
        if i != 0 and i != dataset['timestamp'].count():
            
            dt_before       = dataset.iloc[i, 33]

            dt_after        = dataset.iloc[i+1, 33]
            increment += (float(dt_after) - float(dt_before))/1000
    
    final_sample_rate       = increment / (dataset['timestamp'].count())
    print(final_sample_rate)

    frame_by_sample_rate    = final_sample_rate * framerate

    print(int(frame_by_sample_rate)) 
    
    return frame_by_sample_rate, final_sample_rate

def calculate_new_tax_by_frames(sample_rate, fps_in, fps_out):
    return (sample_rate * fps_out / fps_in)

def sintectic_generation(size):
    file        = open(routes.data_sintectic, 'w')
    
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


# Interpollation functions

def converse_to_grayscale(image_route_in, image_route_out, image_folder_in):

    number_of_files = len([name for name in os.listdir(image_folder_in)])
    for i in range(number_of_files):
        img = cv2.imread(image_route_in.format(i),cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(image_route_out.format(i), img)

def generate_signal_input(image_route, image_folder, fps):
    number_of_files = len([name for name in os.listdir(image_folder)])
    signal_input    = []

    for i in range(number_of_files):
        img_vector  = cv2.imread(image_route.format(i))        
                
        signal_input.append(SignalInput(29, img_vector, i))
        
    return signal_input

def interpollation(input_slice, interval, start_time_stamp):
    # input_slice is a the frames per second originially
    # interval is the final transformation from the input frames

    list_non_uniform_t_series = input_slice
    dt = 0.0

    yi = []
    time_interp = list_non_uniform_t_series[0].timestamp if start_time_stamp == 0 else start_time_stamp
    

    while time_interp <= list_non_uniform_t_series[len(list_non_uniform_t_series) - 1].timestamp:
        
        was_here, loc = binarySearch(list_non_uniform_t_series, SignalInput(None, [], time_interp))
        
        components      = [i for i in range(len(list_non_uniform_t_series[0].frames_vector) - 1)]
        dy              = components.copy()
        slope           = components.copy()
        intercept       = components.copy()

        if not was_here and loc < len(list_non_uniform_t_series) - 1:
            dt = list_non_uniform_t_series[loc + 1].timestamp - list_non_uniform_t_series[loc].timestamp
            
            for i in range(len(components)):
                dy[i]           = list_non_uniform_t_series[loc + 1].frames_vector[i] - list_non_uniform_t_series[loc].frames_vector[i]
                slope[i]        = dy[i] / dt
                intercept[i]    = list_non_uniform_t_series[loc + 1].frames_vector[i] - list_non_uniform_t_series[loc].timestamp * slope[i]
                components[i]   = slope[i] * time_interp + intercept[i] 
        else:
            components = list_non_uniform_t_series[loc].frames_vector

        yi.append(SignalInput(interval, components, time_interp))
        time_interp += interval

    return yi

def compute_FFT(frame):
    number_points   = len(frame)
    real            = []
    img             = []
    
    real            = [[i] for i in range(number_points)]
    img             = [[i] for i in range(number_points)]
    
    real = frame
    fft(number_points)
    
def fft(number_points, imag, real):
    if number_points == 1:
        return
    
    num_stages          = int(log(number_points) / log(2))
    half_num_points     = number_points >> 1
    j                   = half_num_points
    k                   = 0
    
    for i in range(1, number_points - 2):
        if (i < j):
            temp_real   = real[j]
            temp_imag   = imag[j]
            
            real[j]     = real[i]
            imag[j]     = imag[i]
            
            real[i]     = temp_real
            imag[j]     = temp_imag
            
        k = half_num_points
        
        while(k <= j):
            j -= k
            k >>= 1
            
        j += k
    

    for stage in range(1, num_stages + 1):
        LE = 1
        
        for i in range(stage):
            LE <<= 1
            
        LE2 = LE >> 1
        UR  = 1
        UI  = 0

        SR  = cos(pi / LE2)
        SI  = -(sin(pi / LE2))

        for subDFT in range(1, LE2):
            for butterfly in range(subDFT - 1, number_points):
                ip = butterfly + LE2

                temp_real = double(real[ip] * UR - imag[ip] * UI)
                temp_imag = double(real[ip] * UI + imag[ip] * UR)

                real[ip] = real[butterfly] - temp_real
                imag[ip] = imag[butterfly] - temp_imag

                real[butterfly] += temp_real
                imag[butterfly] += temp_imag

            temp_UR = UR
            
            UR      = temp_UR * SR - UI * SI
            UI      = temp_UR * SI + UI * SR

    to_string(real, imag)

def to_string(real, imag):
    values = ""

    for i in range(len(real)):
        values += "[{} , {};] ".format(int(real[i] * 1000) / 1000.0, 
                                        int(imag[i] * 1000) / 1000.0)  

if __name__ == '__main__':
#    frames_by_sample, sample_interval = labeling_by_syncro_frame(29, routes.data_sintectic)
#    
#    file            = open(routes.images_interpollated_data, 'w') 
#    
#    signal_input    = generate_signal_input(routes.images_gray_route, routes.images_gray_folder, 29)
    interval        = calculate_new_tax_by_frames(sample_interval, 29, 15)
    
    
    signal_output   = interpollation(signal_input, interval, 0)
    
    for signal in signal_output:
        file.write("{}\n".format(signal.frames_vector))
    
    

#    for signal in signal_input:
#        print("{}, {}, {}".format(signal.timestamp, signal.frames_per_second, signal.frames_vector))

# Initial call to print 0% progress
"""
printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
for i, item in enumerate(items):
    # Do stuff...
    sleep(0.1)
    # Update Progress Bar
    printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
"""