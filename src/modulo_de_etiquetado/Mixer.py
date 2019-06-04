# External libraries
import cv2
import threading
import time
import os
import sys, getopt
import struct
import pandas as pd
from functools import reduce

# Internal libraries

from modulo_de_funciones_de_soporte import utils
from modulo_de_funciones_de_soporte import routes

def throw_observer(path, time_to_collect, display):
    # Collect data
    os.system("python3 Observer.py -p {} -t {} -d {}".format(path, time_to_collect, display))

def throw_stomper(path, time_to_collect):
    # Collect data
    os.system("python3 Stomper.py -p {} -t {} ".format(path, time_to_collect))
    
    # Clean the whites in data
    os.system("python3 Stomper.py -p {} -c".format(path))

def see_and_stomp(time_to_recollect, path, display=-1):
    # Create two threads as follows
    thread_obs = threading.Thread(target=throw_observer, args=[path,time_to_recollect,display])
    thread_sto = threading.Thread(target=throw_stomper, args=[path,time_to_recollect])

    # Start to execute the threads, and wait until they stops
    thread_obs.start()
    thread_sto.start()

def assign_entry_point_and_transform_datasets(dataframe):
    # Locate a point where it completes a full read of the sensors, and remove from it to the first in order
    # to begin in a common point

    dataframe_aux = dataframe[(dataframe.Value_17 == '-1')]
    index_entry = int(dataframe_aux.iloc[2].name)
    index_exit  = int(dataframe_aux.last_valid_index()) - 3

    new_dataframe = dataframe[(dataframe.index > index_entry) & 
                              (dataframe.index < index_exit)]

    return new_dataframe

def converse_hex_plantar_pressure(dataframe_insole, insole_id, path):
    # Open a file to write
    samples_file_def = open(path, 'w')
    
    # Write the data attributes
    samples_file_def.write("Timestamp_init,Timestamp_end,Date_init,"    +
                            "Date_end,Source,Sensor_1,Sensor_2,"        +
                            "Sensor_3,Sensor_4,Sensor_5,Sensor_6,"      +
                            "Sensor_7,Sensor_8,Sensor_9,Sensor_10,"     +
                            "Sensor_11,Sensor_12,Sensor_13,Sensor_14,"  +
                            "Sensor_15,Sensor_16,Sensor_17,Sensor_18,"  +
                            "Sensor_19,Sensor_20,Sensor_21,Sensor_22,"  +
                            "Sensor_23,Sensor_24,Sensor_25,Sensor_26,"  +
                            "Sensor_27,Sensor_28,Sensor_29,Sensor_30,"  +
                            "Sensor_31,Sensor_32\n")
    
    # Parameters for progress bar
    size = len(dataframe_insole)
    
    utils.printProgressBar(0, size, prefix = 'Progress:', suffix = 'Complete', length = 50)

    for i in range(0, size, 4):
                
        mux_1 = []
        mux_2 = []
        mux_3 = []
        mux_4 = []
        
        try:
            sample_1 = dataframe_insole.iloc[i]
            sample_2 = dataframe_insole.iloc[i + 1]
            sample_3 = dataframe_insole.iloc[i + 2]
            sample_4 = dataframe_insole.iloc[i + 3]
        
        except Exception:
            break
        
        mux_1.extend(sample_1[3:19])
        mux_2.extend(sample_2[3:19])
        mux_3.extend(sample_3[3:19])
        
        mux_4.extend(sample_1[19:])
        mux_4.extend(sample_2[19:])
        mux_4.extend(sample_3[19:])
        mux_4.extend(sample_4[15:19])
        
        mux_1 = reduce((lambda x, y: x + y), list(map(lambda x:x[2:] , mux_1))) 
        mux_2 = reduce((lambda x, y: x + y), list(map(lambda x:x[2:] , mux_2)))
        mux_3 = reduce((lambda x, y: x + y), list(map(lambda x:x[2:] , mux_3)))
        mux_4 = reduce((lambda x, y: x + y), list(map(lambda x:x[2:] , mux_4)))
        
        mux_1_values = struct.unpack("!HHHHHHHH", bytes.fromhex(mux_1))
        mux_2_values = struct.unpack("!HHHHHHHH", bytes.fromhex(mux_2))
        mux_3_values = struct.unpack("!HHHHHHHH", bytes.fromhex(mux_3))
        
        mux_4_values        = struct.unpack("!HHHHHH", bytes.fromhex(mux_4[:24]))
        mux_4_values_rest   = struct.unpack("<HH", bytes.fromhex(mux_4[24:]))
        mux_4_values        += mux_4_values_rest
        
        # Se escribe los timestamps, dates y source en el fichero
        samples_file_def.write("{},{},{},{},{}".format(sample_1.Timestamp,
                               sample_4.Timestamp, sample_1.Date, sample_4.Date,
                               sample_1.Source))
        
        # Se escribe el valor de los sensores
        if insole_id == 0:
            for values in [mux_1_values, mux_2_values, mux_3_values, mux_4_values]:
                samples_file_def.write(",{},{},{},{},{},{},{},{}".format(values[7],
                                        values[6],values[5],values[4],values[3],values[2],
                                        values[1],values[0]))
        else:
            for values in [mux_1_values, mux_2_values, mux_3_values, mux_4_values]:
                samples_file_def.write(",{},{},{},{},{},{},{},{}".format(values[0],
                                        values[1],values[2],values[3],values[4],values[5],
                                        values[6],values[7]))
        # Se introduce un salto de linea para continuar el ciclo de escritura
        samples_file_def.write("\n")
        
        utils.printProgressBar(i, size, prefix = 'Progress:', suffix = 'Complete', length = 50)

        
def repare_samples_dataset(df):
    
    counter = 0
    dataframe = df.copy()
    data_series_last_mux_4 = []
    
    # Parameters for progress bar
    size  = len(dataframe)
    
    utils.printProgressBar(0, size, prefix = 'Progress:', suffix = 'Complete', length = 50)

    for i in range(size):
        counter += 1
        if counter % 4 == 0 and len(dataframe) > i + 2:
            if dataframe.iloc[i].Value_17 != '-1':
                
                if dataframe.iloc[i+1].Value_17 == '-1' and dataframe.iloc[i+2].Value_17 != '-1':
                    # Borrar el siguiente
                    dataframe = dataframe.drop([dataframe.index[i]], axis=0)
                    counter = 0
                
                elif dataframe.iloc[i+2].Value_17 == '-1':
                    # Borrar los dos siguientes
                    dataframe = dataframe.drop([dataframe.index[i],dataframe.index[i+1]], axis=0)
                    counter = 0
                    
                else:
                    dataframe = utils.insert_row_in_pos(i, data_series_last_mux_4, dataframe)
                    counter = 0
                
            else:
                data_series_last_mux_4 = dataframe.iloc[i]
                counter = 0
                
        utils.printProgressBar(i, size, prefix = 'Progress:', suffix = 'Complete', length = 50)

    return dataframe

def mix_sources_of_data():
    full_data_cleaned   = utils.read_dataset("{}/{}".format(routes.data_directory,
                                                             routes.samples_cleaned_uni))
    
    full_frame_data     = utils.read_dataset("{}/{}".format(routes.data_directory,
                                                             routes.frames_dataset))
    
    sources = full_data_cleaned.Source.unique()
    
    id_handle_insoleL = sources[0]
    id_handle_insoleR = sources[1]
    
    # Divide datasets by insole
    insoleL_dataset = full_data_cleaned[(full_data_cleaned.Source == id_handle_insoleL)]
    insoleR_dataset = full_data_cleaned[(full_data_cleaned.Source == id_handle_insoleR)]

    # Eligue un buen punto inicial omitiendo algunas muestras para comenzar la traducción
    # de hexadecimal a datos de presión plantar
    insoleL_dataset = assign_entry_point_and_transform_datasets(insoleL_dataset)
    insoleR_dataset = assign_entry_point_and_transform_datasets(insoleL_dataset)
    
    # Se inserta una muestra para compensar la perdida de pequeñas muestras del multiplexor 4
    # se requiere dar dos pasadas por dataset para reparar de la mejor manera.
    insoleL_dataset = repare_samples_dataset(insoleL_dataset)
    insoleL_dataset = repare_samples_dataset(insoleL_dataset)

    insoleR_dataset = repare_samples_dataset(insoleR_dataset)
    insoleR_dataset = repare_samples_dataset(insoleR_dataset)
    
    # Se vuelve a recortar el numero de registros para evitar problemas de falta de muestras
    insoleL_dataset = assign_entry_point_and_transform_datasets(insoleL_dataset)
    insoleR_dataset = assign_entry_point_and_transform_datasets(insoleL_dataset)
    
    converse_hex_plantar_pressure(insoleL_dataset, 0, "{}/{}".format(routes.data_directory, routes.samples_full_l))
    converse_hex_plantar_pressure(insoleR_dataset, 1, "{}/{}".format(routes.data_directory, routes.samples_full_r))

def assign_frame_to_sample(df_frames, df_samples):

    df_samples_first_timestamp = df_samples.iloc[0].Timestamp_init 
    df_samples_last_timestamp  = df_samples.iloc[len(df_samples) - 1].Timestamp_init 
    
    df_frames_crop = df_frames[(df_frames.Timestamp >= df_samples_first_timestamp) & 
                               (df_frames.Timestamp <= df_samples_last_timestamp) ]    

    df_samples['Frame'] = '0'
    
    # Parameters for progress bar
    size  = len(df_samples)
    
    utils.printProgressBar(0, size, prefix = 'Progress:', suffix = 'Complete', length = 50)

    
    for i in range(len(df_samples)):
        sample = df_samples.iloc[i]
        
        df_frames_in_this_sample = df_frames_crop.copy()
        
        if len(df_frames_in_this_sample) == 0:
            df_samples.loc[i, 'Frame'] = str(int(df_samples.iloc[i-1].Frame) + 1)
            
        elif len(df_frames_in_this_sample) == 1:
            df_samples.loc[i, 'Frame'] = df_frames_in_this_sample.Id_frame
        
        else:
            df_frames_in_this_sample['Diff'] = df_frames_in_this_sample['Timestamp'].apply(lambda x: abs(x-sample.Timestamp_init))
            frame_selected = df_frames_in_this_sample[(df_frames_in_this_sample.Diff == df_frames_in_this_sample['Diff'].min())]
            df_samples.loc[i, 'Frame'] = frame_selected.iloc[0].Id_frame

        utils.printProgressBar(i, size, prefix = 'Progress:', suffix = 'Complete', length = 50)

def create_directories_estructure(subject, number_exp):
    actual_directories = utils.ls(routes.data_directory)
    
    if subject not in actual_directories:
        os.system("mkdir {}/{}".format(routes.data_directory, subject))
        os.system("mkdir {}/{}/{}".format(routes.data_directory, subject, number_exp))
        os.system("mkdir {0}/{1}/{2}/{3} && mkdir {0}/{1}/{2}/{4}".format(routes.data_directory, subject, number_exp, routes.data_directory, routes.frames_directory))

    else:
        inside_subject = utils.ls("{}/{}".format(routes.data_directory, subject))        
        
        while number_exp in inside_subject:
            number_exp += "1"

        os.system("mkdir {}/{}/{}".format(routes.data_directory, subject, number_exp))
        os.system("mkdir {}/{}/{}/{}".format(routes.data_directory, subject, number_exp, routes.frames_directory))
    
    return "{}/{}/{}".format(routes.data_directory, subject, number_exp)
        


if __name__ == '__main__':
    fullCmdArguments = sys.argv

    argument_list = fullCmdArguments[1:]

    unix_options  = "rmt:d:hs:e:"
    gnu_options   = ["recollect", "mix", "time", "display", "help", "subject", "experiment"]

    #Default values
    recollection        = False
    mix                 = False
    time_to_collect     = 10
    display             = -1

    try:
        arguments, values = getopt.getopt(argument_list, unix_options, gnu_options)

    except getopt.error as err:
        print(str(err))
        sys.exit(2)

    for current_arg, current_value in arguments:
        if current_arg in ("-r", "--recollect"):
            print("Recollection enabled")
            recollection    = True

        elif current_arg in ("-m", "--mix"):
            print("Mix sources of data enabled")
            mix = True

        elif current_arg in ("-t", "--time"):
            print("Time set to {} seconds".format(current_value))
            time_to_collect = int(current_value)

        elif current_arg in ("-d","--display"):
            print("Setting display to: {}".format(current_value))
            display = current_value

        elif current_arg in ("-s", "--subject"):
            identifier = hash(current_value)
            print("Subject: {} ({})".format(current_value, identifier))

        elif current_arg in ("-e", "--experiment"):
            experiment_number = current_value
            print("Number of the current experiment: {}".format(current_value))
            
        elif current_arg in ("-h", "--help"):
            print("-r , --recollect: \t\t\tEnables the function of recollect data")
            print("-m , --mix: \t\t\t\tEnables the function of mix the sources of data")
            print("-t {number}, --time {number}: \t\tset the time to recollect data from InGait")
            print("-d {0 or 1}, --display {0 or 1}: \tif display sets to 0, then Observer display frames.")
            print("\t\t\t\t\tif display sets to 1, then Observer doesn't display frames")

    if recollection:
        path = create_directories_estructure(identifier, experiment_number)
        see_and_stomp(time_to_collect, display)

    if mix:
        mix_sources_of_data()
