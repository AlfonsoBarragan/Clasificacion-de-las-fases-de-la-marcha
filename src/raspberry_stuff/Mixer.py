# External libraries
import cv2
import threading
import time
import os
import sys, getopt
import struct
from functools import reduce

# Internal libraries
import utils
import routes

def throw_observer(time_to_collect, display):
    os.system("python3 Observer.py -t {} -d {}".format(time_to_collect, display))

def throw_stomper(time_to_collect):
    os.system("python3 Stomper.py -t {} ".format(time_to_collect))

def see_and_stomp(time, display=-1):
    # Create two threads as follows
    thread_obs = threading.Thread(target=throw_observer, args=[time,display])
    thread_sto = threading.Thread(target=throw_stomper, args=[time])

    thread_obs.start()
    thread_sto.start()

def assign_entry_point_and_transform_datasets(dataframe):
    dataframe_aux = dataframe[(dataframe.Value_20 == '-1')]
    index_syncro = int(dataframe_aux.iloc[2].name)

    new_dataframe = dataframe[(dataframe.index > index_syncro)]

    return new_dataframe

def converse_hex_plantar_pressure(dataframe_insole):
	mux_1 = []
	mux_2 = []
	mux_3 = []
	mux_4 = []
	
	for i in range(0, len(dataframe_insole), 4):
		sample_1 = dataframe_insole.iloc[i-3]
		sample_2 = dataframe_insole.iloc[i-2]
		sample_3 = dataframe_insole.iloc[i-1]
		sample_4 = dataframe_insole.iloc[i]
		
		mux_1.extend(sample_1[3:19])
		mux_2.extend(sample_2[3:19])
		mux_3.extend(sample_3[3:19])
		
		mux_4.extend(sample_1[20:])
		mux_4.extend(sample_2[20:])
		mux_4.extend(sample_3[20:])
		mux_4.extend(sample_4[16:])
		
		mux_1 = reduce((lambda x, y: x + y), list(map(lambda x:x[2:] , mux_1))) 
		mux_2 = reduce((lambda x, y: x + y), list(map(lambda x:x[2:] , mux_2)))
		mux_3 = reduce((lambda x, y: x + y), list(map(lambda x:x[2:] , mux_3)))
		mux_4 = reduce((lambda x, y: x + y), list(map(lambda x:x[2:] , mux_4)))
		
		
		mux_1_values = struct.unpack("!HHHHHHHH", bytes.fromhex(mux_1))
		mux_2_values = struct.unpack("!HHHHHHHH", bytes.fromhex(mux_2))
		mux_3_values = struct.unpack("!HHHHHHHH", bytes.fromhex(mux_3))
		mux_4_values = struct.unpack("!HHHHHHHH", bytes.fromhex(mux_4))
		
		print("{}, {}, {}, {}".format(mux_1_values,mux_2_values,mux_3_values,mux_4_values))
		break
def mix_sources_of_data(id_handle_insoleL, id_handle_insoleR):
    full_data_cleaned = utils.read_dataset("{}/{}".format(routes.sample_directory,                                                                 routes.samples_cleaned_uni))

    # Divide datasets by insole
    insoleL_dataset = full_data_cleaned[(full_data_cleaned.Source == id_handle_insoleL)]
    insoleR_dataset = full_data_cleaned[(full_data_cleaned.Source == id_handle_insoleR)]

    # Eligue un buen punto inicial omitiendo algunas muestras para comenzar la traducción
    # de hexadecimal a datos de presión plantar
    insoleL_dataset = assign_entry_point_and_transform_datasets(insoleL_dataset)
    insoleR_dataset = assign_entry_point_and_transform_datasets(insoleR_dataset)
    
    converse_hex_plantar_pressure(insoleL_dataset)
    converse_hex_plantar_pressure(insoleR_dataset)

    # Una vez divididos, hacer el siguiente preproceso:
    #   -> capturar las muestras de 4 en 4 y asignarles los frames entre  el primer instante
    #      y el ultimo, algo tipo el metodo de abajo.
    #   -> Para asignarle un unico frame a cada muestra se me ocurre, que el frame más
    #      "correcto" es aquel que es más cercano a la recepción de la muestra completa, o en
    #      otras palabras aquel cuya diferencia con el timestamp de la ultima muestra sea menor.
    #      De todas maneras lo hablaré con Ivan para ver si lo ve bien. Dejo esta implementación
    #      para más adelante.


def assign_frame_to_sample(df_to_label, df_labels, probe_time):

  for i in df_labels["UUID"]:
    aux = df_to_label[(int(i) <= df_to_label.UUID ) & (int(i) >= (df_to_label.UUID - probe_time))]

    if not (aux.empty):
        df_to_label.loc[aux.index, 'attack'] = 1


if __name__ == '__main__':
    fullCmdArguments = sys.argv

    argument_list = fullCmdArguments[1:]

    unix_options  = "rmt:d:h"
    gnu_options   = ["recollect", "mix", "time", "display", "help"]

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

        elif current_arg in ("-h", "--help"):
            print("-r , --recollect: \t\t\tEnables the function of recollect data")
            print("-m , --mix: \t\t\t\tEnables the function of mix the sources of data")
            print("-t {number}, --time {number}: \t\tset the time to recollect data from InGait")
            print("-d {0 or 1}, --display {0 or 1}: \tif display sets to 0, then Observer display frames.")
            print("\t\t\t\t\tif display sets to 1, then Observer doesn't display frames")

    if recollection:
        see_and_stomp(time, display)

    if mix:
        mix_sources_of_data()

