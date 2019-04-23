import struct
import os, sys
import time
import binascii
import threading
import pexpect
import routes

import sys, getopt

def stomp(time_to_recolect=10):
    counter = 1

    list_samples = []

    os.system("sudo hcitool lecc 00:a0:50:00:00:11")
    os.system("sudo hcitool lecc 00:a0:50:00:00:02")

    child_recollect = pexpect.spawn("hcidump -x -t")
    
    time.sleep(2)

    time_init = time.time()

    # Problemas:
    #   -> Basicamente cuando se ejecuta el hcidump recoge a punta pala todo lo que le entra 
    #       Asi que despues va a haber que filtrar por handle
    #   -> Tambien otra movida es pillar los value y el timestamp, para ello, como al partirlo por \r\n
    #       se genera un patron de que 1 TimeStamp y handle | 2 Notification | 3 handle LED | 4 value
    
    while counter <= time_to_recolect * 720 :
        # if counter == 120:
        #     counter = 1

        child_recollect.expect("2019")

        list_samples.append(child_recollect.buffer)

        counter += 1

    time_finish = time.time()

    print("Total Samples: Data {}, in {} time".format(len(list_samples), time_finish-time_init ))

    write_data_to_file(list_samples)

def write_data_to_file(list_samples):
    
    samples_contain = open("{}/{}".format(routes.sample_directory,routes.samples_recollect), 'w')
    
    for sample_number in range(len(list_samples) - 1):
        samples_contain.write(str(list_samples[sample_number])+"\n")
        
def process_samples():
	# Hacer el preproceso del fichero de muestras
    pass

if __name__ == '__main__':
    fullCmdArguments = sys.argv
    
    argument_list = fullCmdArguments[1:]
    
    unix_options  = "t:h"
    gnu_options   = ["time", "help"]
    
    try:
        arguments, values = getopt.getopt(argument_list, unix_options, gnu_options)
    
    except getopt.error as err:
        print(str(err))
        sys.exit(2)
    
    for current_arg, current_value in arguments:
        if current_arg in ("-t", "--time"):
            print("Beggining stomp for {} seconds".format(current_value))
            stomp(int(current_value))
        elif current_arg in ("-h", "--help"):
            print("-t {number}, --time {number}: \tset the time to recollect data from InGait")