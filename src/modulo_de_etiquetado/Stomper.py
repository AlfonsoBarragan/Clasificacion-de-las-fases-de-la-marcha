import os
import time
import re
import datetime

import sys, getopt

from modulo_de_funciones_de_soporte.utils import printProgressBar
from modulo_de_funciones_de_soporte import routes

insole_l = "00:a0:50:00:00:11"
insole_r = "00:a0:50:00:00:02"

def stomp(path, time_to_recolect=10):
    counter = 1

    # Insole_L 
    os.system("sudo hcitool lecc {}".format(insole_l))
    
    # Insole_R
    os.system("sudo hcitool lecc {}".format(insole_r))
    
    # Wait until begins the record
    time.sleep(1)

    os.system("sudo hcidump -x -t > {}/{} &".format(path, routes.samples_recollect))
    time_init = time.time()
    
    while (time.time() - time_init) < time_to_recolect :
        counter += 1
        
    time_finish = time.time()
    
    time.sleep(0.5)
    os.system("sudo ./killer.sh")

    time_total = time_finish - time_init
    
    time.sleep(1)
    print("Recolected {} samples in {} seconds".format(int(os.system('wc -l {}/{}'.format(path, routes.samples_recollect)))/time_total*6, time_total))

def write_data_to_file(list_samples, output_file_name, path):
    
    samples_contain = open("{}/{}".format(path, output_file_name), 'w')
    
    for sample_number in range(len(list_samples) - 1):
        samples_contain.write(str(list_samples[sample_number])+"\n")
        
def convert_date_to_timestamp(date_list):
    date_string = "{} {}".format(date_list[0], date_list[1])

    dt_obj = datetime.datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S.%f')
    return dt_obj.timestamp() * 1000

def remove_carriage_return(path):
    print("[*] Eliminando retornos de carro...")

    samples_file            = open("{}/{}".format(path, routes.samples_recollect), 'r')
    samples_without_return  = open("{}/{}".format(path, routes.samples_without_return), 'w') 

    text = samples_file.readlines()

    # Parameters progress bar
    size 	= len(text)
    index 	= 0

    printProgressBar(index, size, prefix = 'Progress:', suffix = 'Complete', length = 50)
	
    for line in text:
        samples_without_return.write(line.replace("\r", "").replace("\t", "").replace("      ", "").replace("    ",""))

        index += 1
        printProgressBar(index, size, prefix = 'Progress:', suffix = 'Complete', length = 50)

    samples_without_return.close()
    samples_file.close()

    print("[*] Retornos de carro eliminados exitosamente!")

def process_samples(path):
    # Eliminar los retornos de carro para leer bien el fichero
    remove_carriage_return(path)

	# Cargar el fichero de datos de las plantillas
    samples_file    = open("{}/{}".format(path, routes.samples_without_return), 'r')
    samples_cleaned = open("{}/{}".format(path, routes.samples_cleaned_uni), 'w') 

    # Se crean las expresiones regulares para poder capturar los datos relevantes
    # de cada linea
    timestamp_patron    = re.compile(' dlen 27| dlen 23')
    handle_patron       = re.compile(' handle ')
    values_patron       = re.compile('value ')
    
    # Escribimos la cabecera del csv
    samples_cleaned.write("Timestamp,Date,Source,Value_1,Value_2,Value_3,Value_4,Value_5,Value_6,Value_7,Value_8,Value_9," +
                            "Value_10,Value_11,Value_12,Value_13,Value_14,Value_15,Value_16,Value_17,Value_18,Value_19,Value_20\n")
    
    # Parameters progress bar
    size 	= len(samples_file.readlines())
    index 	= 2

    # Set the pointer of the file to position 0
    samples_file.seek(0)

    # Leer las dos primeras lineas y no hacer nada con ellas, porque no contienen
    # información relevante
    _ = samples_file.readline()
    _ = samples_file.readline()

    # Leemos la primera linea util y preparamos los contenedores de datos
    # para luego escribir el fichero de salida
    line            = samples_file.readline()
    line_to_write   = ""

    timestamps_container    = []
    handle_container        = []
    values_container        = []

    printProgressBar(index, size, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    while line:

        if timestamp_patron.search(line) != None:
            timestamp_init  = timestamp_patron.search(line).start()
            
            if handle_patron.search(line) != None:
                handle_end      = handle_patron.search(line).end()

                timestamp_split = line[:timestamp_init].split(' ')
                handle_split    = line[handle_end:].split(' ')

                timestamps_container.append([timestamp_split[0], timestamp_split[1]])
                handle_container.append(handle_split[0])

        elif values_patron.search(line) != None:
            values_list = line[values_patron.search(line).end():].split(' ')
            values_container.append(values_list[0:len(values_list)-1])

        if len(values_container) == 1:
            
            if len(timestamps_container) > 0:
                line_to_write += "{},{} {},{}".format(convert_date_to_timestamp(timestamps_container[0]), timestamps_container[0][0],
                                                        timestamps_container[0][1], handle_container[0])
                for value in values_container[0]:
                    line_to_write += ",{}".format(value)
                
                if len(values_container[0]) != 20:
                    line_to_write += ",-1,-1,-1,-1"

                line_to_write += "\n"

                samples_cleaned.write(line_to_write)

            # Reseteo de los contenedores de información
            timestamps_container    = []
            handle_container        = []
            values_container        = []
            line_to_write           = ""

        # Avance en la lectura de las lineas
        line = samples_file.readline()

        index += 1
        printProgressBar(index, size, prefix = 'Progress:', suffix = 'Complete', length = 50)


if __name__ == '__main__':
    fullCmdArguments = sys.argv
    
    argument_list = fullCmdArguments[1:]
    
    unix_options  = "ct:hp:"
    gnu_options   = ["clean","time", "help", "path"]
    
    try:
        arguments, values = getopt.getopt(argument_list, unix_options, gnu_options)
    
    except getopt.error as err:
        print(str(err))
        sys.exit(2)
    
    for current_arg, current_value in arguments:
        if current_arg in ("-t", "--time"):
            print("Beggining stomp for {} seconds".format(current_value))
            time = current_value
            collect = True

        elif current_arg in ("-h", "--help"):
            print("-p , --process: \t\tclean and prepare the raw values from InGait to be used")
            print("-t {number}, --time {number}: \tset the time to recollect data from InGait")
        
        elif current_arg in ("-p", "--path"):
            print("Setting path for data in: {}".format(current_value))
            path = current_value
        
        elif current_arg in ("-c", "--clean"):
            print("Processing samples file")
            clean = True
            
    if collect and path is not None:
        stomp(path, time)

    if clean and path is not None:    
        process_samples(path)
        
        