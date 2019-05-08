# from bluepy import btle
# from queue import Queue

import struct
import os, sys
import time
import binascii
import threading
import pexpect
import routes

# # class ScanDelegate(btle.DefaultDelegate):
# #     def __init__(self):
# #         btle.DefaultDelegate.__init__(self)

# #     def handleDiscovery(self, dev, isNewDev, isNewData):
# #         if isNewDev:
# #             print ("Discovered device "+ dev.addr)
# #         elif isNewData:
# #             print ("Received new data from "+ dev.addr)

# # scanner = btle.Scanner().withDelegate(ScanDelegate())
# # devices = scanner.scan(10.0)

# # for dev in devices:
# #     print ("Device %s (%s), RSSI=%d dB" % (dev.addr, dev.addrType, dev.rssi))
# #     for (adtype, desc, value) in dev.getScanData():
# #         print ("  %s = %s" % (desc, value)) 

# class MyDelegate(btle.DefaultDelegate):
#     def __init__(self, name, counter):
#         btle.DefaultDelegate.__init__(self)

#         self.counter        = 0
#         self.name = name

#         self.counter_vuelta = 0
#         self.counter_data   = 0

#         self.mux_1          = 0
#         self.mux_2          = 0
#         self.mux_3          = 0
#         self.mux_4          = []

#         self.acc            = 0
#         self.gyro           = 0

#         self.queue_time     = []

#     def handleNotification(self, cHandle, data):
#         # Preguntar a cosmin por el formato de las movidas de presion
#         # Osea, se que esta administrado 20 Bytes (16 B un mux y 4 B del ultimo)
#         # Lo que no me acuerdo es que simbolizaba cada Byte
#         print("{}, {}".format(cHandle, data))
#         self.queue_time.append(time.time())
#         self.counter += 1

        # if len(data) == 20:
        #     if self.counter_data == 0:
        #         self.mux_1 = struct.unpack("!HHHHHHHH", data[:16])
                
        #         if self.counter_data == 0:
        #             self.mux_4.append(struct.unpack("!HH", data[16:]))

        #         self.counter_data += 1

        #     elif self.counter_data == 1:
        #         self.mux_2 = struct.unpack("!HHHHHHHH", data[:16])
                
        #         if self.counter_data == 1:
        #             self.mux_4.append(struct.unpack("!HH", data[16:]))
                
        #         self.counter_data += 1

        #     elif self.counter_data == 2:
        #         self.mux_3 = struct.unpack("!HHHHHHHH", data[:16])
                
        #         if self.counter_data == 2:
        #             self.mux_4.append(struct.unpack("!HH", data[16:]))

        #         self.counter_data += 1

        # elif len(data) == 16:
        #         self.mux_4.append(struct.unpack("<HH", data[12:16]))

        #         os.system("echo '[INSOLE {}] mux_0: {} mux_1: {} mux_2: {} mux_3: {}' >> registro_muestras.txt".format(self.name, self.mux_1, self.mux_2, self.mux_3, self.mux_4))
        #         self.mux_4 = []

        #         self.counter_data = 0

        # self.counter_vuelta += 1


#  InsoleR
# plantilla_r = btle.Peripheral("00:A0:50:00:00:02", btle.ADDR_TYPE_PUBLIC)

# # InsoleL
# plantilla_l = btle.Peripheral("00:a0:50:00:00:11", btle.ADDR_TYPE_PUBLIC)

# # plantilla_r.withDelegate(MyDelegate("RIGHT"))

# plantilla_l.withDelegate(MyDelegate("LEFT_1", 0))

# local_counter   = 0
# total_times     = 999

# plantilla_r_serv_1    = plantilla_r.getServiceByUUID("9c056500-3b96-42e2-a104-6746503d4ba0")
# plantilla_r_serv_2    = plantilla_r.getServiceByUUID("2fb7e922-9c55-4445-9fcd-c77842ff2360")

# plantilla_r_acc_gyro     = plantilla_r_serv_1.getCharacteristics("9c056500-3b96-42e2-a104-6746503d4ba1")[0]

# plantilla_r_mux_1        = plantilla_r_serv_2.getCharacteristics("2fb7e922-9c55-4445-9fcd-c77842ff2361")[0]
# plantilla_r_mux_2        = plantilla_r_serv_2.getCharacteristics("2fb7e922-9c55-4445-9fcd-c77842ff2362")[0]
# plantilla_r_mux_3        = plantilla_r_serv_2.getCharacteristics("2fb7e922-9c55-4445-9fcd-c77842ff2363")[0]

# plantilla_l_serv_1    = plantilla_l.getServiceByUUID("9c056500-3b96-42e2-a104-6746503d4ba0")
# plantilla_l_serv_2    = plantilla_l.getServiceByUUID("2fb7e922-9c55-4445-9fcd-c77842ff2360")

# plantilla_l_acc_gyro     = plantilla_l_serv_1.getCharacteristics("9c056500-3b96-42e2-a104-6746503d4ba1")[0]

# plantilla_l_mux_1        = plantilla_l_serv_2.getCharacteristics("2fb7e922-9c55-4445-9fcd-c77842ff2361")[0]
# plantilla_l_mux_2        = plantilla_l_serv_2.getCharacteristics("2fb7e922-9c55-4445-9fcd-c77842ff2362")[0]
# plantilla_l_mux_3        = plantilla_l_serv_2.getCharacteristics("2fb7e922-9c55-4445-9fcd-c77842ff2363")[0]

# ch          = p.getCharacteristics()

# p.delegate.counter_vuelta = 0
# # print("mux_1: {}\nmux_2: {}\nmux_3: {}\nacc_gyro: {}".format(mux_1_data, mux_2_data, mux_3_data, acc_gyro_data))
# local_counter += 1
# def worker(colita):
#     while True:
#         data = colita.get()
#         if data is None:
#             break

#         if len(data) == 20:
#             os.system('echo "{},{}" >> bytes_file.txt'.format(struct.unpack("!HHHHHHHH",data[:16]), struct.unpack("!HH",data[16:])))
#         else:
#             os.system('echo "{}" >> bytes_file.txt'.format(struct.unpack("<HH", data[12:16])))

# def listen(plantilla_l):       
#     time_init = time.time()
#     while plantilla_l.delegate.counter < 1000:
#         plantilla_l.getCharacteristics()


#     time_finish = time.time()

#     print("{} samples in {}".format(plantilla_l.delegate.counter, time_finish - time_init))
#     print("{}".format(plantilla_l.delegate.queue_time))

# t = threading.Thread(target=listen(plantilla_l))
# t.start()

# plantilla_l.disconnect()
counter = 1

list_l = []
list_d = []

os.system("sudo hcitool lecc 00:a0:50:00:00:11")
os.system("sudo hcitool lecc 00:a0:50:00:00:02")

child_R = pexpect.spawn("hcidump -i insoleR -x -t")
child_L = pexpect.spawn("hcidump -i insoleL -x -t")

time_to_recolect = int(input("Give me the time baby > "))

time.sleep(2)

time_init = time.time()

# Problemas:
#   -> Basicamente cuando se ejecuta el hcidump recoge a punta pala todo lo que le entra 
#       Asi que despues va a haber que filtrar por handle
#   -> Tambien otra movida es pillar los value y el timestamp, para ello, como al partirlo por \r\n
#       se genera un patron de que 1 TimeStamp y handle | 2 Notification | 3 handle LED | 4 value
 
while counter <= time_to_recolect * 360 :
    # if counter == 120:
    #     counter = 1

    child_L.expect("\r\n")
    child_R.expect("\r\n")

    list_l.append(child_L.buffer)
    list_d.append(child_R.buffer)

    counter += 1

time_finish = time.time()

print("Total Samples: InsoleL {}, InsoleR {} in {} time".format(len(list_l), len(list_d), time_finish-time_init ))


file_l = open(routes.samples_l, 'w')
file_d = open(routes.samples_d, 'w')

if (os.path.getsize(routes.samples_l)==0):
    file_l.write("Timestamp,Source,S0,S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16,S17,S18,S19,S20,S21,S22,S23,S24,S25,S26,S27,S28,S29,S30,S31,A_X,A_Y,A_Z,VA_X,VA_Y,VA_Z")

if (os.path.getsize(routes.samples_d)==0):
    file_d.write("Timestamp,Source,S0,S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16,S17,S18,S19,S20,S21,S22,S23,S24,S25,S26,S27,S28,S29,S30,S31,A_X,A_Y,A_Z,VA_X,VA_Y,VA_Z")
    
for sample_number in range(len(list_d) - 1):

    sample_contain_l = str(list_l[sample_number])
    sample_contain_d = str(list_d[sample_number])


    file_l.write(sample_contain_l+"\n")
    file_d.write(sample_contain_d+"\n")

    # if len(data) == 20:
        #     if self.counter_data == 0:
        #         self.mux_1 = struct.unpack("!HHHHHHHH", data[:16])
                
        #         if self.counter_data == 0:
        #             self.mux_4.append(struct.unpack("!HH", data[16:]))

        #         self.counter_data += 1

        #     elif self.counter_data == 1:
        #         self.mux_2 = struct.unpack("!HHHHHHHH", data[:16])
                
        #         if self.counter_data == 1:
        #             self.mux_4.append(struct.unpack("!HH", data[16:]))
                
        #         self.counter_data += 1

        #     elif self.counter_data == 2:
        #         self.mux_3 = struct.unpack("!HHHHHHHH", data[:16])
                
        #         if self.counter_data == 2:
        #             self.mux_4.append(struct.unpack("!HH", data[16:]))

        #         self.counter_data += 1

        # elif len(data) == 16:
        #         self.mux_4.append(struct.unpack("<HH", data[12:16]))

        #         os.system("echo '[INSOLE {}] mux_0: {} mux_1: {} mux_2: {} mux_3: {}' >> registro_muestras.txt".format(self.name, self.mux_1, self.mux_2, self.mux_3, self.mux_4))
        #         self.mux_4 = []

        #         self.counter_data = 0

        # self.counter_vuelta += 1


