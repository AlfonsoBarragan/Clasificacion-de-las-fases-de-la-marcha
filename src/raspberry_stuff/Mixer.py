# Combinar fuentes de datos

import cv2
import threading
import time
import os

def throw_observer(time_to_collect, display):
    os.system("python3 Observer.py -t {} -d {}".format(time_to_collect, display))
    
def throw_stomper(time_to_collect):
    os.system("python3 Stomper.py -t {} ".format(time_to_collect))

def see_and_stomp():
    # Create two threads as follows
    thread_obs = threading.Thread(target=throw_observer, args=[20,-1])
    thread_sto = threading.Thread(target=throw_stomper, args=[20])    
    
    thread_obs.start()
    thread_sto.start()

if __name__ == '__main__':
    print("Stomp and observe...")
    see_and_stomp()
    
