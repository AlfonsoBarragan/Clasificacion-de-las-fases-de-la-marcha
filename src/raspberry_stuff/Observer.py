# USAGE
# python picamera_fps_demo.py
# python picamera_fps_demo.py --display 1

# import the necessary packages
from __future__ import print_function
from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera

import argparse
import imutils
import time
import cv2
import sys, getopt
import routes
from utils import printProgressBar

def observe(time_to_record=10, display=-1):
	# initialize the camera and stream
	frames_to_record = time_to_record * 240
        
	camera = PiCamera()
	camera.resolution = (1280, 720)
	camera.framerate = 120
	rawCapture = PiRGBArray(camera, size=(1280, 720))
	stream = camera.capture_continuous(rawCapture, format="bgr",
		use_video_port=True)

	# allow the camera to warmup and start the FPS counter
	print("[INFO] sampling frames from `picamera` module...")
	fps = FPS().start()

	stream.close()
	rawCapture.close()
	camera.close()

	# created a *threaded *video stream, allow the camera sensor to warmup,
	# and start the FPS counter
	print("[INFO] sampling THREADED frames from `picamera` module...")
	vs = PiVideoStream().start()
	fps = FPS().start()
	frame_list = []

	# loop over some frames...this time using the threaded stream
	while fps._numFrames < frames_to_record:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		frame = vs.read()
		frame_list.append([time.time(), frame])
		frame = imutils.resize(frame, width=1280)

		# check to see if the frame should be displayed to our screen
		if display > 0:
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF

		# update the FPS counter
		fps.update()

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()
	
	write_frames_and_dataset(frame_list)

def write_frames_and_dataset(frame_list):
	frames_dataset = open("{}/{}".format(routes.data_directory , routes.frames_dataset), 'w')
	frames_dataset.write("Timestamp,Id_frame\n")

	# Parameters progress bar
	size 	= len(frame_list)
	index 	= 0

	printProgressBar(index, size, prefix = 'Progress:', suffix = 'Complete', length = 50)
	
	for i in frame_list:
		timestamp = (i[0] * 1000)
		frames_dataset.write("{},{}\n".format(timestamp, frame_list.index(i)))
		cv2.imwrite("{}/frame_{}.jpg".format(routes.frames_directory, frame_list.index(i)), i[1])
		
		index += 1
		printProgressBar(index, size, prefix = 'Progress:', suffix = 'Complete', length = 50)

if __name__ == '__main__':
	fullCmdArguments = sys.argv

	argument_list = fullCmdArguments[1:]

	unix_options  = "t:d:h"
	gnu_options   = ["time", "display", "help"]

	try:
		arguments, values = getopt.getopt(argument_list, unix_options, gnu_options)

	except getopt.error as err:
		print(str(err))
		sys.exit(2)

	time_to_collect = 10
	display         = -1

	for current_arg, current_value in arguments:
		if current_arg in ("-t", "--time"):
			print("Beggining observancy for {} seconds".format(current_value))
			time_to_collect = current_value
		
		elif current_arg in ("-d","--display"):
			print("Setting display to: {}".format(current_value))
			display = current_value

		elif current_arg in ("-h", "--help"):
			print("-t {number}, --time {number}: \t\tset the time to recollect data from InGait")
			print("-d {0 or 1}, --display {0 or 1}: \tif display sets to 0, then Observer display frames.")
			print("\t\t\t\t\tif display sets to 1, then Observer doesn't display frames")

	observe(int(time_to_collect), int(display))