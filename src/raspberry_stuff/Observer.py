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

def observe(time_to_record=10, display=-1):
	# initialize the camera and stream
	frames_to_record = time_to_record * 120
        
	camera = PiCamera()
	camera.resolution = (1280, 720)
	camera.framerate = 120
	rawCapture = PiRGBArray(camera, size=(1280, 720))
	stream = camera.capture_continuous(rawCapture, format="bgr",
		use_video_port=True)

	# allow the camera to warmup and start the FPS counter
	print("[INFO] sampling frames from `picamera` module...")
	time.sleep(2.0)
	fps = FPS().start()

	stream.close()
	rawCapture.close()
	camera.close()

	# created a *threaded *video stream, allow the camera sensor to warmup,
	# and start the FPS counter
	print("[INFO] sampling THREADED frames from `picamera` module...")
	vs = PiVideoStream().start()
	time.sleep(2.0)
	fps = FPS().start()
	frame_list = []

	# loop over some frames...this time using the threaded stream
	while fps._numFrames < frames_to_record:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		frame = vs.read()
		frame_list.append([time.time(), frame])
		frame = imutils.resize(frame, width=400)

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
	
	write_frames(frame_list)

def write_frames(frame_list):
    for i in frame_list:
        timestamp = int(round(i[0] * 1000))
        cv2.imwrite("{}/frame_{}.jpg".format(routes.frames_directory, timestamp), i[1])


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
			print("Beggining stomp for {} seconds".format(current_value))
			time_to_collect = current_value
		elif current_arg in ("-h", "--help"):
			print("-t {number}, --time {number}: \tset the time to recollect data from InGait")

		elif current_arg in ("-d","--display"):
			print("Setting display to: {}".format(current_value))
			display = current_value

	observe(int(time_to_collect), int(display))