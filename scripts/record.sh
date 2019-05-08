#!/usr/bin/bash
# Recording script from a raspberry pi to record 90 fps at
# 720 x 1080 (WIP)

raspivid -t 10000 -w 720 -h 1080 -fps 90 -pts timecodes.txt -o test.h264
mkvmerge -o video.mkv --timecodes 0:timecodes.txt test.h264