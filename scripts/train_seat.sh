#!/bin/bash

#./darknet detector train ./cfg/seat.data ./cfg/seat.cfg darknet19_448.conv.23
#./darknet detector train ./cfg/seat.data ./cfg/seat.cfg backup_colorb/seat_32000.weights
./darknet detector train ./cfg/seat.data ./cfg/seat.cfg backup_colorc/seat_35000.weights
