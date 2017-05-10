#!/bin/bash

./darknet detector train ./cfg/seat.data ./cfg/seat.cfg darknet19_448.conv.23
#./darknet detector train ./cfg/seat.data ./cfg/seat.cfg backup_q2/seat_10000.weights

