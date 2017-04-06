** Build

From main folder, run:
> make

** Run test

To run it, call (from inside the AQdarknet folder), the following command:  

> ./darknet detector test cfg/seat.data cfg/seat.cfg backup/seat_normals_40000.weights [FILELIST.TXT]

where [FILELIST.TXT] is a text file where each line corresponds to an image to test
The code outputs a "results.txt" file containing, for each image/line of [FILELIST.TXT] the detection results.
- Each detection result is made of 6 values prob class x1 y1 x2 y2, where
prob: probability of detection (between 0 and 1)
class: Should always be 0 since we have only 1 class
[x1 y1 x2 y2] the top-left and bottom-right coordinates of the bounding box

- If there are no detection in an image, the line is empty
- If there are, say, 3 detections in an image, the line has 18 values
- You can find a python example calling the detector command in AQdarknet\script\seat\test.py
