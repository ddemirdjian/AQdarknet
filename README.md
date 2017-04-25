** Build

From main folder, edit Makefile (if necessary) and run:
> make

** Convert seat data
(from inside the AQdarknet root folder)

Modify variables from 'scripts/seat/convert2yoloRT.py' 
- seatDBSource: folder containing the rendered seats
- indexSource: folder containing the train.txt and test.txt files (containing the list of training and testing data/folders)
- dbRoot: basename of destination folders (containing the converted training and testing data)

Run scripts/seat/convert2yoloRT.py
> python scripts/seat/convert2yoloRT.py


** Train network 
(from inside the AQdarknet root folder)
Modify cfg/seat.data to set the location of the destination folders (with basename dbRoot, modified above) containing the converted training and testing data)

Run training script
> source scripts/train_seat.sh

This will save models in the backup folder.

** Run test on testing dataset

Modify cripts/seat/test.py script
- DARKNET_DIR: location of darknet folder
- seat_path: location of the converted testing set (with basename dbRoot)
To run it, call:
> python scripts/seat/test.py

** Miscellaneous

Run individual tests by calling (from inside the AQdarknet root folder), the following command:
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
