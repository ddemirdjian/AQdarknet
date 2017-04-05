#!/usr/bin/env python
import numpy as np
import cv2
import os, sys, glob, subprocess, shutil, csv, time

DARKNET_DIR = "/mnt/dev/AQdarknet"
LABEL_NAMES = ['']

# draw bounding box in image
def drawBox(img, text, pt1, pt2, boxColor, prob = -1):
    # draw rectangle
    cv2.rectangle(img, pt1, pt2, boxColor, 2)

    # draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.6
    thickness = 1
    size = cv2.getTextSize(text, font, font_size, thickness)[0]
    x, y = ((pt1[0] + pt2[0]) / 2, pt1[1] - 2)
    label_top_left = (x - size[0] / 2, y - size[1] / 2)
    if prob >= 0:
        text = text + str(round(float(prob), 2))
    cv2.putText(img, text, label_top_left, font, font_size, boxColor, thickness)

# call app
def runApp(out_path, list_imgs, prob_thresh = 0.2):

    # cd darknet folder
    owd = os.getcwd()
    os.chdir(DARKNET_DIR)

    # write list of files to process in a (temporary) text file
    filelist_filename = "filelist.txt"
    filelist_file = open(filelist_filename, 'w', 0)
    for imFilename in list_imgs:
        filelist_file.write("%s\n" % imFilename)
    filelist_file.flush()
    os.fsync(filelist_file.fileno())
    filelist_file.close()

    # run yolo binary (which writes results in a text file "results.txt")
    args = ("./darknet", "detector", "test", "cfg/seat.data", "cfg/seat.cfg", "backup/seat_normals_40000.weights", filelist_filename)
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.communicate()

    # read results
    with open("results.txt") as labelFile:
        labelsDetectedList = list(csv.reader(labelFile, delimiter=' '))

    # process all results (compute TP, FP, etc.. and generate images with bounding boxes)
    TN = FP = TP = FN = 0
    i = 0
    idOut = 0
    boxColorDetection = (0, 255, 0)
    for filenameAP in list_imgs:
        # get full path of file
        labelsDetected = labelsDetectedList[i]

        # read image
        img = cv2.imread(filenameAP)
        imgHeight, imgWidth, imgDepth = img.shape

        # initialize groundtruth and detection states
        isDefect = False
        detectedDefect = False

        # read labels
        labelPath = os.path.join(os.path.dirname(os.path.dirname(filenameAP)), 'labels')
        labelFilename = os.path.join(labelPath, os.path.splitext(os.path.basename(filenameAP))[0] + '.txt')
        with open(labelFilename) as labelFile:
            labels = list(csv.reader(labelFile, delimiter=' '))

        # draw labels
        boxColor = (0, 0, 255)
        for label in labels:
            isDefect = True
            # draw box
            pt1 = (int(float(label[1]) * float(imgWidth)), int(float(label[2]) * float(imgHeight)))
            pt2 = (pt1[0] + int(float(label[3]) * float(imgWidth)), pt1[1] + int(float(label[4]) * float(imgHeight)))
            tag = LABEL_NAMES[int(label[0])]
            drawBox(img, tag, pt1, pt2, boxColor)

        for labelIdx in range(len(labelsDetected)/6):
            label = labelsDetected[labelIdx * 6 : labelIdx * 6 + 6]
            prob = float(label[0])
            if prob >= prob_thresh:
                detectedDefect = True
                x1, y1 = int(label[2]), int(label[3])
                x2, y2 = int(label[4]), int(label[5])
                pt1 = (x1 + (x2-x1)/2, y1 + (y2-y1)/2)
                pt2 = (x2 + (x2-x1)/2, y2 + (y2-y1)/2)
                tag = LABEL_NAMES[int(label[1])]
                drawBox(img, tag, pt1, pt2, boxColorDetection, prob)

        if not isDefect:
            if not detectedDefect:
                TN = TN + 1
            else:
                FP = FP + 1
        else:
            if not detectedDefect:
                FN = FN + 1
            else:
                TP = TP + 1

        # draw input image with groundtruth label
        # cv2.imshow("Groundtruth", img)
        # cv2.waitKey(0)

        # save image
        if isDefect or detectedDefect:
            imgFilenameOut = os.path.join(out_path, '{:06d}'.format(idOut) + '.jpg')
            cv2.imwrite(imgFilenameOut, img)
        # print 'FP rate = ' + str((100 * FP) / (FP + TN))
        # print 'Precision = ' + str((100 * TP) / (TP + FP))
        # print 'Recall = ' + str((100 * TP) / (TP + FN))
        idOut = idOut + 1
        i = i+1

    # move back to initial folder
    os.chdir(owd)
    return TP, FP, TN, FN


if __name__ == "__main__":
    show = 0

    # run detection
    seat_path = '/mnt/data/seat/seatset2_yolo_normals_1055b'
    # seat_path = '/mnt/data/seat/seatset2_yolo_color_1055b'
    im_defect_path = seat_path + '/seat_test.txt'
    outPathDefect = seat_path + '/result/defect'

    # read images in folder
    listDefectImgs = [line.rstrip('\n') for line in open(im_defect_path)]
    #listImgs.sort()

    # create result folder
    if not os.path.exists(outPathDefect):
        os.makedirs(outPathDefect)

    # process all images in folder
    TP, FP, TN, FN = runApp(outPathDefect, listDefectImgs)

    # 0.63 TP, 0.31 FP
    print 'Precision = ' + str((100 * TP) / (TP + FP))
    print 'Recall = ' + str((100 * TP) / (TP + FN))
