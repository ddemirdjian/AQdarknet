#!/usr/bin/env python
import numpy as np
import cv2
import os, sys, glob, subprocess, shutil, csv, time

DARKNET_DIR = "/home/david/dev/AQDarknet"
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

# compute intersection-over-union (iou) of two windows
def bb_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

# call app
def runApp(out_path, list_imgs, seqId, prob_thresh = 0.2):

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

    # run yolo binary (the yolo binary "darknet" writes results in a text file "results.txt")
    print "Running detector on images..."
    args = ("./darknet", "detector", "test", "cfg/seat.data", "cfg/seat.cfg", "-thresh", str(prob_thresh), "backup/seat_" + str(seqId) + ".weights", filelist_filename, "-output", "/home/david/tmp/results.txt")
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.communicate()

    # read results
    with open("/home/david/tmp/results.txt") as labelFile:
        labelsDetectedList = list(csv.reader(labelFile, delimiter=' '))

    # process all results (compute TP, FP, etc.. and generate images with bounding boxes)
    iouList = []
    TN = FP = TP = FN = 0.0
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
        isDefect = False # groundtruth
        detectedDefect = False # prediction

        # read labels
        labelPath = os.path.join(os.path.dirname(os.path.dirname(filenameAP)), 'labels')
        labelFilename = os.path.join(labelPath, os.path.splitext(os.path.basename(filenameAP))[0] + '.txt')
        with open(labelFilename) as labelFile:
            labels = list(csv.reader(labelFile, delimiter=' '))

        # draw gt labels
        boxColor = (0, 0, 255)
        for label in labels:
            isDefect = True
            # draw box
            xc, yc = float(label[1]), float(label[2])
            winWidth, winHeight = float(label[3]), float(label[4])
            pt1 = (int((xc - winWidth/2) * float(imgWidth)), int((yc - winHeight/2) * float(imgHeight)))
            pt2 = (int((xc + winWidth/2) * float(imgWidth)), int((yc + winHeight/2) * float(imgHeight)))
            tag = LABEL_NAMES[int(label[0])]
            drawBox(img, tag, pt1, pt2, boxColor)

            # check for matching detections, update iou
            for labelIdx in range(len(labelsDetected)/6):
                label = labelsDetected[labelIdx * 6 : labelIdx * 6 + 6]
                prob = float(label[0])
                if prob >= prob_thresh:
                    x1, y1 = int(label[2]), int(label[3])
                    x2, y2 = int(label[4]), int(label[5])
                    iou = bb_iou((x1, y1, x2, y2), (pt1[0], pt1[1], pt2[0], pt2[1]))
                    if iou > 0:
                        iouList.append(iou)


    # draw detected labels
        for labelIdx in range(len(labelsDetected)/6):
            label = labelsDetected[labelIdx * 6 : labelIdx * 6 + 6]
            prob = float(label[0])
            if prob >= prob_thresh:
                detectedDefect = True
                x1, y1 = int(label[2]), int(label[3])
                x2, y2 = int(label[4]), int(label[5])
                pt1 = (x1, y1)
                pt2 = (x2, y2)
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
            #basename = '{:06d}'.format(idOut) 
            basename = os.path.splitext(os.path.split(filenameAP)[1])[0]
            imgFilenameOut = os.path.join(out_path, basename + '.jpg')
            cv2.imwrite(imgFilenameOut, img)
        # print 'FP rate = ' + str((100 * FP) / (FP + TN))
        # print 'Precision = ' + str((100 * TP) / (TP + FP))
        # print 'Recall = ' + str((100 * TP) / (TP + FN))
        idOut = idOut + 1
        i = i+1

    # move back to initial folder
    os.chdir(owd)
    return TP, FP, TN, FN, iouList


if __name__ == "__main__":
    show = 0

    # run detection
    dataset_type = 'test'
    seat_path = '/home/david/data/seat/seat4_285n_q2_' + dataset_type
    im_defect_path = seat_path + '/seat_' + dataset_type + '.txt'
    outPathDefect = seat_path + '/result'

    # read images in folder
    listDefectImgs = [line.rstrip('\n') for line in open(im_defect_path)]
    #listImgs.sort()

    # create result folder
    if not os.path.exists(outPathDefect):
        os.makedirs(outPathDefect)

    # process all images in folder
    TP, FP, TN, FN, iouList = runApp(outPathDefect, listDefectImgs, int(sys.argv[1]), 0.2)

    # diplay results
    print 'Num gt defects = ' + str(TP + FN)
    print 'Num detected defects = ' + str(TP + FP)
    print 'Precision = ' + str((TP) / (TP + FP))
    print 'Recall = ' + str((TP) / (TP + FN))
    iouMean = sum(iouList) / len(iouList)
    print 'IOU = ' + str(iouMean)
