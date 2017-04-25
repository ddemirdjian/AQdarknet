#!/usr/bin/env python
import lxml.etree as et
import numpy as np
import csv, shutil, cv2, os, random, operator, csv

# convert images from a seat folder to yolo format
# - copy images and creates label files
def processFolder(labelFilename, imgFolder, id):
    idStart = id

    # read labels.txt file
    with open(labelFilename) as labelFile:
        # read each label in label file
        labels = list(csv.reader(labelFile, delimiter=' '))
        for label in labels:

            # extract filename and number of labels
            file = label[0]
            numLabels = (len(label) - 1) / 6

            # -- read image, image size
            imgFilename = os.path.join(imgFolder, file)
            img = cv2.imread(imgFilename)
            imgHeight, imgWidth, imgDepth = img.shape

            # --- read labels
            labelNormalized = []
            for k in range(numLabels):
                # set quality
                defectQuality = int(label[5 + 6 * k])
                defectType = int(label[6 + 6 * k])
                x, y = float(label[1 + 6 * k]) / imgWidth, float(label[2 + 6 * k]) / imgHeight
                w, h = float(label[3 + 6 * k]) / imgWidth, float(label[4 + 6 * k]) / imgHeight
                x = x + w/2
                y = y + h/2
                # add label only if normalized coordinates are in image, and quality > 0
                if (x > 0 and x < 1 and y > 0 and y < 1 and defectQuality > 0):
                    tag = 0
                    labelNormalized.append([tag, x, y, w, h])
            numUsableLabels = len(labelNormalized)

            # -- create a label file and save it to labels folder (same level as image JPEG folder)
            labelFile = open(os.path.join(dstLabelFolder, '{:06d}'.format(id) + '.txt'), 'w')
            for k in range(numUsableLabels):
                labelFile.write('%d %f %f %f %f\n' % (int(labelNormalized[k][0]),
                                                      labelNormalized[k][1], labelNormalized[k][2],
                                                      labelNormalized[k][3], labelNormalized[k][4]))

            # -- rename using id, and copy to JPEG folder
            fileExtension = os.path.splitext(file)[1]
            dstFilename = '{:06d}'.format(id) + fileExtension
            shutil.copy2(imgFilename, os.path.join(dstImgFolder, dstFilename))
            id = id + 1

    return id - idStart

def writeImgPath(filename, folder, trainIndices, fileExtension):
    file = open(os.path.join(dbRoot, filename), 'w')
    for idx in trainIndices:
        file.write('%s\n' % os.path.join(folder, '{:06d}'.format(idx) + '.' + fileExtension))

def createIndicesFromMapList(indexMapList):
    indices = []
    for indexMap in indexMapList:
        idx = indexMap[0]
        for i in range(indexMap[1]):
            indices.append(idx)
            idx = idx + 1
    return indices

if __name__ == '__main__':

    # source folder
    seatDBSource = '/data/seat/seatset2v2_1055'

    # dst folder
    trainTestSplit = 0.9
    dbRoot = '/home/david/data/seat/seatset2v2_1055_normals'
    imgFolderName = 'normals1'
    #imgFolderName = 'color0'
    dstImgFolder = os.path.join(dbRoot, 'JPEGImages')
    dstLabelFolder = os.path.join(dbRoot, 'labels')

    # create folders
    print 'Creating folders'
    if not os.path.exists(dstImgFolder):
        os.makedirs(dstImgFolder)
    if not os.path.exists(dstLabelFolder):
        os.makedirs(dstLabelFolder)

    # parse all folder inside seatDBSource
    print 'Reading folder'
    id = 0
    indexMap = []
    dirs = next(os.walk(seatDBSource))[1]
    for dir in dirs:
        normalsFolder = os.path.join(seatDBSource,  dir, dir, imgFolderName)
        labelFilename = os.path.join(normalsFolder, 'labels.txt')
	print 'processing: ' +  normalsFolder
        numLabelProcessed = processFolder(labelFilename, normalsFolder, id)
        if numLabelProcessed > 0:
            indexMap.append([id, numLabelProcessed])
            id = id + numLabelProcessed

    # generate train and test files
    numLabels = len(indexMap)
    random.shuffle(indexMap)
    trainIndices = createIndicesFromMapList(indexMap[0:int(numLabels * trainTestSplit)])
    testIndices = createIndicesFromMapList(indexMap[int(numLabels * trainTestSplit):])

    if imgFolderName == 'color0':
        fileExtension = 'jpg'
    else:
        fileExtension = 'png'
    writeImgPath('seat_train.txt', dstImgFolder, trainIndices, fileExtension)
    writeImgPath('seat_test.txt', dstImgFolder, testIndices, fileExtension)

