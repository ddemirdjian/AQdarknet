#!/usr/bin/env python
import lxml.etree as et
import numpy as np
import csv, shutil, cv2, os, random, operator, csv

def generateVOCXML(xmlFilename, subDirname, imgFilename, imgWidth, imgHeight, imgDepth, boundingBoxList):
    '''
    Write an xml file (VOC format) containing labeling information
    @param xmlFilename: XML filename
    @param subDirname:  Folder containing image file
    @param imgFilename: Image filename
    @param imgWidth: Image width
    @param imgHeight: Image height
    @param imgDepth: Image depth
    @param boundingBoxList: List of bounding boxes associated with image
    @return:
    '''

    # set root, folder, filename
    root = et.Element('annotation')
    folder = et.SubElement(root, 'folder')
    filename = et.SubElement(root, 'filename')
    folder.text = subDirname
    filename.text = imgFilename

    # set image size
    size = et.SubElement(root, 'size')
    width = et.SubElement(size, 'width')
    height = et.SubElement(size, 'height')
    depth = et.SubElement(size, 'depth')
    width.text = str(imgWidth)
    height.text = str(imgHeight)
    depth.text = str(imgDepth)

    # set label for all bounding boxes
    for boundingBox in boundingBoxList:
        object = et.SubElement(root, 'object')
        name = et.SubElement(object, 'name')
        pose = et.SubElement(object, 'pose')
        truncated = et.SubElement(object, 'truncated')
        difficult = et.SubElement(object, 'difficult')
        bndbox = et.SubElement(object, 'bndbox')
        xmin = et.SubElement(bndbox, 'xmin')
        ymin = et.SubElement(bndbox, 'ymin')
        xmax = et.SubElement(bndbox, 'xmax')
        ymax = et.SubElement(bndbox, 'ymax')
        name.text = 'defect'
        pose.text = 'Front'
        truncated.text = str(0)
        difficult.text = str(0)
        xmin.text = str(int(imgWidth * (boundingBox[1] - boundingBox[3]/2)))
        ymin.text = str(int(imgHeight * (boundingBox[2] - boundingBox[4]/2)))
        xmax.text = str(int(imgWidth * (boundingBox[1] + boundingBox[3]/2)))
        ymax.text = str(int(imgHeight * (boundingBox[2] + boundingBox[4]/2)))

    # write to file
    tree = et.ElementTree(root)
    tree.write(xmlFilename, pretty_print=True, xml_declaration=True)

# convert images from a seat folder to yolo format
# - copy images and creates label files
def processFolder(labelFilename, imgFolder, id, xmlMode):
    idStart = id

    # read labels.txt file
    dstFilenameList = []
    tagFilenameList = []
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
            bboxList = []
            for k in range(numLabels):
                # set quality
                defectQuality = int(label[5 + 6 * k])
                defectType = int(label[6 + 6 * k])
                x, y = float(label[1 + 6 * k]) / imgWidth, float(label[2 + 6 * k]) / imgHeight
                w, h = float(label[3 + 6 * k]) / imgWidth, float(label[4 + 6 * k]) / imgHeight
                x = x + w/2
                y = y + h/2
                # add label only if normalized coordinates are in image, and quality > 0
                if (x-w/2 > 0 and x+w/2 < 1 and y-h/2 > 0 and y+h/2 < 1 and defectQuality > 0):
                    tag = 0
                    bboxList.append([tag, x, y, w, h])
            numUsableLabels = len(bboxList)

            # -- create a label file and save it to labels folder (same level as image JPEG folder)
            fileExtension = os.path.splitext(file)[1]
            labelBasename = os.path.basename(os.path.split(imgFolder)[0]) + '_' + '{:06d}'.format(id)
            dstFilename = labelBasename + fileExtension
            labelFile = open(os.path.join(dstLabelFolder, labelBasename + '.txt'), 'w')
            if xmlMode:
               # create xml file 
               generateVOCXML(labelFile, imgFolder, dstFilename, imgWidth, imgHeight, imgDepth, bboxList)
            else:
	       # create simple text file
               for k in range(numUsableLabels):
                  labelFile.write('%d %f %f %f %f\n' % (int(bboxList[k][0]),
                                                      bboxList[k][1], bboxList[k][2],
                                                      bboxList[k][3], bboxList[k][4]))

	    # generate boxes and display
            #boxColor = (255, 0, 0)
            #for k in range(numUsableLabels):
	    #   pt1 = (int(imgWidth * (bboxList[k][1] - bboxList[k][3]/2)), int((bboxList[k][2] - bboxList[k][4]/2)* imgHeight))
	    #   pt2 = (int(imgWidth * (bboxList[k][1] + bboxList[k][3]/2)), int((bboxList[k][2] + bboxList[k][4]/2)* imgHeight))
            #   cv2.rectangle(img, pt1, pt2, boxColor, 2)
            #if numUsableLabels>0:
	    #   cv2.imshow("data", img)
            #   cv2.waitKey(0)

            # -- rename using id, and copy to JPEG folder
            dstFilenameList.append(os.path.join(dstImgFolder, dstFilename))
            tagFilenameList.append(os.path.join(dstLabelFolder, labelBasename + '.txt'))
            shutil.copy2(imgFilename, os.path.join(dstImgFolder, dstFilename))
            id = id + 1

    return id - idStart, dstFilenameList, tagFilenameList

def writeImgXMLPath(filename, imgFolder, xmlFolder, trainIndices, imgFilenameList, tagFilenameList):
    file = open(os.path.join(dbRoot, filename), 'w')
    for idx in trainIndices:
        print imgFilenameList[idx]
        print tagFilenameList[idx]
        file.write('%s %s\n' % (os.path.join(imgFolder, imgFilenameList[idx]), os.path.join(xmlFolder, tagFilenameList[idx])))

#def writeImgPath(filename, imgFolder, trainIndices, fileExtension):
#    file = open(os.path.join(dbRoot, filename), 'w')
#    for idx in trainIndices:
#        basename = '{:06d}'.format(idx)
#        file.write('%s\n' % os.path.join(imgFolder, basename + '.' + fileExtension))

def writeImgPath(filename, imgFolder, trainIndices, filenameList):
    file = open(os.path.join(dbRoot, filename), 'w')
    for idx in trainIndices:
        basename = filenameList[idx]
        file.write('%s\n' % os.path.join(imgFolder, basename))

def createIndicesFromMapList(indexMapList):
    indices = []
    for indexMap in indexMapList:
        idx = indexMap[0]
        for i in range(indexMap[1]):
            indices.append(idx)
            idx = idx + 1
    return indices


if __name__ == '__main__':

  for set in ['train','test']:

    # source folder
    #seatDBSource = '/data/seat/seatset4_v3_LL_LB/normalrender' # '/data/seat/seatset2v2_1055'
    seatDBSource = '/data/seat/rend' 
    #lindexSource = '/home/david/dev/NitrogenApps/nitrogen/apps/multicamscan/datasetLists/'
    indexSource = '/data/seat/seatset4_v3_LL_LB/'
    if set == 'train':
	indexSource = indexSource + 'train.txt' #'scansAvailableTrain.txt'
    else:
	indexSource = indexSource + 'test.txt' #'scansAvailableTest.txt'

    # dst folder
    xmlMode = False
    dbRoot = '/home/david/data/seat/seat4n_' + set
    indexFilename = 'seat_' + set + '.txt'
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
    dirs = [line.rstrip('\n').split()[0] for line in open(indexSource)]
    dstFilenameFullList = []
    tagFilenameFullList = []
    for dir in dirs:
        normalsFolder = os.path.join(seatDBSource,  dir, imgFolderName)
        labelFilename = os.path.join(normalsFolder, 'labels.txt')
	print 'processing: ' +  normalsFolder
        numLabelProcessed, dstFilenameList, tagFilenameList = processFolder(labelFilename, normalsFolder, id, xmlMode)
        if numLabelProcessed > 0:
            indexMap.append([id, numLabelProcessed])
            id = id + numLabelProcessed
            dstFilenameFullList = dstFilenameFullList + dstFilenameList
            tagFilenameFullList = tagFilenameFullList + tagFilenameList

    # generate train files
    numLabels = len(indexMap)
    random.shuffle(indexMap)
    trainIndices = createIndicesFromMapList(indexMap)

    if imgFolderName == 'color0':
        fileExtension = 'jpg'
    else:
        fileExtension = 'png'
   
    if xmlMode:
       writeImgXMLPath(indexFilename, dstImgFolder, dstLabelFolder, trainIndices, dstFilenameFullList, tagFilenameFullList)
    else: 
       writeImgPath(indexFilename, dstImgFolder, trainIndices, dstFilenameFullList)
