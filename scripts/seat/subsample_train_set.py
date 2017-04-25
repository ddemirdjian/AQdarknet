import os


def readLabelFilenames(train_filename, dbRoot):
    img_filenames = []
    label_filenames = []
    file = open(os.path.join(dbRoot, train_filename), 'r')
    lines = [line.rstrip('\n') for line in file]
    for line in lines:
        img_filenames.append(line)
        idx = os.path.basename(line)
        idx = os.path.splitext(idx)[0]
        label_filename = os.path.join(dbRoot, 'labels', idx + '.txt')
        label_filenames.append(label_filename)
    return label_filenames, img_filenames

if __name__ == '__main__':
    dbRoot = '/mnt/data/seat/seatset2_yolo_color_1055b'
    imgFolderName = 'normals1'
    #imgFolderName = 'color0'

    # read train files
    label_filenames, img_filenames = readLabelFilenames('seat_train.txt', dbRoot)

    # get pos/neg data
    num_pos, num_neg= 0, 0
    posneg_list = []
    for label_filename in label_filenames:
        if os.path.getsize(label_filename) == 0:
            num_neg = num_neg + 1
            posneg_list.append(0)
        else:
            num_pos = num_pos + 1
            posneg_list.append(1)

    # keep only X% of neg data so that pos/neg ratio is
    ratio = 3
    num_neg_max = num_pos * ratio
    i = 0
    num_neg_added = 0
    img_filenames_subsample = []
    for img_filename in img_filenames:
        # set flag
        add_img = 1
        if posneg_list[i] == 0 and num_neg_added >= num_neg_max:
            add_img = 0
        # add image to list
        if add_img > 0:
            img_filenames_subsample.append(img_filename)
            if posneg_list[i] == 0:
                num_neg_added = num_neg_added + 1
        i = i+1

    dstTrainSubFolder = os.path.join(dbRoot, 'seat_train_sub0.txt')
    file = open(dstTrainSubFolder, 'w')
    for img_filename in img_filenames_subsample:
        file.write("%s\n" % img_filename)


