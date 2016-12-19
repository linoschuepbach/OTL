import os
from natsort import natsorted
from scipy import misc
import numpy as np

smooth = 1


def dice_coef(X, Y):
    intersection = np.logical_and(X, Y)
    return 2. * intersection.sum() / (X.sum() + Y.sum())


masks_path = './data/masks/devices/'
outputPathResults = './results/'
gt_path = './data/gt/'
imgLst = []
grader1Lst = []
grader2Lst = []
datasetLst = []

for dirName, subdirList, fileList in os.walk(masks_path):
    for dataset in subdirList:
        datasetLst.append(dataset)
    for filename in fileList:
        if 'grader_1' in filename.lower():
            grader1Lst.append(filename)
        elif 'grader_2' in filename.lower():
            grader2Lst.append(filename)
        else:
            imgLst.append(filename)

imgLst = natsorted(imgLst)
grader1Lst = natsorted(grader1Lst)
grader2Lst = natsorted(grader2Lst)
datasetLst = natsorted(datasetLst)
total_pos_counter = 0
total_neg_counter = 0

f = open(outputPathResults + 'dataOverview.txt', mode='w')
f.write(
    'Dataset,Nr. of Images,Nr. of pos. Labels grader 1,Nr. of neg. Labels grader 1,Nr. of pos. Labels grader 2,Nr. of neg. Labels grader 2,Overlap between G1 and G2,Overlap between G1 and GT,Overlap between G2 and GT,ratio1,ratio2\n')

for dataset in datasetLst:
    set_img_counter = 0
    set_pos_counter1 = 0
    set_neg_counter1 = 0
    set_pos_counter2 = 0
    set_neg_counter2 = 0
    set_array = []
    label1_array = []
    label2_array = []
    gt_array = []
    ratio_array1 = []
    ratio_array2 = []
    for idx in xrange(len(grader1Lst)):
        if dataset in grader1Lst[idx]:
            set_array.append(grader1Lst[idx])
            mask1 = misc.imread(masks_path + dataset + '/' + grader1Lst[idx])
            mask2 = misc.imread(masks_path + dataset + '/' + grader2Lst[idx])
            if np.sum(mask1) > 0:
                set_pos_counter1 += 1
                ratio_array1.append(np.count_nonzero(mask1) / np.float(mask1.size) * 100)
            else:
                set_neg_counter1 += 1
            label1_array.append(mask1)

            if np.sum(mask2) > 0:
                set_pos_counter2 += 1
                ratio_array2.append(np.count_nonzero(mask2) / np.float(mask2.size) * 100)
            else:
                set_neg_counter2 += 1
            label2_array.append(mask2)
            gt_overlap = np.logical_and(mask1.astype(np.bool), mask2.astype(np.bool))
            gt_mask = gt_overlap.astype(np.uint8)
            name_split = str.split(grader1Lst[idx], '-')
            misc.imsave(gt_path + name_split[0] + '-' + name_split[1] + '-gt_mask.png', gt_mask)
            gt_array.append(gt_overlap)

    label1_array = np.array(label1_array).astype(np.bool)
    label2_array = np.array(label2_array).astype(np.bool)
    ratio_array1 = np.array(ratio_array1)
    ratio_array2 = np.array(ratio_array2)
    ratio1 = np.mean(ratio_array1)
    ratio2 = np.mean(ratio_array2)
    gt_array = np.array(gt_array)
    dice = dice_coef(label1_array, label2_array)
    dsc_gt1 = dice_coef(label1_array, gt_array)
    dsc_gt2 = dice_coef(label2_array, gt_array)
    set_img_counter = len(set_array)
    f.write(str(dataset) + ',' + str(set_img_counter) + ',' + str(set_pos_counter1) + ',' + str(
        set_neg_counter1) + ',' + str(set_pos_counter2) + ',' + str(set_neg_counter2) + ',' + str(dice) + ',' + str(
        dsc_gt1) + ',' + str(dsc_gt2) + ',' + str(ratio1) + ',' + str(ratio2) + '\n')

f.close()
