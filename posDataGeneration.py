import os
from natsort import natsorted
from scipy import misc
import numpy as np

masks_path = './data/masks/'
output_masks_path = './data/generatedMasks/grader/'
output_img_path = './data/generatedMasks/img/'
output_gt_path = './data/generatedMasks/gt/'
outputPathResults = './results/'

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

f = open(outputPathResults + 'genData.txt', mode='w')
f.write('Dataset,Generated Data Grader 1,Generated Data Grader2\n')

for dataset in datasetLst:
    label1_array = []
    label2_array = []
    counter1 = 0
    counter2 = 0
    set_array = []
    for idx in xrange(len(grader1Lst)):
        if dataset in grader1Lst[idx]:
            set_array.append(grader1Lst[idx])
            mask1 = misc.imread(masks_path + dataset + '/' + grader1Lst[idx])
            mask2 = misc.imread(masks_path + dataset + '/' + grader2Lst[idx])
            if np.sum(mask1) > 0:
                img_name = grader1Lst[idx].split('-')
                img = misc.imread(masks_path + dataset + '/' + img_name[0] + '-' + img_name[1] + '.png')
                misc.imsave(output_img_path + img_name[0] + '-' + img_name[1] + '_genimg.png', np.fliplr(img))
                misc.imsave(output_masks_path + img_name[0] + '-' + img_name[1] + '_grader1_genmask.png',
                            np.fliplr(mask1))
                counter1 += 1
            if np.sum(mask2) > 0:
                img_name = grader2Lst[idx].split('-')
                img = misc.imread(masks_path + dataset + '/' + img_name[0] + '-' + img_name[1] + '.png')
                misc.imsave(output_img_path + img_name[0] + '-' + img_name[1] + '_genimg.png',
                            np.fliplr(img))
                misc.imsave(
                    output_masks_path + img_name[0] + '-' + img_name[1] + '_grader2_genmask.png',
                    np.fliplr(mask2))
                counter2 += 1
            if np.sum(mask1) > 0 or np.sum(mask2) > 0:
                gt_overlap = np.logical_and(np.fliplr(mask1).astype(np.bool), np.fliplr(mask2).astype(np.bool))
                gt_mask = gt_overlap.astype(np.uint8)
                name_split = str.split(grader1Lst[idx], '-')
                misc.imsave(output_gt_path + name_split[0] + '-' + name_split[1] + '-gt_genmask.png', gt_mask)

    f.write(str(dataset) + ',' + str(counter1) + ',' + str(counter2) + '\n')
