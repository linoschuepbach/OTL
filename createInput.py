import os
from natsort import natsorted
from scipy import misc
from scipy.ndimage.interpolation import zoom
import numpy as np
from sklearn.model_selection import train_test_split

input_augmask_path = './data/generatedMasks/gt/'
input_augimg_path = './data/generatedMasks/img/'
input_mask_path = './data/gt/'
input_img_path = './data/masks/img/'
output_npy = './data/input/'


def getFiles(path_to_files):
    file_list = []
    for dirName, subdirList, fileList in os.walk(path_to_files):
        for filename in fileList:
            if 'grader_1' in filename.lower():
                continue
            elif 'grader_2' in filename.lower():
                continue
            else:
                file_list.append(path_to_files + filename)
    file_list = natsorted(file_list)
    return file_list


def readFiles(mask_list, image_cols, image_rows):
    obj_array = []
    for idx in mask_list:
        mask = misc.imread(idx)
        col, row = mask.shape[:2]
        zMask = zoom(mask, [image_cols / float(col), image_rows / float(row)])
        zMask = np.array([zMask])
        obj_array.append(zMask)
    return np.asarray(obj_array)


img_list = getFiles(input_img_path)
img_list = np.asarray(img_list)
tmp = getFiles(input_augimg_path)
img_list = np.append(img_list, np.asarray(tmp))

mask_list = getFiles(input_mask_path)
mask_list = np.asarray(mask_list)
tmp = getFiles(input_augmask_path)
mask_list = np.append(mask_list, np.asarray(tmp))

x_train, x_test, y_train, y_test = train_test_split(img_list, mask_list, test_size=0.1)
np.save(output_npy + 'x_train_id.npy', x_train)
np.save(output_npy + 'x_test_id.npy', x_test)
np.save(output_npy + 'y_train_id.npy', y_train)
np.save(output_npy + 'y_test_id.npy', y_test)

x_train_array = readFiles(x_train, 256, 256)
x_test_array = readFiles(x_test, 256, 256)
y_train_array = readFiles(y_train, 256, 256)
y_test_array = readFiles(y_test, 256, 256)

np.save(output_npy + 'x_train.npy', x_train_array)
np.save(output_npy + 'x_test.npy', x_test_array)
np.save(output_npy + 'y_train.npy', y_train_array)
np.save(output_npy + 'y_test.npy', y_test_array)
