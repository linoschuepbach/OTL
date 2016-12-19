from __future__ import print_function
from natsort import natsorted
from scipy import misc
import os
import dicom
import numpy as np
import shutil
import matplotlib.pyplot as plt

# TODO check influence of dicom origin (1,1) in regards of the the label coordinates
# flag = 1: Create the overlay masks to see if everything ok
overlayFlag = 0
# Find all Dicom/XML files in the given folder and include their path/names into a list
PathDicom = "./data/dicom"
output_mask_path = './data/masks/mask/'
output_img_path = './data/masks/img/'
lstFilesDCM = []  # create an empty list
lstFilesXML = []
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName, filename))
        if ".xml" in filename.lower():  # check whether the file's XML
            lstFilesXML.append(os.path.join(dirName, filename))
lstFilesDCM = natsorted(lstFilesDCM)
lstFilesXML = natsorted(lstFilesXML)

# Loop over the found Dicom files and extract the data
for dcmDataset in xrange(len(lstFilesDCM)):
    dcmFile = dicom.read_file(lstFilesDCM[dcmDataset])
    dcmFile.SamplesPerPixel = 1
    dcmArray = dcmFile.pixel_array

    dcmRows = dcmArray.shape[1]
    dcmColumns = dcmArray.shape[2]
    projectName = os.path.dirname(lstFilesDCM[dcmDataset]).split('/')[3]

    # find the corresponding grader files for this scan
    graderFiles = []
    for l in lstFilesXML:
        if projectName in str(l):
            graderFiles.append(lstFilesXML.index(l))
    graderFiles = np.asarray(graderFiles, dtype='uint')

    # Check if directory exists, if yes delete and remake it
    outputPath = './data/masks/devices/' + projectName + '/'
    if os.path.exists(outputPath):
        shutil.rmtree(outputPath)
        os.mkdir(outputPath)
    else:
        os.mkdir(outputPath)

    for grader in xrange(len(graderFiles)):  # Loop over the Graders
        # Get the coordinates from the xml file into numpy
        coordList = []
        with open(lstFilesXML[graderFiles[grader]]) as f:
            for line in f:
                coordList.append(line.split(','))
        coordList = np.asarray(coordList, dtype='uint')
        coordList -= 1

        for ii in xrange(len(dcmArray)):  # Loop over the bscans and save the image and the labelled mask
            mask = np.zeros([dcmRows, dcmColumns])
            if grader == 0:
                misc.imsave(outputPath + projectName + '-%02d.png' % int(ii + 1), dcmArray[ii])
                misc.imsave(output_img_path + projectName + '-%02d.png' % int(ii + 1), dcmArray[ii])
            if coordList[np.where(coordList[:, 2] == ii)].size == 0:
                misc.imsave(outputPath + projectName + '-%02d-Grader_%d.png' % (int(ii + 1), grader + 1), mask)
                misc.imsave(output_mask_path + projectName + '-%02d-Grader_%d.png' % (int(ii + 1), grader + 1), mask)
            else:
                labelArray = coordList[np.where(coordList[:, 2] == ii)]
                mask[labelArray[:, 0], labelArray[:, 1]] = 1
                misc.imsave(outputPath + projectName + '-%02d-Grader_%d.png' % (int(ii + 1), grader + 1), mask)
                misc.imsave(output_mask_path + projectName + '-%02d-Grader_%d.png' % (int(ii + 1), grader + 1), mask)

            if overlayFlag:
                if np.sum(mask) > 0:
                    plt.figure()
                    plt.subplot(1, 2, 1)
                    plt.imshow(dcmArray[ii], 'gray', interpolation='none')
                    plt.subplot(1, 2, 2)
                    plt.imshow(dcmArray[ii], 'gray', interpolation='none')
                    plt.imshow(mask, 'jet', interpolation='none', alpha=0.5)
                    plt.show()
