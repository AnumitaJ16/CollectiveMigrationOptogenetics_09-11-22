# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 21:26:21 2022

@author: anumi
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure

#%%
def renameFiles(DirPath, startingNo = 0):
    """

    :param DirPath: Path of the diretory with the files you want to rename
    :type DirPath: string
    :return: Renames files with numbers to be easily callable in Python for manipulation later. Usually .tif files.

    """
    files = os.listdir(DirPath)
    i = startingNo
    j = 0
    for file in files:
        if 'Copy' in file:
            if i < 10:
                os.rename(os.path.join(DirPath,file), os.path.join(DirPath, '000'+str(i)+'.tif'))
                i+=1
            elif i >= 10 and i < 100:
                os.rename(os.path.join(DirPath,file), os.path.join(DirPath, '00'+str(i)+'.tif'))
                i+=1
            elif i >= 100:
                os.rename(os.path.join(DirPath,file), os.path.join(DirPath, '0'+str(i)+'.tif'))
                i+=1
        else:
            file = str(j)+'.tif'
            if i < 10:
                os.rename(os.path.join(DirPath,file), os.path.join(DirPath, '000'+str(i)+'.tif'))
                i+=1
                j+=1
            elif i >= 10 and i < 100:
                os.rename(os.path.join(DirPath,file), os.path.join(DirPath, '00'+str(i)+'.tif'))
                i+=1
                j+=1
            elif i >= 100:
                os.rename(os.path.join(DirPath,file), os.path.join(DirPath, '0'+str(i)+'.tif'))
                i+=1
                j+=1

def getLineMask(DirLines, speed):
    dirPath = DirLines+'/'+speed
    file_spec = '*.tif'
    load_pattern = os.path.join(dirPath, file_spec)
    ic = io.imread_collection(load_pattern)
    imgArray = np.uint8(io.concatenate_images(ic))
    
    lineMask = np.mean(imgArray, axis = 2).T
    imgArray = lineMask.copy()
    
    imgArray.max()
    imgArray.min()
    imgArray_scaled = exposure.rescale_intensity(imgArray)
    imgArray_scaled.max()
    imgArray_scaled.min()
    scaledArray = np.round(imgArray_scaled)
    plt.imshow(scaledArray)
    return scaledArray

