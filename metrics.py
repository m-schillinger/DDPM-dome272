# -*- coding: utf-8 -*-
#main.py --model=phiregan --mode=train --data_dir=C:\Users\yuduf\Downloads\WiSoSuper-main\WiSoSuper-main\PhIREGAN\wind_tfrecords\wind_tfrecords --data_type=wind
import os
import math
import numpy as np
import cv2
from skimage.metrics import structural_similarity as SSIM

def read_image(folder):
    '''
    Returns:
    images: numpy.ndarray: Image exists in "path"
    size: number of image
    dimension: Image dimension (number of rows and columns)
    '''
    
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    images = np.array(images)
    size=images.shape[0]
    dimension=(images[0].shape[0], images[0].shape[1])
    return images, size, dimension


def psnr(imageA, imageB):
    mse = np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)
    # MSE is zero means no noise is present in the signal and PSNR has no importance.
    if(mse == 0):  
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def mse(imageA, imageB):
    mse_error = np.mean((imageA.astype("float") - imageB.astype("float")) ** 2) # *0.5
    return mse_error

def mae(imageA, imageB):
    mae = np.mean(np.absolute((imageB.astype("float") - imageA.astype("float"))))
    #mae /= float(imageA.shape[0] * imageA.shape[1] * 255)
    if (mae < 0):
        return mae * -1
    else:
        return mae

def ssim(imageA, imageB):
    return SSIM(imageA, imageB, multichannel=True)

if __name__ == '__main__':
    tpath = "./metric_test_true" #ground truth path
    rpath = "./metric_test_result" #result image path
    img_t, size , _= read_image(tpath)
    img_r, _ , _= read_image(rpath)
    
    p=[]
    s=[]
    for i in range(size):
        p.append(psnr(img_t[i], img_r[i]))
        s.append(ssim(img_t[i], img_r[i]))
    p=np.array(p)
    s=np.array(s)
    psnr_val = np.mean(p)
    ssim_val = np.mean(s)
    mse_val = mse(img_t,img_r)
    mae_val = mae(img_t,img_r)
    print(psnr_val)
    print(ssim_val)
    print(mse_val)
    print(mae_val)
    