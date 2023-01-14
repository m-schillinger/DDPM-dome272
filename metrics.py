# -*- coding: utf-8 -*-
#main.py --model=phiregan --mode=train --data_dir=C:\Users\yuduf\Downloads\WiSoSuper-main\WiSoSuper-main\PhIREGAN\wind_tfrecords\wind_tfrecords --data_type=wind
import os
import math
import numpy as np
import cv2
from skimage.metrics import structural_similarity as SSIM
from PIL import Image
import scipy.stats as stats
from scipy.stats import entropy
import matplotlib.pyplot as plt

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

def power_spectrum(img_path):

    img = cv2.imread(img_path)

    image = split_and_average(img)

    npix = image.shape[0]
    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)**2

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        statistic = "mean",
                                        bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    return kvals, Abins

def get_power_spectrum(folder):
    K = []
    E = []
    for filename in os.listdir(folder):
        k,e= power_spectrum(os.path.join(folder,filename))
        if k is not None:
            K.append(k)
        if e is not None:
            E.append(e)
    K = np.array(K)
    E = np.array(E)    
    K = np.flip(np.mean(K, axis=0))
    E = np.mean(E, axis=0) / 10000
    return K, E

def energy_spectrum(img_path):


    
    img = cv2.imread(img_path)

    image = split_and_average(img)

    npix = image.shape[0]
    ampls = abs(np.fft.fftn(image))/npix
    ek = ampls**2
    ek = np.fft.fftshift(ek)
    ek = ek.flatten()

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    
    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5*(kbins[1:] + kbins[:-1])
    
    ek, _, _ = stats.binned_statistic(knrm, ek,
                                         statistic = "mean",
                                         bins = kbins)
    
    ek *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)


    return kvals, ek

def get_energy_spectrum(folder):
    K = []
    E = []
    for filename in os.listdir(folder):
        k,e= energy_spectrum(os.path.join(folder,filename))
        if k is not None:
            K.append(k)
        if e is not None:
            E.append(e)
    K = np.array(K)
    E = np.array(E)    
    K = np.flip(np.mean(K, axis=0))
    E = np.mean(E, axis=0) / 10000
    return K, E

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
    print('PSNR: ',psnr_val)
    print('SSIM: ',ssim_val)
    print('MSE: ',mse_val)
    print('MAE: ',mae_val)
    
    kt,Et = get_energy_spectrum(tpath)
    kr,Er = get_energy_spectrum(rpath)

    plt.loglog(kt, Et, color='b', label='Ground Truth')
    plt.loglog(kr, Er, color='r', label='Generated Result')
    plt.xlabel("k (wavenumber)")
    plt.ylabel("Kinetic Energy")
    plt.tight_layout()
    plt.title("Energy Spectrum")
    plt.legend()
    plt.savefig("wind_energy_spectrum_ground_truth.png", dpi=1000, transparent=True, bbox_inches='tight')
    plt.show()
    
    