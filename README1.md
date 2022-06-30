# Medical_image_code
import matplotlib.pyplot as plt
from pydicom.filereader import dcmread

dataset = dcmread("images/MRI_images/CT_small.dcm")
img=dataset.pixel_array
plt.imshow(img, cmap=plt.cm.bone)
plt.imsave("images/MRI_images/dcm_to_tiff_converted.png", img, cmap='gray')

#Gaussian
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio
from matplotlib import pyplot as plt
from skimage import io
from scipy import ndimage as nd

noisy_img = img_as_float(io.imread("images/MRI_images/MRI_noisy.tif"))
#Need to convert to float as we will be doing math on the array
#Also, most skimage functions need float numbers
ref_img = img_as_float(io.imread("images/MRI_images/MRI_clean.tif"))
gaussian_img = nd.gaussian_filter(noisy_img, sigma=5)
plt.imshow(gaussian_img, cmap='gray')
plt.imsave("images/MRI_images/Gaussian_smoothed.png", gaussian_img, cmap='gray')

noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
gaussian_cleaned_psnr = peak_signal_noise_ratio(ref_img, gaussian_img)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", gaussian_cleaned_psnr)

#Bilateral, TV and Wavelet
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import img_as_float

noisy_img = img_as_float(io.imread("images/MRI_images/MRI_noisy.tif"))
sigma_est = estimate_sigma(noisy_img, multichannel=True, average_sigmas=True)

denoise_bilateral = denoise_bilateral(noisy_img, sigma_spatial=15,
                multichannel=False)

noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
bilateral_cleaned_psnr = peak_signal_noise_ratio(ref_img, denoise_bilateral)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", bilateral_cleaned_psnr)

plt.imsave("images/MRI_images/bilateral_smoothed.png", denoise_bilateral, cmap='gray')





###### TV ###############
denoise_TV = denoise_tv_chambolle(noisy_img, weight=0.3, multichannel=False)
noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
TV_cleaned_psnr = peak_signal_noise_ratio(ref_img, denoise_TV)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", TV_cleaned_psnr)
plt.imsave("images/MRI_images/TV_smoothed.png", denoise_TV, cmap='gray')

####Wavelet #################
wavelet_smoothed = denoise_wavelet(noisy_img, multichannel=False,
                           method='BayesShrink', mode='soft',
                           rescale_sigma=True)
noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
Wavelet_cleaned_psnr = peak_signal_noise_ratio(ref_img, wavelet_smoothed)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", Wavelet_cleaned_psnr)

plt.imsave("images/MRI_images/wavelet_smoothed.png", wavelet_smoothed, cmap='gray')






import matplotlib.pyplot as plt

from skimage.restoration import denoise_wavelet, cycle_spin
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
from skimage import io


noisy_img = img_as_float(io.imread("images/MRI_images/MRI_noisy.tif"))
ref_img = img_as_float(io.imread("images/MRI_images/MRI_clean.tif"))


denoise_kwargs = dict(multichannel=False, wavelet='db1', method='BayesShrink',
                      rescale_sigma=True)

all_psnr = []
max_shifts = 3     #0, 1, 3, 5

Shft_inv_wavelet = cycle_spin(noisy_img, func=denoise_wavelet, max_shifts = max_shifts,
                            func_kw=denoise_kwargs, multichannel=False)

noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
shft_cleaned_psnr = peak_signal_noise_ratio(ref_img, Shft_inv_wavelet)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", shft_cleaned_psnr)

plt.imsave("images/MRI_images/Shift_Inv_wavelet_smoothed.png", Shft_inv_wavelet, cmap='gray')





import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio
import bm3d
import numpy as np

noisy_img = img_as_float(io.imread("images/MRI_images/MRI_noisy.tif", as_gray=True))
ref_img = img_as_float(io.imread("images/MRI_images/MRI_clean.tif"))


BM3D_denoised_image = bm3d.bm3d(noisy_img, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.ALL_STAGES)
#BM3D_denoised_image = bm3d.bm3d(noisy_img, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)

 #Also try stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING                     


noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
BM3D_cleaned_psnr = peak_signal_noise_ratio(ref_img, BM3D_denoised_image)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", BM3D_cleaned_psnr)


plt.imshow(BM3D_denoised_image, cmap='gray')
plt.imsave("images/MRI_images/BM3D_denoised.png", BM3D_denoised_image, cmap='gray')
