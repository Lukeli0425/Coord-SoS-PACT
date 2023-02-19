import sys
import logging
import os
from tkinter import W
# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH
sys.path.append("D:\Tsinghua\Biooptics\K-wave\k-wave-python-master\k-wave-python-master")
from kwave.kgrid import kWaveGrid
from numpy.fft import fftshift,fftn,ifftn,ifftshift
from scipy.interpolate import interp2d
from kwave.utils.interputils import interpolate2D
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import cv2


filepath = "D:\Tsinghua\Biooptics/linear_array/data/linear/1.mat"
data = scio.loadmat(filepath)
sinogram = data['linear'].T
sinogram = np.concatenate((np.flip(sinogram,0),sinogram[1:,:]))
Nt,Ny = sinogram.shape
c = 1500 #sound speed
dt = 2e-8
dy = 1e-4 #换能器间隔
kgrid =  kWaveGrid([Nt,  Ny], [dt * c,dy])
w = kgrid.kx*c
w_new = kgrid.k*c
w = np.complex128(w)
c0 = np.complex128(kgrid.ky)
sf = (c**2)*((np.sqrt((w/c)**2-c0**2)))/2/w
sf= np.nan_to_num(sf,nan=c/2) 
p = sf*fftshift(fftn(ifftshift(sinogram),axes=[0,0]))
p[np.abs(w)<np.abs(c*kgrid.ky)]=0
# p = interpolate2D([w,0.1*c0],p,[w_new,c0])
p = np.nan_to_num(p,nan=0)
p = np.real(fftshift(ifftn(ifftshift(p),axes=[1])));
p = p[np.int16(((Nt+1)/2)+1):Nt,:]
p = 2*2*p/c
Nx_recon, Ny_recon = p.shape
kgrid = kWaveGrid([128,128],[1e-4,1e-4])
kgrid_recon = kWaveGrid([Nx_recon,Ny_recon], [dt * 1500,  1e-4])
p_xy_rs = interpolate2D([kgrid_recon.y.T, (kgrid_recon.x - kgrid_recon.x.min()).T], p.T, [kgrid.y.T, (kgrid.x - kgrid.x.min()).T],copy_nans=False)
p_1 =p_xy_rs.T
# # # pa = p.copy()
plt.imshow(p_1)
plt.show()