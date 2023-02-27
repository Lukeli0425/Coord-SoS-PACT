import sys
import logging
import os
from tkinter import W
# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH
sys.path.append("D:\Tsinghua\Biooptics\K-wave\k-wave-python-master\k-wave-python-master")
from kwave.kgrid import kWaveGrid
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC,kspaceFirstOrder2DG
from kwave.utils.maputils import makeDisc
from kwave.utils.matrixutils import smooth
from kwave.utils import dotdict
from kwave.ktransducer import *
from kwave.kmedium import kWaveMedium
from tempfile import gettempdir
import matplotlib.pyplot as plt


filepath = "./data/input/sinogram.mat"
data = scio.loadmat(filepath)
sinogram = data['linear']
print(sinogram.shape)
pathname = gettempdir()
# create the computational grid
PML_size = 10              # size of the PML in grid points
Nx = 148 - 2 * PML_size    # number of grid points in the x direction
Ny = 148 - 2 * PML_size    # number of grid points in the y direction
dx = 0.1e-3                # grid point spacing in the x direction [m]
dy = 0.1e-3                # grid point spacing in the y direction [m]
kgrid = kWaveGrid([Nx, Ny], [dx, dy])

# define the properties of the propagation medium
medium = kWaveMedium(sound_speed=1500)

# assign to the source structure
source = kSource()
source.p0 = np.zeros((Nx,Ny))
# create initial pressure distribution using makeDisc
# disc_magnitude = 5 # [Pa]
# disc_x_pos = 60    # [grid points]
# disc_y_pos = 80  	# [grid points]
# disc_radius = 5    # [grid points]
# disc_2 = disc_magnitude * makeDisc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius)

# disc_x_pos = 30    # [grid points]
# disc_y_pos = 50 	# [grid points]
# disc_radius = 8    # [grid points]
# disc_1 = disc_magnitude * makeDisc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius)

# source = kSource()
# source.p0 = disc_1 + disc_2
# define a binary line sensor
# p_m = np.ones((2,128))
# p_m[0,:]=np.ones((1,128))*6.3*1e-3
# p_m[1,:]=np.linspace(-6.35e-3,6.35e-3,128)
sensor_mask = np.zeros((Nx, Ny))
sensor_mask[1, :] = 1
sensor = kSensor(sensor_mask)

# # sensor.mask=p_m.T
# sensor.p_mask=p_m
# create the time array
kgrid.makeTime(medium.sound_speed)
kgrid.Nt = 708
kgrid.dt = 2e-8
sensor.time_reversal_boundary_data = sinogram

# set the input arguements: force the PML to be outside the computational
# grid switch off p0 smoothing within kspaceFirstOrder2D
input_args = {
    'PMLInside': False,
    'PMLSize': PML_size,
    'Smooth': False,
    'SaveToDisk': os.path.join(pathname, f'input.h5'),
    'SaveToDiskExit': False
}
recon = kspaceFirstOrder2DG(**{
        'medium': medium,
        'kgrid': kgrid,
        'source': source,
        'sensor': sensor,
        **input_args
    })

plt.figure()
plt.imshow(recon)
plt.savefig('./data/output/d1_recon_2D.png')