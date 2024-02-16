import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tempfile import gettempdir

import matplotlib.pyplot as plt

from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC, kspaceFirstOrder2DG
from kwave.ktransducer import *
from kwave.utils import *
from kwave.utils import dotdict

# pathname for the input and output files
pathname = gettempdir()

# =========================================================================
# SIMULATION
# =========================================================================

# create the computational grid
PML_size = 20               # size of the PML in grid points
Nx = 128 - 2 * PML_size     # number of grid points in the x (row) direction
Ny = 256 - 2 * PML_size     # number of grid points in the y (column) direction
dx = 0.1e-3                 # grid point spacing in the x direction [m]
dy = 0.1e-3                 # grid point spacing in the y direction [m]
kgrid = kWaveGrid([Nx, Ny], [dx, dy])

# define the properties of the propagation medium
medium = kWaveMedium(sound_speed=1500)

# create initial pressure distribution using makeDisc
disc_magnitude = 5 # [Pa]
disc_x_pos = 60    # [grid points]
disc_y_pos = 140  	# [grid points]
disc_radius = 5    # [grid points]
disc_2 = disc_magnitude * makeDisc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius)

disc_x_pos = 30    # [grid points]
disc_y_pos = 110 	# [grid points]
disc_radius = 8    # [grid points]
disc_1 = disc_magnitude * makeDisc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius)

source = kSource()
source.p0 = disc_1 + disc_2

# smooth the initial pressure distribution and restore the magnitude
source.p0 = smooth(source.p0, True)

# define a binary line sensor
sensor_mask = np.zeros((Nx, Ny))
sensor_mask[0, :] = 1
sensor = kSensor(sensor_mask)

# create the time array
kgrid.makeTime(medium.sound_speed)

# set the input arguements: force the PML to be outside the computational
# grid switch off p0 smoothing within kspaceFirstOrder2D
input_args = {
    'PMLInside': False,
    'PMLSize': PML_size,
    'Smooth': False,
    'SaveToDisk': os.path.join(pathname, f'example_input.h5'),
    'SaveToDiskExit': False
}

# run the simulation
sensor_data = kspaceFirstOrder2DC(**{
    'medium': medium,
    'kgrid': kgrid,
    'source': source,
    'sensor': sensor,
    **input_args
})
# assert compare_against_ref(f'out_pr_2D_FFT_line_sensor', input_args['SaveToDisk']), 'Files do not match!'

plt.figure()
plt.imshow(sensor_data)
plt.savefig('./data/output/d1_2D.png')