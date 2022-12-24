import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tempfile import gettempdir
from kwave.kgrid import kWaveGrid
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC,kspaceFirstOrder3DG
from kwave.utils import *
from kwave.ktransducer import *
from kwave.kmedium import kWaveMedium
import matplotlib.pyplot as plt
import scipy.io as scio
from scipy.io import savemat
pathname = gettempdir()

# =========================================================================
# SIMULATION
# =========================================================================

# change scale to 2 to reproduce the higher resolution figures used in the help file
scale = 1

# create the computational grid
PML_size = 10                   # size of the PML in grid points
Nx = 148 * scale - 2 * PML_size  # number of grid points in the x direction
Ny = 148 * scale - 2 * PML_size  # number of grid points in the y direction
Nz = 32 * scale - 2 * PML_size  # number of grid points in the z direction
dx = 0.1e-3 / scale             # grid point spacing in the x direction [m]
dy = 0.1e-3 / scale             # grid point spacing in the y direction [m]
dz = 0.1e-3 / scale             # grid point spacing in the z direction [m]
kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dy, dz])

# define the properties of the propagation medium
medium = kWaveMedium(sound_speed=1500)

# create initial pressure distribution using makeBall
ball_magnitude = 10         # [Pa]
ball_radius = 3 * scale     # [grid points]
p0 = ball_magnitude * makeBall(Nx, Ny, Nz, Nx/2, Ny/2, Nz/2, ball_radius)
filepath = "./data/input/1.mat"
data = scio.loadmat(filepath)
skin_vessel = data['skin'][:,:,22:34]-2

# smooth the initial pressure distribution and restore the magnitude
source = kSource()
source.p0 = skin_vessel

# define a binary planar sensor
sensor_mask = np.zeros((kgrid.Nx, kgrid.Ny, kgrid.Nz))
sensor_mask[:,0,np.int(Nz/2-1)] = 1
sensor = kSensor(sensor_mask)

# create the time array
kgrid.makeTime(medium.sound_speed)

# set the input settings
input_filename  = f'example_input.h5'
input_file_full_path = os.path.join(pathname, input_filename)
# set the input settings
input_args = {
    'PMLInside': False,
    'PMLSize': PML_size,
    'Smooth': False,
    'DataCast': 'single',
    'SaveToDisk': input_file_full_path,
    'SaveToDiskExit': False

}

# run the simulation
sensor_data = kspaceFirstOrder3DC(**{
    'medium': medium, 
    'kgrid': kgrid,
    'source': source,
    'sensor': sensor,
    **input_args
})

savemat('./data/output/d1.mat',{'sensor':sensor_data})

plt.figure()
plt.imshow(sensor_data)
plt.savefig('./data/output/d1_3D.png')