import argparse
import os
import sys
from tempfile import gettempdir

sys.path.append('../')

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.ktransducer import *
from kwave.utils import *
from utils.dataio import load_mat, save_mat
from utils.simulations import *


def kwaveForward2D(SoS):
    
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='K-wave numerical simulation.')
    parser.add_argument('--SoS', type=str, default='uniform', choices=['uniform', 'easy_phantom', 'hard_phantom'])
    opt = parser.parse_args()
    
    kwaveForward2D(SoS=opt.SoS)