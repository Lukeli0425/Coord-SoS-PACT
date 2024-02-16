import os
import sys
# Set environment variable to binaries to get rid of user warning
# This code is a crutch and should be removed when kspaceFirstOrder
# is refactored
from os import environ

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ktransducer import NotATransducer
from kwave.options import SimulationOptions

if sys.platform.startswith('darwin'):
    system = "macOS"
elif sys.platform.startswith('linux'):
    system = "linux"
elif "win" in sys.platform:
    system = "windows"

environ['KWAVE_BINARY_PATH'] = os.path.join(os.path.dirname(__file__), 'bin', system)
