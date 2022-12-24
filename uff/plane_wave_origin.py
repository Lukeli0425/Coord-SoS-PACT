from dataclasses import dataclass

from uff.origin import Origin
from uff.rotation import Rotation


@dataclass
class PlaneWaveOrigin(Origin):
    rotation: Rotation = Rotation()
