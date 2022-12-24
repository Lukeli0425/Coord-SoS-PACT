from dataclasses import dataclass

from uff.origin import Origin
from uff.position import Position


@dataclass
class SphericalWaveOrigin(Origin):
    position: Position = Position()
