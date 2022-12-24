from dataclasses import dataclass

from uff.origin import Origin
from uff.position import Position


@dataclass
class CylindricalWaveOrigin(Origin):
    position: Position = Position()

    @staticmethod
    def str_name():
        return 'cylindrical_wave_origin'
