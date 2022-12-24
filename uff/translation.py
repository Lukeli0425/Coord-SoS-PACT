from dataclasses import dataclass

from uff.uff_io import Serializable
from uff.position import Position


@dataclass
class Translation(Serializable):
    """Define a translation operation in a 3D Cartesian system"""
    x: float
    y: float
    z: float

    @staticmethod
    def str_name():
        return 'translation'

    def __call__(self, pos):
        return Position(x=pos.x + self.x, y=pos.y + self.y, z=pos.z + self.z)
