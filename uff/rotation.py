from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import numpy as np

from uff.uff_io import Serializable
from uff.position import Position


@dataclass
class Rotation(Serializable):
    """Contains a rotation in space in spherical coordinates and SI units.
    The rotation is specified using Euler angles that are applied in the order ZYX."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @staticmethod
    def str_name():
        return 'rotation'

    def __call__(self, pos):
        return Position(*self.rotation_object.apply(np.array(pos)))

    @property
    def matrix(self):
        return R.from_euler('xyz', [self.x, self.y, self.z], degrees=False).matrix()

    @property
    def rotation_object(self):
        return R.from_euler('xyz', [self.x, self.y, self.z], degrees=False)
