from dataclasses import dataclass

from uff.rotation import Rotation
from uff.position import Position
from uff.translation import Translation
from uff.uff_io import Serializable


@dataclass
class Transform(Serializable):
    """Specifies a 3D affine transformation of rotation plus translation,
    IN THAT ORDER, where the translation is done on the unrotated
    coordinate system. The direction is given by local coordinate
    system of the object that owns this object.

    Attributes:
        translation (Translation): 	change in position in meters
        rotation (Rotation): 	    change in rotation in radians
    """
    translation: Translation
    rotation: Rotation

    @staticmethod
    def str_name():
        return 'transform'

    def __call__(self, point):
        if type(point) is Position:
            return self.translation(self.rotation(point))
        else:
            raise TypeError(f"Type {type(point)} not recognized.")
        pass
