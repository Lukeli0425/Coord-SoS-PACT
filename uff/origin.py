from dataclasses import dataclass

from uff.position import Position
from uff.rotation import Rotation
from uff.uff_io import Serializable


@dataclass
# TODO: saved as origin but defined as wave origin in spec
# TODO: when loading origin, how to know what type of origin is being loaded?
class Origin(Serializable):
    @staticmethod
    def str_name():
        return 'origin'

    rotation: Rotation = Rotation()
    position: Position = Position()
