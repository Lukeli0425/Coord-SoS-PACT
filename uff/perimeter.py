from dataclasses import dataclass
from typing import List

from uff.position import Position
from uff.uff_io import Serializable


@dataclass
class Perimeter(Serializable):
    """Describes the geometry of an ultrasonic element."""
    position: List[Position]

    @staticmethod
    def str_name():
        return 'perimeter'
