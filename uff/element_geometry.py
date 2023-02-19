from dataclasses import dataclass

from uff.perimeter import Perimeter
from uff.uff_io import Serializable


@dataclass
class ElementGeometry(Serializable):
    """Describes the geometry of an ultrasonic element.

    Notes: Here we assume that the acoustic center of the element is at origin O = (0; 0; 0) pointing towards Z = (0;
    0; 1). The element shape is defined by a closed perimeter contained within the XY -plane, that is in turn
    composed of an ordered set of uff.position instances.
    """
    perimeter: Perimeter

    @staticmethod
    def str_name():
        return 'element_geometry'
