from enum import Enum


class WaveType(Enum):
    """
    enumerated type (int)
        converging = 0
        diverging = 1
        plane = 2
        cylindrical = 3
        photoacoustic = 4
        default = 0
    """

    CONVERGING = 0
    DIVERGING = 1
    PLANE = 2
    CYLINDRICAL = 3
    PHOTOACOUSTIC = 4
    DEFAULT = 0
