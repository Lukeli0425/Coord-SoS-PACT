from dataclasses import dataclass

from uff.position import Position
from uff.uff_io import Serializable


@dataclass
class Aperture(Serializable):
    """
    UFF class to define analytically the aperture use in an ultrasound wave.

    Notes:
    The class aperture defines the transmit apodization profile. In this case
    origin defines the center of the aperture. The size of the aperture can
    be described with fixed_size, or with f_number in which case the aperture
    size is d/f_number where d is the distance between uff.wave.origin and
    uff.wave.aperture.origin. The parameter window is a string describing
    the apodization window.

    Attributes:
        origin (Position): 	    Location of the aperture center in space.
        window (str): 	        String defining the apodization window type and
                                parameter (e.g., 'Hamming', 'Gauss(8)', 'Tukey(0.5)')
        f_number list[float]:	Desired F-number of the aperture [Az, El]
        fixed_size list[float]: If non-zero, this overwrites the size of the aperture
                                in [m] [Az, El]
        minimun_size (float):   (Optional) If non-zero, this sets a limit for the minimum
                                dynamic aperture in m [Az, El]
        maximum_size (float): 	(Optional) If non-zero, this sets a limit for the maximum
                                dynamic aperture in m [Az, El]
    """

    @staticmethod
    def str_name():
        return 'aperture'

    # @classmethod
    # def deserialize(cls: object, data: dict):
    #     data['position'] = data.pop('origin')
    #     return super().deserialize(data)

    # TODO: standard has this named aperture but defined as position
    origin: Position
    # TODO: what should fixed size type be? list? float? how do you reproduce the same functionality
    fixed_size: float
    f_number: float = 1.0
    window: str = 'rectwin'
    minimum_size: float = None
    maximum_size: float = None
