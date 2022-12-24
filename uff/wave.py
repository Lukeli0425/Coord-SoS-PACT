from dataclasses import dataclass

from uff.aperture import Aperture
from uff.origin import Origin
from uff.uff_io import Serializable
from uff.wave_type import WaveType


@dataclass
class Wave(Serializable):
    """
    UFF class to describe the geometry of a transmitted wave or beam.

    Attributes:
        origin 	(WaveOrigin):       Geometric origin of the wave.
        type (WaveType):         	enumerated type (int)
                                    (converging = 0,
                                    diverging = 1,
                                    plane = 2,
                                    cylindrical = 3,
                                    photoacoustic = 4,
                                    default = 0)
        aperture (Aperture):     	Description of the aperture used to produce the wave
        excitation 	(int): 	        (Optional) index to the unique excitation in the parent group
    """

    @staticmethod
    def str_name():
        return 'unique_waves'

    @classmethod
    def deserialize(cls: object, data: dict):
        data['wave_type'] = data.pop('type')
        return super().deserialize(data)

    def serialize(self):
        data = super().serialize()
        data['type'] = data.pop('wave_type')
        return data

    origin: Origin
    wave_type: WaveType
    aperture: Aperture
    excitation: int = None
