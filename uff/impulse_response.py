from dataclasses import dataclass
from typing import List

from uff.uff_io import Serializable


@dataclass
class ImpulseResponse(Serializable):
    """Specifies a temporal impulse response"""
    initial_time: float
    sampling_frequency: int
    data: List[float]
    units: str

    @staticmethod
    def str_name():
        return 'element_impulse_response'
