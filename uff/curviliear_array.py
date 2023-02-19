from dataclasses import dataclass

from uff.probe import Probe


@dataclass
class CurvilinearArray(Probe):
    """
    Describes a linear array, made of identical elements, uniformly distributed on a line.

    Attributes:
        number_elements (int): 	Number of elements in the array
        pitch (float): 	 	    Distance between the acoustic ceneter of adyacent elements [m]
        radius (float):
        element_width (float):	(Optional) Element size in the x-axis [m]
        element_height (float):	(Optional) Element size in the y-axis [m]
    """

    def str_name(self):
        return 'probe.curvilinear_array'

    number_elements: int
    pitch: float
    radius: float
    element_width: float = None
    element_height: float = None
