from dataclasses import dataclass

from uff.probe import Probe


@dataclass
class LinearArray(Probe):
    """
    Describes a matrix array, made of identical elements, uniformly distributed on a 2D grid.

    Attributes:
        number_elements_x (int):Number of elements in the x-axis
        pitch_x (float):        Distance between the acoustic center of adjacent elements along the x-axis [m]
        number_elements_y (int):Number of elements in the y-axis
        pitch_y (float):    	Distance between the acoustic center of adjacent elements along the y-axis [m]
        element_width (float):  (Optional) Element size in the x-axis [m]
        element_height (float): (Optional) Element size in the y-axis [m]
    """

    number_elements_x: int
    number_elements_y: int
    pitch_x: float
    pitch_y: float
    element_width: float
    element_height: float
