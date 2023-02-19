from dataclasses import dataclass
from typing import List

from uff.element import Element
from uff.element_geometry import ElementGeometry
from uff.impulse_response import ImpulseResponse
from uff.transform import Transform
from uff.uff_io import Serializable


@dataclass
class Probe(Serializable):
    """
    Describes an generic ultrsound probe formed by a collection of elements.

    Note:

    Where focal_length specifies the lens focusing distance. Note that the
    elements in element_geometry and impulse_response are referred by the
    fields inside each member in element, avoiding unnecessary replication
    of information.

    More compact, although less general, descriptions are available for:

        uff.probe.linear_array,
        uff.probe.curvilinear_array, and
        uff.probe.matrix_array.
    """

    @staticmethod
    def str_name():
        return 'probes'

    def serialize(self):
        assert self.element_geometry is None or isinstance(self.element_geometry, list), \
            'Probe.element_geometry should be a list of element geometries!'
        return super().serialize()

    # @classmethod
    # def deserialize(cls, data: dict):
    #     pass

    # >> TODO: These parameters are not defined in the standard
    number_elements: int
    pitch: float
    element_height: float
    element_width: float
    ##  <<
    transform: Transform = None
    # TODO for conformity call `elements`
    element: List[Element] = None
      
    element_impulse_response: List[ImpulseResponse] = None
    focal_length: float = None
    element_geometry: List[ElementGeometry] = None
    # # >> TODO: These parameters are not defined in the standard
    number_elements: int = 0
    pitch: float = 0
    element_height: float = 0
    element_width: float = 0
    ##  <<
    ## TODO: add element list factory to generate element list from these parameters,
    # and getters to generate properties from these properties.
    # OPEN QUESTION: what to do when file contains extra properties that do not conform to standard?
    # TODO idea for getters
    # @property
    # def number_elements(self):
    #     return len(self.element)
    # @property
    # def pitch(self):
    #     # check if all element spacings are the same
    #     # return pitch
    #     # else:
    #     # catch and warn of non-standard spacing
    #     return len(self.element)
    #
    # @property
    # def element_width(self):
    #     # check for conformal element width
    #     # return element_width
    #     # else:
    #     # catch and warn of non-conformal element width
    #     pass
    #
    # @property
    # def element_height(self):
    #     # check for conformal element height
    #     # return element_height
    #     # else:
    #     # catch and warn of non-conformal element height
    #     pass
