from dataclasses import dataclass
from typing import List

from uff.uff_io import Serializable


@dataclass
class ReceiveSetup(Serializable):
    """
    Describes the setup used to receive and sample data. If more than one probe is used in reception, this is a list
    of setups.

    Notes: The channel_mapping specifies the connection of system channels to transducer elements. The map is a
    M-by-N matrix (HDF5 dimension ordering), where N is the number of system channels and M is the maximum number of
    elements connected to a single channel. In most common cases, M=1. An unconnected state is marked by the value 0.

    Attributes:
    probe (int):                     	Index of the uff.probe used in reception within the list of probes in the
                                        uff.channel_data structure
    time_offset (float):            	Time delay between the event start and the acquisition of the first sample [s]
    channel_mapping	(list(list(int)): 	Map of receive channels to transducer elements
    sampling_frequency 	(float):    	Sampling frequency of the recorded channel data [Hz]
    tgc_profile (list(float)):    	    (Optional) Analog TGC profile sampled at tgc_sampling_frequency [dB]
    tgc_sampling_frequency (float): 	(Optional) Sampling frequency of the TGC profile [Hz]
    modulation_frequency 	(float): 	(Optional) Modulation frequency used in case of IQ-data [Hz]

    """

    @staticmethod
    def str_name():
        return 'receive_setup'

    probe: int
    time_offset: float
    channel_mapping: List[List[int]]
    sampling_frequency: float
    tgc_profile: List[float] = None
    tgc_sampling_frequency: float = None
    modulation_frequency: float = None

    def __eq__(self, other):
        return super().__eq__(other)
