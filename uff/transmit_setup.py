from dataclasses import dataclass
from typing import List

from uff.transmit_wave import TransmitWave
from uff.uff_io import Serializable


@dataclass
class TransmitSetup(Serializable):
    """
    UFF class to describe the transmit event (probe/channels, waves, excitations, etc.).

    Attributes:
        probe 	(int):                      	Index of the uff.probe used for transmit within the list of probes in
                                                the uff.channel_data structure
        transmit_waves (list(TransmitWave)): 	List of transmit waves used in this event with their respective time
                                                offset and weight
        channel_mapping (list(list(int))): 	    Map of transmit channels to transducer elements
        sampled_delays (float):	                (Optional) Transmit delay as set in the system for active channels [s].
        sampled_excitations	(float):        	(Optional) Matrix of sampled excitation waveforms [normalized units].
        sampling_frequency 	(float): 	        (Optional) Sampling frequency of the excitation waveforms [Hz]
        transmit_voltage	(float): 	        (Optional) Peak amplitude of the pulse generator [V]
    """

    @staticmethod
    def str_name():
        return 'transmit_setup'

    probe: int
    transmit_waves: List[TransmitWave]
    channel_mapping: List[List[int]]
    sampled_delays: float = None
    sampled_excitations: float = None
    sampling_frequency: float = None
    transmit_voltage: float = None
