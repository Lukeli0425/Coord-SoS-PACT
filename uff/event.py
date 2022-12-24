from dataclasses import dataclass

from uff.receive_setup import ReceiveSetup
from uff.transmit_setup import TransmitSetup
from uff.uff_io import Serializable


@dataclass
class Event(Serializable):
    """
    UFF class to describe an unique ultrasound event, composed by a single transmit and receive setup

    Attributes:
    transmit_setup (TransmitSetup):     Description of the transmit event (probe/channels, waves, excitations, etc.).
                                        If more than one probe is used in reception, this is a list of setups.
    receive_setup (ReceiveSetup):       Description of the transmit event (probe/channels, waves, excitations, etc.).
                                        If more than one probe is used in reception, this is a list of setups.

    """
    transmit_setup: TransmitSetup
    receive_setup: ReceiveSetup

    @staticmethod
    def str_name():
        return 'unique_events'
