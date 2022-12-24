import warnings

import numpy as np

import uff
from uff import ChannelData
from uff.uff_io import Serializable
from uff.utils import is_keys_str_decimals, snake_to_camel_case, save_dict_to_hdf5, load_uff_dict


class UFF(Serializable):
    def __init__(self):
        self.channel_data: ChannelData = None
        self.event = None
        self.timed_event = None
        self.transmit_setup = None
        self.receive_setup = None
        self.transmit_wave = None
        self.excitation = None
        self.wave = None
        self.aperture = None
        self.probe = None
        self.element = None
        self.element_geometry = None
        self.impulse_response = None
        self.perimeter = None
        self.transform = None
        self.rotation = None
        self.translation = None
        self.version = None

    def __copy__(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    def __deepcopy__(self, memodict={}):
        pass

    @staticmethod
    def str_name():
        return "UFF"

    @classmethod
    def deserialize(cls: object, data: dict):
        obj = UFF()
        primitives = (np.ndarray, np.int64, np.float64, str, bytes, int, float)

        for k, v in data.items():
            if isinstance(v, primitives):
                setattr(obj, k, v)
                continue
            assert isinstance(v,
                              dict), f'{type(v)} did not pass type-assertion'

            if k != 'channel_data':
                warnings.warn(
                    f'\nWarning: The UFF standard specifies how objects of class uff.channel_data are '
                    f'written.\n Although other properties can be saved to the file, they very likely '
                    f'cannot be read back.\n In order to avoid unnecessary crashes, property `uff.{k}` will '
                    f'not be deserialized.')
                continue
            else:
                property_cls = ChannelData

            if not is_keys_str_decimals(v):
                setattr(obj, k, property_cls.deserialize(v))
            else:
                # TODO: assert keys are correct => ascending order starting from 000001
                list_of_objs = list(v.values())
                list_of_objs = [
                    property_cls.deserialize(item) for item in list_of_objs
                ]
                setattr(obj, k, list_of_objs)

        return obj

    @classmethod
    def load(cls: object, file_name: str):
        uff_dict = load_uff_dict(file_name)
        return cls.deserialize(uff_dict)

    @staticmethod
    def check_version(uff_h5):
        if uff_h5['version']:
            # read uff version
            file_version = uff_h5['version/major'], uff_h5[
                'version/minor'], uff_h5['version/patch']
            package_version = uff.__version_info__
            if file_version == package_version:
                raise ValueError(
                    f"The file version given ({'.'.join(str(i) for i in file_version)})does not have a matching "
                    f"version. Version must be {uff.__version__} "
                )
        else:
            raise Exception(
                "Not a valid uff file. UFF version field is missing")

    @property
    def summary(self):
        print(f'Summary:\n'
              f'\tSystem:\t{self.channel_data.system}\n'
              f'\tAuthors:\t{self.channel_data.authors}\n'
              f'\tCountry:\t{self.channel_data.country_code}\n'
              f'\tLocal Time:\t{self.channel_data.local_time}\n'
              f'\tDescription:\t{self.channel_data.description}\n')

    def _init_obj_from_param_list(self, class_name: str, params: dict):
        class_ = globals()[class_name]
        obj_params = list(class_.__annotations__)
        return class_(**self._instantiate_args(obj_params, params))

    def _instantiate_list(self, real_list, class_name):
        # List of type title
        for idx, l in enumerate(real_list):
            assert isinstance(l, dict), f"Unrecognized list entry {l}"
            real_list[idx] = self._init_obj_from_param_list(class_name, l)
        return real_list

    def _str_indexed_list2regular_list(self, parameter, object_dictionary):
        # list of values in fake dictionary if key equals index - 1 because keys start with one
        real_list = [
            value for ind, (key, value) in enumerate(object_dictionary.items())
            if int(key) - 1 == ind
        ]
        if snake_to_camel_case(parameter) in globals().keys():
            title = snake_to_camel_case(parameter)
            # TODO: Aperture doesn't define origin type.
        else:
            # TODO: when parameter == 'probes' => this is not conform to the standard

            param_title_map = {
                'probes': 'Probe',
                'sequence': 'TimedEvent',
                'unique_events': 'Event',
                'element_impulse_response': 'ImpulseResponse',
                'unique_excitations': 'Excitation',
                'unique_waves': 'Wave',
                'transmit_waves': 'TransmitWave',
            }
            assert parameter in param_title_map, f"What am I? {parameter}"

            title = param_title_map[parameter]

        return self._instantiate_list(real_list, title)

    def save(self, output_uff_path: str, version=None):
        if not output_uff_path.endswith('.uff'):
            output_uff_path += '.uff'

        serialized = self.serialize()
        if version is not None:
            serialized['version'] = version
        if 'channel_data' in serialized:
            serialized['uff.channel_data'] = serialized.pop('channel_data')
        save_dict_to_hdf5(serialized, output_uff_path)
