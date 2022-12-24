from __future__ import annotations

import abc
import typing
from typing import List, Type

import numpy as np

from uff.utils import PRIMITIVES, is_keys_str_decimals


class Serializable(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def str_name():
        return

    def serialize(self):
        serialized = {}
        # TODO: fix import cycle
        from uff import WaveType

        for k, value in vars(self).items():
            if value is None:
                continue

            if isinstance(value, PRIMITIVES):
                serialized[k] = value
                continue
            elif isinstance(value, list):
                keys = [f'{i:08d}' for i in range(1, len(value) + 1)]
                values = []
                for val in value:
                    if isinstance(val, PRIMITIVES):
                        values.append(val)
                    elif isinstance(val, Serializable):
                        values.append(val.serialize())
                    elif isinstance(val, list):
                        keys2 = [f'{i:08d}' for i in range(1, len(val) + 1)]
                        values.append(dict(zip(keys2, val)))
                    else:
                        raise NotImplementedError

                serialized[k] = dict(zip(keys, values))
            elif isinstance( value, Serializable):  # cls in Serializable.__subclasses__():
                serialized[k] = value.serialize()
            elif isinstance(value, WaveType):
                serialized[k] = value.value
            else:
                raise TypeError(f'Unknown type [{type(value)}] for serialization!')
        return serialized

    @classmethod
    def deserialize(cls: object, data: dict):
        fields = cls.__annotations__

        for k, v in data.items():
            assert k in fields, f'Class {cls} does not have property named {k}.'
            if isinstance(v, PRIMITIVES):
                continue
            assert isinstance(v, dict), f'{type(v)} did not pass type-assertion'

            # property_cls = Serializable.get_subcls_with_name(k)
            property_cls = fields[k]
            if isinstance(property_cls, typing._GenericAlias):  # TODO explain this
                property_cls = property_cls.__args__[0]
            assert property_cls is not None, f'Class {k} is not Serializable!'

            if not is_keys_str_decimals(v):
                data[k] = property_cls.deserialize(v)
            else:
                # TODO: assert keys are correct => ascending order starting from 000001
                list_of_objs = list(v.values())
                list_of_objs = [property_cls.deserialize(item) for item in list_of_objs]
                data[k] = list_of_objs

        return cls(**data)

    @staticmethod
    def all_subclasses(cls=None) -> List[Type[Serializable]]:
        if cls is None:
            cls = Serializable
        subclasses = set(cls.__subclasses__()).union([
            s for c in cls.__subclasses__()
            for s in Serializable.all_subclasses(c)
        ])
        return list(subclasses)

    @staticmethod
    def get_subcls_with_name(name) -> Type[Serializable]:
        all_subclasses = Serializable.all_subclasses()
        for subcls in all_subclasses:
            if subcls.str_name() == name:
                return subcls
        return None

    def assign_primitives(self, dictionary: dict):
        primitives = (np.ndarray, np.int64, np.float64, str, bytes)
        set_list = []
        obj_attrs = list(self.__annotations__)
        for k, v in dictionary.items():
            assert k in obj_attrs
            if isinstance(v, primitives):
                setattr(self, k, primitives)
                set_list.append(k)

        remaining_list = list(set(obj_attrs) - set(set_list))
        return set_list, remaining_list

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        equal = True
        last_attr = None

        for k in self.__annotations__.keys():
            if not equal:
                break
            last_attr = k

            my_attr = getattr(self, k)
            other_attr = getattr(other, k)
            if isinstance(my_attr, np.ndarray):
                if not isinstance(other_attr, np.ndarray):
                    equal = False
                elif not np.allclose(my_attr, other_attr):
                    equal = False
            else:
                try:
                    if getattr(self, k) != getattr(other, k):
                        equal = False
                except ValueError as err:
                    print(
                        'Ooops! Something went wrong! Probably unsupported comparision. '
                        'Try to override the __eq__ method & handle custom data structures properly. '
                        'Error message is given below:')
                    raise err

        if not equal:
            print('Non-matching attributes detected!')
            print(f'Class name: {self.__class__}')
            print(f'Attrb name: {last_attr}')

        return equal
