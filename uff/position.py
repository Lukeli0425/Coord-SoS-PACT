from dataclasses import dataclass, field

from uff.uff_io import Serializable


@dataclass
class Position(Serializable):
    """Contains a location in space in Cartesian coordinates and SI units."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    _index: int = field(repr=False, init=False, default=0)

    def __add__(self, p2):
        self.x += p2.x
        self.y += p2.y
        self.z += p2.z
        return self

    @staticmethod
    def str_name():
        return 'position'

    def __len__(self):
        return 3

    def __add__(self, p2):
        if type(p2) == Position:
            return Position(x=p2.x + self.x, y=p2.y + self.y, z=p2.z + self.z)
        else:
            return Position(x=p2 + self.x, y=p2 + self.y, z=p2 + self.z)

    def __sub__(self, p2):
        if type(p2) is Position:
            return Position(x=p2.x - self.x, y=p2.y - self.y, z=p2.z - self.z)
        else:
            return Position(x=self.x - p2, y=self.y - p2, z=self.z - p2)

    def __truediv__(self, p2):
        if type(p2) in [Position, Position]:
            return Position(x=self.x / p2.x, y=self.y / p2.y, z=self.z / p2.z)
        else:
            return Position(x=self.x / p2, y=self.y / p2, z=self.z / p2)

    def __mul__(self, p2):
        if type(p2) in [Position, Position]:
            return Position(x=p2.x * self.x, y=p2.y * self.y, z=p2.z * self.z)
        else:
            return Position(x=p2 * self.x, y=p2 * self.y, z=p2 * self.z)

    def __floordiv__(self, p2):
        if type(p2) in [Position, Position]:
            return Position(x=self.x // p2.x, y=self.y // p2.y, z=self.z // p2.z)
        else:
            return Position(x=self.x // p2, y=self.y // p2, z=self.z // p2)

    def __pow__(self, other):
        return Position(x=self.x ** other, y=self.y ** other, z=self.z ** other)

    def __iter__(self):
        return PositionIterator(self)

    def __getitem__(self, item):
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        elif item == 2:
            return self.z
        else:
            raise IndexError(f"Index {item} out of bounds for type {__name__}")


class PositionIterator:

    def __init__(self, position):
        self._position = position
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self._position):
            if self._index == 0:
                result = self._position.x
            elif self._index == 1:
                result = self._position.y
            elif self._index == 2:
                result = self._position.z
            self._index += 1
            return result
        else:
            raise StopIteration
