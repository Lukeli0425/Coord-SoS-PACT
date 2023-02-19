from uff.position import Position


class TimeZeroReferencePoint(Position):
    """Contains a location in space in Cartesian coordinates and SI units for t0."""

    @staticmethod
    def str_name():
        return 'time_zero_reference_point'
