"""Typical error types are presented here.
The file is under development.
"""


class StatLabErrors(Exception):
    pass


class LengthCriteria(StatLabErrors):
    pass


class ShapeError(StatLabErrors):
    pass


class IncorrectIndex(StatLabErrors):
    pass


class IncorrectDimension(StatLabErrors):
    pass


class IncorrectValue(StatLabErrors):
    pass


class IncorrectType(StatLabErrors):
    pass
