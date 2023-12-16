from enum import Enum, IntEnum
from typing import Callable


# number of arguments
class Arguments(IntEnum):
    UNARY = 1,
    ADD = 2,
    SUBTRACT = 2,
    MULTIPLY = 2,
    DIVIDE = 2,
    POWER = 2


class Func:

    def __init__(self, exec: Callable, n_args: int):
        self._executable = exec
        self.args_num = n_args

    @property
    def get_args_num(self):
        return self.args_num

    def evaluate(self, args):
        if len(args) != self.args_num:
            return None
        return self._executable(args)


# operators or funcs(+,-,/,*,...)
class Funcs:
    UNARY_SUBTRACTION = Func(lambda operand: -operand[0], n_args=Arguments.UNARY)
    ADD = Func(lambda operands: operands[0] + operands[1], n_args=Arguments.ADD)
    POWER = Func(lambda operand: operand[0] ** operand[1], n_args=Arguments.POWER)
    SUBTRACT = Func(lambda operands: operands[0] - operands[1], n_args=Arguments.SUBTRACT)
    MULTIPLY = Func(lambda operands: operands[0] * operands[1], n_args=Arguments.MULTIPLY)
    DIVIDE = Func(lambda operands: operands[0] / operands[1], n_args=Arguments.DIVIDE)


# tree linkers (+,max,min)
class TreeLinker(Enum):
    SUM = 0,
    MAX = 1,
    MIN = 2
