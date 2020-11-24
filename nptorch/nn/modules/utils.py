from itertools import repeat


def _make_tuple(n):
    def parse(x):
        if isinstance(x, (tuple, list)):
            return x
        return tuple(repeat(x, n))
    return parse


_single = _make_tuple(1)
_pair = _make_tuple(2)
_triple = _make_tuple(3)
