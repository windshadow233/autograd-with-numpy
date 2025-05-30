import numpy as np


def set_ndim_eq(a: np.ndarray, b: np.ndarray):
    if a.ndim == b.ndim:
        return a, b
    diff_dims = tuple(range(abs(a.ndim - b.ndim)))
    if a.ndim < b.ndim:
        a = np.expand_dims(a, diff_dims)
    else:
        b = np.expand_dims(b, diff_dims)
    return a, b


def get_tile_dims(a: np.ndarray, b: np.ndarray):
    try:
        a + b
    except ValueError as e:
        raise e
    a, b = set_ndim_eq(a, b)
    if a.shape == b.shape:
        return (), ()
    shape_a = np.array(a.shape)
    shape_b = np.array(b.shape)
    tile_a = shape_b // shape_a
    tile_b = shape_a // shape_b
    tile_a += (tile_a == 0)
    tile_b += (tile_b == 0)
    tile_a = np.where(tile_a != 1)[0]
    tile_b = np.where(tile_b != 1)[0]
    return tuple(tile_a), tuple(tile_b)
