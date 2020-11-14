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
    a, b = set_ndim_eq(a, b)
    if a.shape == b.shape:
        return (), ()
    try:
        a1, b1 = np.broadcast_arrays(a, b)
    except ValueError as e:
        raise e
    shape_a = np.array(a.shape)
    shape_b = np.array(b.shape)
    shape_a1 = np.array(a1.shape)
    shape_b1 = np.array(b1.shape)
    tile_a = np.where(shape_a != shape_a1)[0]
    tile_b = np.where(shape_b != shape_b1)[0]
    return tuple(tile_a), tuple(tile_b)
