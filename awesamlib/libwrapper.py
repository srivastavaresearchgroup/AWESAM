import os
import ctypes
import numpy as np
import numpy.ctypeslib as npct

directory = os.path.dirname(os.path.abspath(__file__))

lib = npct.load_library("awesamlib.so", directory)

array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags="CONTIGUOUS")
array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags="CONTIGUOUS")

lib.adaptive_maxfilter.argtypes = [
    array_1d_double,
    array_1d_int,
    array_1d_double,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.adaptive_maxfilter.restypes = None


def adaptive_maxfilter(x: np.ndarray, kernels: np.ndarray, downsampling: int):
    kernels = kernels.astype(np.int32)
    output = np.empty(len(x) // downsampling, dtype=np.double)

    lib.adaptive_maxfilter(x, kernels, output, len(x), len(kernels), downsampling)
    return output


lib.compute_probabilities.argtypes = [
    array_1d_double,
    array_1d_double,
    array_1d_double,
    array_1d_double,
    array_1d_double,
    array_1d_double,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_double,
]
lib.compute_probabilities.restypes = None


def compute_probabilities(
    principal_catalog: np.ndarray,
    complementary_catalog: np.ndarray,
    metric: np.ndarray,
    window_length: float,
) -> np.ndarray:
    """
    Computes a probability for each event in the principal catalog, that its source is the volcano,
    by finding the minimum distance between the events. Distance is defined by the metric.
    """

    out = np.empty(len(principal_catalog), dtype=np.double)

    lib.compute_probabilities(
        principal_catalog[:, 0].astype(np.double),
        principal_catalog[:, 1].astype(np.double),
        complementary_catalog[:, 0].astype(np.double),
        complementary_catalog[:, 1].astype(np.double),
        out,
        metric.astype(np.double),
        len(principal_catalog),
        len(complementary_catalog),
        window_length,
    )
    return out
