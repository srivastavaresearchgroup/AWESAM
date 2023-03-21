"""
Event detection algorithm for seismic data with adaptive max-filter.

Core functions: find_peaks, find_amplitudes
"""

import numpy as np
import scipy.signal
import scipy.interpolate
import torch
import typing

from awesam import awesamlib


def find_peaks(
    x: np.ndarray,
    kernel_factor: float,
    downsampling: int,
    threshold_factor: float,
    threshold_window_size: int,
    min_max_kernel_size: tuple,
    return_extras: bool = False,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Core peak detection algorithm.
    Given the 1d-waveform data (`x`), the function returns a numpy array containing the time (and amplitude if return_heights is set to true) of events:
    - Time in samples
    - Amplitude in counts

    Hyperparameters:
    - threshold_factor: prominence threshold factor for peak detection.
    - downsampling: Downsampling in maxfilter-step. Ideally it would be 1 (no downsampling), which is very computationally intensive.
    - kernel_factor: factor for the adaptive kernels used in adaptive MaxFilter
    - threshold_window_size: the computation of the prominence threshold is based on a moving time window. Window size in both directions.

    Parameters:
    - x: 1d-numpy array containing seismic data.
    - return_heights: if set to true, also the amplitude of events is returned (as 2d numpy array)
    - return_extras: if debug information should be returned
    """

    # double precision to allow squaring without overflowing
    x = x.astype(np.double)
    # get kernel sizes used for maxfilter
    kernels = maxfilter_kernels(
        x, kernel_factor, downsampling, threshold_window_size, min_max_kernel_size
    )
    # apply maxfilter to input data
    maxfilter_x = awesamlib.adaptive_maxfilter(x**2, kernels, downsampling)
    # get thresholds for prominence event detection
    thresholds = _get_thresholds(
        x,
        maxfilter_x,
        len(x) // downsampling,
        downsampling,
        threshold_window_size,
        threshold_factor,
    )
    # apply prominence event detection
    peaks, properties = _get_peaks(maxfilter_x, thresholds)
    widths = _get_widths(maxfilter_x, peaks, properties)

    if not return_extras:
        return peaks, widths
    else:
        # unsquare data
        maxfilter_x = maxfilter_x ** (0.5)
        thresholds = thresholds ** (0.5)

        # compute amplitudes
        amplitudes = find_amplitudes(
            stream=None, event_times=peaks, maxfilter=maxfilter_x
        )

        return peaks, amplitudes, widths, kernels, maxfilter_x, thresholds


def find_amplitudes(
    stream: np.ndarray,
    event_times: np.ndarray,
    downsampling: int = None,
    maxfilter_kernel: int = None,
    maxfilter: np.ndarray = None,
) -> np.ndarray:
    """
    Returns amplitude of given events in event_times.
    """
    if maxfilter is None:
        # applies a maxfilter with constant kernel
        maxfilter = awesamlib.adaptive_maxfilter(
            stream.astype(float),
            np.ones(len(stream) // downsampling) * maxfilter_kernel,
            downsampling,
        )

    heights = np.zeros_like(event_times)  # initialize output array
    for i, peak in enumerate(event_times):
        heights[i] = maxfilter[peak]  # extract height information

    return heights


def maxfilter_kernels(
    x: np.ndarray,
    kernel_factor: int,
    downsampling: int,
    window_size: int,
    min_max_kernel_size: tuple,
) -> np.ndarray:
    """
    Returns an array containing the kernel sizes for adaptive maxfiltering. It is based on a simple average convolution.
    The output shape is len(x) // downsampling.
    """
    conv = torch.nn.Conv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=window_size // 2,
        stride=downsampling,
        padding=window_size // 4 - 1,
        padding_mode="replicate",
    ).requires_grad_(False)
    conv.weight[0][0] = torch.tensor(
        scipy.signal.windows.gaussian(window_size // 2, window_size / 10)
    )
    kernels = (
        conv(torch.absolute(torch.tensor(x).float().view(1, 1, -1))).flatten().numpy()
    )

    # maximum and minimum kernel sizes
    kernels = np.maximum(kernels, min_max_kernel_size[0])
    kernels = np.minimum(np.sqrt(kernels) * kernel_factor, min_max_kernel_size[1])
    return kernels


def _threshold_loop(
    x: np.ndarray,
    maxfilter: np.ndarray,
    downsampling: np.ndarray,
    threshold_window_size: int,
    threshold_factor: float,
) -> np.ndarray:
    """
    Computes prominence thresholds. See get_thresholds for more information.
    """

    def compute_threshold(window_x, window_maxfilter):
        mean = np.absolute(window_x).mean()
        std = np.absolute(window_x).std()
        return (
            (threshold_factor * (mean / std) ** 2 * window_maxfilter.mean())
            if std != 0
            else 0.0
        )

    window_size = threshold_window_size // downsampling
    total_samples = len(x) // downsampling
    i = window_size
    thresholds = []

    while i + window_size <= total_samples:
        threshold = compute_threshold(
            x[(i - window_size) * downsampling : (i + window_size) * downsampling],
            maxfilter[(i - window_size) : (i + window_size)],
        )
        thresholds.append((i, threshold))
        i += window_size

    if len(thresholds) == 0:  # if total length of x is too short
        thresholds = [(0, compute_threshold(x, maxfilter))]
    if len(thresholds) == 1:  # if too short, duplicate entry
        thresholds += [(len(maxfilter) - 1, thresholds[0][1])]

    return np.array(thresholds)


def _get_thresholds(
    x: np.ndarray,
    maxfilter: np.ndarray,
    target_size: int,
    downsampling: int,
    threshold_window_size: int,
    threshold_factor: float,
) -> np.ndarray:
    """
    Computes prominence thresholds from the MaxFilter data and interpolates it to have the same shape as the maxfilter data.
    The target size is the shape of the returned value.
    """
    thresholds = _threshold_loop(
        x, maxfilter, downsampling, threshold_window_size, threshold_factor
    )

    # interpolate between threshold values
    f = scipy.interpolate.interp1d(
        thresholds[:, 0],
        thresholds[:, 1],
        kind=1,
        fill_value=(thresholds[0][1], thresholds[-1][1]),
        bounds_error=False,
    )
    thresholds_interpolated = f(np.arange(target_size))

    return thresholds_interpolated


def _get_peaks(
    maxfilter_x: np.ndarray, thresholds: np.ndarray
) -> typing.Tuple[np.ndarray, dict]:
    """
    Returns two arrays with time and properties of all peaks in the maxfilter data.

    Parameter:
    - maxfilter_x: Maxfilter data, 1d-numpy array.
    - thresholds: Thresholds for peak detection. Must have same shape as maxfilter data.
    """
    return scipy.signal.find_peaks(maxfilter_x, prominence=thresholds)


def _get_widths(
    maxfilter_x: np.ndarray, peaks: np.ndarray, properties: dict
) -> np.ndarray:
    widths, _, _, _ = scipy.signal.peak_widths(
        maxfilter_x, peaks, rel_height=0.5, prominence_data=list(properties.values())
    )
    return widths


def adaptive_maxfilter(
    x: np.ndarray,
    kernel_function,
    downsampling: int,
    threshold_window_size: int,
    minmax: tuple,
    return_kernels=False,
) -> np.ndarray:
    """Applies the adaptive maxfilter algorithm"""
    kernels = maxfilter_kernels(x, 1, downsampling, threshold_window_size, minmax)
    kernels = kernel_function(kernels)
    maxfilter = awesamlib.adaptive_maxfilter(x, kernels, downsampling)
    if return_kernels is False:
        return maxfilter
    else:
        return maxfilter, kernels
