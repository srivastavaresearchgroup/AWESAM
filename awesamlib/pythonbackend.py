import numpy as np
import numba


@numba.jit(nopython=True)
def adaptive_maxfilter(
    x: np.ndarray, kernels: np.ndarray, downsampling: int
) -> np.ndarray:
    """
    computed the adaptive maxfiltering data from the seismic data and corresponding kernel sizes.
    The shape of the kernels should be
    >>> len(kernels) = len(x) // config.downsampling_factor

    Parameter:
    - x: 1d numpy array of seismic data
    - kernels: 1d numpy array with kernel sizes
    """
    output = np.zeros(len(x) // downsampling)  # initialize output array
    i = 0
    while i < len(kernels) - 1:
        # extract current window (with adaptive kernel size)
        # at the array bounds smaller kernel sizes are possible
        window = x[
            max(0, i * downsampling - int(kernels[i] / 2)) : i * downsampling
            + int(kernels[i] / 2)
        ]
        # get maximum value in this window
        output[i] = max(window)
        i += 1

    return output


def compute_probabilities(
    pc: np.ndarray, cc: np.ndarray, metric: np.ndarray, window_length: float
) -> np.ndarray:
    """
    Computes a probability for each event in the principal catalog, that its source is the volcano,
    by finding the minimum distance between the events. Distance is defined by the metric.
    """
    probabilities = np.zeros(len(pc))

    for i in range(len(pc)):

        pc_event = pc[i]
        cc_events = cc[
            (cc[:, 0] > (pc_event[0] - window_length))
            & (cc[:, 0] < (pc_event[0] + window_length))
        ]

        if len(cc_events) != 0:
            probabilities[i] = _find_minimum_distance(
                pc_event, cc_events, get_metric(metric)
            )

    return probabilities


def _find_minimum_distance(
    pc_event: np.ndarray, cc_events: np.ndarray, metric
) -> np.ndarray:
    """
    For one pc_event the probability is determined by (possibly multiple) cc_events, by first finding the minimal distance.
    """
    distances = np.zeros(len(cc_events))

    for i, cc_event in enumerate(cc_events):
        distances[i] = _metric_distance(pc_event, cc_event, metric)

    return np.exp(-distances.min())


def _metric_distance(pc_event: np.ndarray, cc_event: np.ndarray, metric) -> float:
    """
    Computed distance of two events using the metric
    """
    return metric(*(pc_event - cc_event), min(pc_event[1], cc_event[1]))


def get_metric(metric: np.ndarray):
    def compute_metric(dt: np.ndarray, dy: np.ndarray, amp: np.ndarray) -> float:
        return np.linalg.norm(
            [metric[0] * dt + metric[1] * dy, metric[2] * dt + metric[3] * dy]
        )

    return compute_metric
