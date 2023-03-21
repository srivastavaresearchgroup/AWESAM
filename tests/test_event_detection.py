import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import obspy

from .. import EventDetection, config, util
from . import test_util


def test_event_det(time, duration, filter=True):
    """
    Create test
    """
    downsampling = config.settings["EventDetection"]["downsampling"]

    stream = test_util.get_waveforms(
        time,
        duration,
        channel=util.get_channels(config.settings["general"]["stations"][0]["station"])[
            0
        ],
        station=util.get_station_seeds(config.settings["general"]["stations"])[0],
    )

    if stream is None:
        raise ValueError("No data available. Try to change the time.")
    trace = stream.merge()[0]

    if isinstance(trace.data, np.ma.masked_array):
        trace.data = np.ma.filled(trace.data, fill_value=0)

    if filter and config.settings["EventCatalog"]["event_detection_filter"] is not None:
        trace.filter(**config.settings["EventCatalog"]["event_detection_filter"])

    event_detection_settings = {
        "kernel_factor": config.settings["EventDetection"]["kernel_factor"],
        "downsampling": config.settings["EventDetection"]["downsampling"],
        "threshold_factor": config.settings["EventDetection"]["threshold_factor"],
        "threshold_window_size": config.settings["EventDetection"][
            "threshold_window_size"
        ],
        "min_max_kernel_size": config.settings["EventDetection"]["min_max_kernel_size"],
    }

    p, a, _, k, m, t = EventDetection.find_peaks(
        trace.data, **event_detection_settings, return_extras=True
    )

    return _plot_figure(
        trace, p, a, downsampling, m, k, t, figsize=test_util.FIGSIZE, dpi=test_util.DPI
    )


def _plot_figure(
    trace,
    peaks,
    amplitudes,
    downsampling,
    maxfilter,
    kernels,
    thresholds,
    figsize,
    dpi=250,
):
    fig, ax = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=figsize,
        dpi=dpi,
        gridspec_kw=dict(height_ratios=[3, 1]),
    )
    fig.subplots_adjust(hspace=0)

    sampling_rate = trace.stats.sampling_rate
    starttime = trace.stats.starttime.datetime
    ax[0].plot(trace.data, c="C2", alpha=0.6)
    ax[0].plot(abs(trace.data), c="C2", label="original data")
    ax[0].plot(
        peaks * downsampling,
        amplitudes,
        "x",
        c="C0",
        markersize=10,
        markeredgewidth=2,
        label="detected events",
    )
    ax[0].plot(
        np.arange(0, len(maxfilter)) * downsampling,
        maxfilter,
        c="C1",
        lw=1,
        label="adaptive MaxFilter",
    )
    ax[0].set_ylim(-trace.data.max() / 4, None)

    ax[1].plot(
        np.arange(len(kernels)) * downsampling,
        kernels / 100,
        c="C3",
        label="filter-length",
    )
    ax[0].plot(
        np.arange(len(thresholds)) * downsampling,
        thresholds,
        c="teal",
        label="threshold",
    )

    ax[0].legend(loc="upper right"), ax[1].legend(loc="upper right")
    ax[0].set_title(starttime)

    def formatter(x, pos):
        return int(x / sampling_rate / 60)

    f = matplotlib.ticker.FuncFormatter(formatter)
    ax[1].xaxis.set_major_formatter(f)

    ax[1].set_xlabel("time (in minutes)")
    # ax[2].set_ylabel('threshold\n(in counts)\n', rotation=90)#, ha='right')
    ax[1].set_ylabel("filter-length\n(in seconds)\n", rotation=90)  # , ha='right')
    ax[0].set_ylabel("amplitude\n(in counts in 1.59 nm/s)")

    return fig
