import matplotlib
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

from .. import awesamlib, EventDetection, config, util
from . import test_util


def catalog_consolidation_plot(
    date: dt.datetime, duration: dt.timedelta = dt.timedelta(hours=1)
) -> plt.Figure:
    # create catalogs for two stations for specified period
    (pc, pc_trace), (cc, cc_trace) = get_catalogs(date, duration)

    # consolidate catalogs
    downsampling = config.settings["EventDetection"]["downsampling"]
    metric = config.settings["CatalogConsolidation"]["default_metric"]
    window_length = config.settings["CatalogConsolidation"]["window_length"]
    p = awesamlib.compute_probabilities(pc, cc, metric, window_length)

    fig, (axes) = plt.subplots(
        3,
        2,
        figsize=test_util.FIGSIZE,
        dpi=test_util.DPI,
        gridspec_kw={"width_ratios": (25, 1), "height_ratios": (3, 1, 1)},
    )
    fig.subplots_adjust(wspace=0.02, hspace=0)
    ax, ax_cbar = axes[0, 0], axes[0, 1]
    ax_pc, ax_cc = axes[1, 0], axes[2, 0]
    axes[1, 1].set_visible(False)
    axes[2, 1].set_visible(False)
    cmap = matplotlib.cm.get_cmap("inferno", 256)
    cmap.colors[:, 1] = cmap.colors[:, 1] * 0.8

    ax.scatter(
        pc[:, 0],
        pc[:, 1],
        c=p,
        marker="x",
        s=200,
        linewidth=4,
        cmap=cmap,
        label="principal catalog",
    )
    ax.scatter(cc[:, 0], cc[:, 1], c="teal", label="complementary catalog", s=40)

    colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=ax_cbar)
    colorbar.ax.set_ylabel("probability")

    def f(x, pos):
        return int(x / 60)

    formatter = matplotlib.ticker.FuncFormatter(f)

    ax_cc.xaxis.set_major_formatter(formatter)
    ax_cc.set_xticks(np.arange(0, pc[-1, 0] + 500, 60 * 20))
    ax_cc.set_xticks(np.arange(0, pc[-1, 0] + 500, 60 * 5), minor=True)

    ax_cc.set_xlabel("time (in minutes)")
    ax.set_ylabel("amplitude in counts (in 1.59 nm/s)")

    x_array = np.arange(len(cc_trace.data)) / downsampling

    ax.plot(
        x_array,
        abs(cc_trace.data),
        alpha=0.4,
        c="teal",
        label="ISTR (complementary catalog)",
    )
    ax.plot(
        x_array, abs(pc_trace.data), alpha=0.4, c="C2", label="IST3 (principal catalog)"
    )

    ax_pc.plot(
        x_array[::5], pc_trace.data[::5], c="C2", label="IST3 (principal catalog)"
    )
    ax_pc.plot(x_array, abs(pc_trace.data), c="C2")
    ax_cc.plot(
        x_array[::5], cc_trace.data[::5], c="teal", label="ISTR (complementary catalog)"
    )
    ax_cc.plot(x_array, abs(cc_trace.data), c="teal")

    ax_pc.plot(pc[:, 0], pc[:, 1], "x")
    ax_cc.plot(cc[:, 0], cc[:, 1], "x")

    ax_pc.legend(loc="upper right")
    ax_cc.legend(loc="upper right")

    ax.set_ylim(0, None)

    ax_pc.set_ylabel("amplitude")
    ax_cc.set_ylabel("amplitude")

    leg = ax.legend(loc="upper right")
    plt.setp(leg.get_title(), multialignment="left")

    return fig


def get_catalogs(
    date: dt.datetime, duration: dt.timedelta, filter: bool = True
) -> tuple:
    pc_stream = test_util.get_waveforms(
        date,
        duration=duration,
        station=util.get_station_seeds(config.settings["general"]["stations"])[0],
    )
    cc_stream = test_util.get_waveforms(
        date,
        duration=duration,
        station=util.get_station_seeds(config.settings["general"]["stations"])[1],
    )

    if pc_stream is None or cc_stream is None:
        raise ValueError("Data is not available for both stations, try different time.")

    pc_trace = config.settings["EventCatalog"]["preparation_function"](pc_stream)[0]
    cc_trace = config.settings["EventCatalog"]["preparation_function"](cc_stream)[0]

    if filter == True:
        pc_trace.filter(**config.settings["EventCatalog"]["event_detection_filter"])
        cc_trace.filter(**config.settings["EventCatalog"]["event_detection_filter"])

    event_detection_settings = {
        "kernel_factor": config.settings["EventDetection"]["kernel_factor"],
        "downsampling": config.settings["EventDetection"]["downsampling"],
        "threshold_factor": config.settings["EventDetection"]["threshold_factor"],
        "threshold_window_size": config.settings["EventDetection"][
            "threshold_window_size"
        ],
        "min_max_kernel_size": config.settings["EventDetection"]["min_max_kernel_size"],
    }

    pc_times = EventDetection.find_peaks(pc_trace.data, **event_detection_settings)[0]
    cc_times = EventDetection.find_peaks(cc_trace.data, **event_detection_settings)[0]

    pc_ampl = EventDetection.find_amplitudes(
        pc_trace.data,
        pc_times,
        config.settings["EventDetection"]["downsampling"],
        config.settings["EventDetection"]["maxfilter_kernel"],
    )
    cc_ampl = EventDetection.find_amplitudes(
        cc_trace.data,
        cc_times,
        config.settings["EventDetection"]["downsampling"],
        config.settings["EventDetection"]["maxfilter_kernel"],
    )

    pc = np.stack([pc_times, pc_ampl]).T
    cc = np.stack([cc_times, cc_ampl]).T

    return (pc, pc_trace), (cc, cc_trace)
