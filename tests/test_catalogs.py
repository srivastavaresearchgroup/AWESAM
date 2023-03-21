import numpy as np
import pandas as pd
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import obspy

from .. import CatalogPipeline, EventDetection, util
from .. import config
from . import test_util

matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(
    color=plt.cm.inferno.colors[50::50]
)


def test_catalog(
    name: str,
    time: dt.datetime,
    duration=dt.timedelta(hours=1),
    stations: list = None,
    plot_unfiltered: bool = True,
    plot_maxfilter: bool = False,
) -> pd.DataFrame:
    """
    Generate test_plots of the catalog with given name.
    """
    if stations is None:
        station_seeds = util.get_station_seeds(config.settings["general"]["stations"])

    catalog = CatalogPipeline._load_catalog(
        config.settings["general"]["catalog_directory"], name, "FINAL", suffix=".csv"
    )

    fig = create_test_plot(
        time,
        catalog,
        station_seeds=station_seeds,
        duration=duration,
        plot_unfiltered=plot_unfiltered,
        plot_maxfilter=plot_maxfilter,
    )

    return fig, catalog


def create_test_plot(
    time: dt.datetime,
    ev_catalog: pd.DataFrame,
    station_seeds: list,
    duration=dt.timedelta(hours=1),
    plot_unfiltered: bool = True,
    plot_maxfilter: bool = False,
) -> plt.Figure:
    streams, streams_filtered = [], []

    downsampling = config.settings["EventDetection"]["downsampling"]
    sampling_rate = util.get_sampling_rate(station_seeds[0])

    for station_seed in station_seeds:
        stream = test_util.get_waveforms(
            time,
            duration,
            channel=util.get_channels(station_seed)[0],
            station=station_seed,
        )
        stream = config.settings["EventCatalog"]["preparation_function"](stream)
        if stream is not None:
            _trim(stream.merge(), time, time + duration)
            stream_filtered = stream.copy()
            for trace in stream_filtered:
                if isinstance(trace.data, np.ma.masked_array):
                    trace.data = np.ma.filled(trace.data, fill_value=0)
            if config.settings["EventCatalog"]["event_detection_filter"] is None:
                print("Warning: No filter specified")
            else:
                stream_filtered.filter(
                    **config.settings["EventCatalog"]["event_detection_filter"]
                )
        else:
            stream_filtered = None

        streams.append(stream)
        streams_filtered.append(stream_filtered)

    def date_to_sec(dates):
        if hasattr(dates, "__iter__"):
            return np.array([(d - time).total_seconds() for d in dates])
        else:
            return (dates - time).total_seconds()

    # filter catalogs
    ev_catalog = ev_catalog[
        (ev_catalog["time"] > time) & (ev_catalog["time"] < time + duration)
    ]

    # plot
    fig, ax = plt.subplots(
        len(station_seeds),
        2,
        figsize=test_util.FIGSIZE,
        gridspec_kw={
            "width_ratios": (40, 1),
            "height_ratios": [5] + [2] * (len(station_seeds) - 1),
        },
        dpi=test_util.DPI,
        sharex="col",
        squeeze=False,
    )
    fig.subplots_adjust(wspace=0.02, hspace=0.04)

    # colorbar
    colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.inferno), cax=ax[0, 1])
    colorbar.ax.set_ylabel("probability")

    # data
    for i, (a, station_seed, st, st_filt) in enumerate(
        zip(ax[:, 0], station_seeds, streams, streams_filtered)
    ):
        if st is not None and plot_unfiltered is True:
            a.plot(
                st[0].data,
                label=f"{station_seed} ({time.strftime('%Y-%m-%d %H:%M')})",
                alpha=1 if i == 0 else 0.5,
            )
        if st_filt is not None:
            a.plot(
                st_filt[0].data,
                label=f"{station_seed} (filtered)",
                alpha=1 if i == 0 else 0.5,
            )

    event_detection_settings = {
        "kernel_factor": config.settings["EventDetection"]["kernel_factor"],
        "downsampling": config.settings["EventDetection"]["downsampling"],
        "threshold_factor": config.settings["EventDetection"]["threshold_factor"],
        "threshold_window_size": config.settings["EventDetection"][
            "threshold_window_size"
        ],
        "min_max_kernel_size": config.settings["EventDetection"]["min_max_kernel_size"],
    }

    # maxfilter
    if (st is not None) and (st[0] is not None) and plot_maxfilter:
        _, _, _, _, maxfilter, _ = EventDetection.find_peaks(
            st_filt[0].data, **event_detection_settings, return_extras=True
        )
        ax[0, 0].plot(np.arange(len(maxfilter)) * downsampling, maxfilter)

    # events
    cm = plt.cm.inferno
    cm.set_bad("teal")

    if "event_probability" in ev_catalog.columns:
        color = (ev_catalog["event_probability"],)
    else:
        color = "teal"
    ax[0, 0].scatter(
        date_to_sec(ev_catalog["time"]) * sampling_rate,
        ev_catalog["HHN"],
        marker="x",
        s=150,
        linewidth=3,
        c=color,
        cmap=cm,
        label="events",
        vmin=0,
        vmax=1,
        plotnonfinite=True,
        zorder=200,
    )

    ax[0, 0].set_ylabel("amplitude (in counts)")
    ax[-1, 0].set_xlabel("time (in minutes)")

    def f(x, pos):
        return int(x / sampling_rate / 60)

    formatter = matplotlib.ticker.FuncFormatter(f)

    if streams[0] is not None:
        ax[-1, 0].xaxis.set_major_formatter(formatter)
        ax[0, 0].set_xticks(
            np.arange(
                0,
                len(streams[0][0].data) + 1,
                round(len(streams[0][0].data) / 6 / 60) * 60,
            )
        )

    for i in range(0, len(station_seeds)):
        ax[i, 0].legend(loc="upper right")

    for i in range(1, len(station_seeds)):
        ax[i, 1].axis("off")

    return fig


def _trim(stream, start, end):
    start = obspy.UTCDateTime(start)
    end = obspy.UTCDateTime(end)
    stream.trim(start, end, pad=True)
