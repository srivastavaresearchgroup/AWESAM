import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate
import os

from awesam.tests import test_util

from .. import config, util


def plot_downloaded_files():
    maprange = scipy.interpolate.interp1d([-1, 3], [0, 1], bounds_error=False)

    existing_files = []
    root_directory = config.settings["general"]["mseed_directory"]
    stations = util.get_station_seeds(config.settings["general"]["stations"])

    for station in stations:
        station_directory = os.path.join(
            root_directory, station.split(".")[1].lower(), "mseed"
        )

        for year in os.listdir(station_directory):
            year_directory = os.path.join(station_directory, year)
            files = os.listdir(year_directory)

            for file in files:
                date = pd.to_datetime(file[:10])
                channel = file[22:25]
                existing_files.append(
                    [station, channel, date.year, date.timetuple().tm_yday]
                )

    existing_files_df = pd.DataFrame(
        existing_files, columns=["station", "channel", "year", "day"]
    )
    existing_files_df.sort_values(["station", "channel", "year", "day"], inplace=True)
    channel_id = np.empty(len(existing_files_df))
    station_id = np.empty(len(existing_files_df))

    years = existing_files_df["year"].unique()
    channels = existing_files_df["channel"].unique()
    stations = existing_files_df["station"].unique()

    for i, channel in enumerate(channels):
        channel_id[existing_files_df["channel"] == channel] = i
    for i, station in enumerate(stations):
        station_id[existing_files_df["station"] == station] = i

    y_coord = existing_files_df["year"] + channel_id / 15 + station_id / 4 - 0.2
    color_value = maprange(station_id + 0.3 * channel_id)

    fig, ax = plt.subplots(figsize=test_util.FIGSIZE, dpi=test_util.DPI)
    ax.scatter(
        existing_files_df["day"],
        y_coord,
        c=plt.cm.inferno(color_value),
        marker="|",
        s=7,
    )

    for i, station in enumerate(stations):
        for j, channel in enumerate(channels):
            ax.plot(
                [],
                [],
                label=station + "." + channel,
                color=plt.cm.inferno(maprange(i + 0.3 * j).item()),
            )

    ax.set_yticks(range(years.min(), years.max() + 1))

    ax.set_xlabel("day in year")
    ax.set_ylabel("year")
    ax.set_title("Downloaded Data")
    ax.legend(loc="upper right")

    return fig
