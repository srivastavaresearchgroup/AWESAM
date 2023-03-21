"""
Function `download` allows downloading `.mseed`-data for large time intervals.
"""

import obspy.clients.fdsn.mass_downloader as mass_downloader
import multiprocessing
import datetime as dt
import pandas as pd
import numpy as np
import itertools
import logging
import os
import typing
from awesam import config


def download(
    start: dt.datetime,
    end: dt.datetime,
    stations: typing.List[dict],
    location: np.ndarray,
    directory: str,
    service: str,
):
    """
    Download and save mseed data from server between given `start` and `end` date.
    `.mseed`-Files (one file per day, per channel, per station) are saved in `location`.
    - start: start date
    - end: end date
    - stations: List of dictionaries in the following form:
        {'station': 'network.station.location', 'channels': [...], 'sampling_rate': ...}
    - location: location of the volcano (search for stations in proximity to this location)
    - directory: where to store mseed files.
    - service: FDSN-network service
    """

    def _get_mseed_directory(network, station, location, channel, starttime, endtime):
        return os.path.join(
            directory,
            station.lower(),
            "mseed",
            f"{starttime.year}",
            f"{starttime.strftime('%Y-%m-%d')}_{endtime.strftime('%Y-%m-%d')}_{channel}.mseed",
        )

    # Logging
    logger = logging.getLogger("obspy.clients.fdsn.mass_downloader")
    logger.name = f"mass_downloader.{start.year}.{start.month}.{start.day}"

    domain = mass_downloader.RectangularDomain(
        minlatitude=location[0]
        - 0.5,  # search for stations in a radius of 0.5 latitude/longitude
        maxlatitude=location[0] + 0.5,
        minlongitude=location[1] - 0.5,
        maxlongitude=location[1] + 0.5,
    )

    for station in stations:
        logger.info(f"START DOWNLOAD FOR {station['station']}")

        try:
            network, station_name, station_location = station["station"].split(".")
        except ValueError:
            raise ValueError(
                'Station name must contain "[network].[station].[location]"'
            )

        restrictions = mass_downloader.Restrictions(
            starttime=start,
            endtime=end,
            chunklength_in_sec=86400,  # one file per day
            network=network,
            station=station_name,
            location=station_location,
            channel=",".join(station["channels"]),
            reject_channels_with_gaps=False,
            minimum_length=0.0,
        )
        logger.info(
            f"network={restrictions.network}, station={restrictions.station}, location={restrictions.location}, channel={restrictions.channel}"
        )

        mdl = mass_downloader.MassDownloader(providers=[service])

        mdl.download(
            domain,
            restrictions,
            mseed_storage=_get_mseed_directory,
            stationxml_storage=os.path.join(
                directory, station_name.lower(), "stations"
            ),
        )


def multiprocessing_download(
    start: dt.datetime, end: dt.datetime, n_subprocesses=16, use_multiprocessing=True
):
    """
    Download and save mseed data from server between given `start` and `end` date.
    `.mseed`-Files (one file per day, per channel, per station) is saved.
    The stations to download are configured in config.settings['general'].
    """

    stations = config.settings["general"]["stations"]
    location = config.settings["general"]["coordinates"]
    directory = config.settings["general"]["mseed_directory"]
    service = config.settings["general"]["service"]

    chunk_length = dt.timedelta(days=15)
    dates = pd.date_range(start, end, freq=chunk_length).to_pydatetime()
    chunks = list(itertools.product(*[stations, dates]))

    # compile parameter list
    parameter_list = []
    for station, date_start in chunks:
        if date_start == dates[-1]:  # if last chunk, use end date
            date_end = end
        else:
            date_end = date_start + chunk_length

        parameter_list.append(
            [date_start, date_end, [station], location, directory, service]
        )

    # download data
    if use_multiprocessing:
        p = multiprocessing.Pool(n_subprocesses)
        p.starmap(download, parameter_list)
        p.close()
        p.join()
    else:
        for param in parameter_list:
            download(*param)
