import numpy as np
import datetime as dt
import obspy
from obspy.clients.fdsn import Client

from .. import config

FIGSIZE = (11, 6)
DPI = 200


def get_waveforms(
    start, duration, channel="HHN", station="IV.IST3.--", offset=dt.timedelta(seconds=0)
) -> obspy.Stream:
    client = Client(config.settings["general"]["service"])
    try:
        stream = client.get_waveforms(
            *station.split("."),
            channel,
            obspy.UTCDateTime(start + offset),
            obspy.UTCDateTime(start + duration + offset)
        )
    except obspy.clients.fdsn.header.FDSNNoDataException as e:
        stream = None

    return stream


def select_random_dates(n: int, start: dt.datetime, end: dt.datetime) -> list:
    random_dates = []
    num_days = (end - start).total_seconds() / 60 / 60 / 24

    for i in range(n):
        random_dates.append(
            start
            + dt.timedelta(
                days=np.random.randint(0, num_days), hours=np.random.randint(0, 23)
            )
        )
    return random_dates
