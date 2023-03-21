"""
Uses the EventDetection algorithm to create an event catalog from seismic data.
The function `create_catalog()` generates an event- and gap-catalog (from global_start to global_end) and saves them as .csv-files.
"""

import numpy as np
import pandas as pd
import datetime as dt
import logging
import os
import typing

import obspy
from awesam import EventDetection, util
from awesam import config


# Logging
logger = logging.getLogger("Catalog")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)


class Signal:
    def __init__(self, date: dt.datetime, streams: dict, station_seed: str):
        """
        Class for packaging all streams from one day. For example if channels include HHN, HHE and HHZ, the
        dict should be in the following form:

        >>> {'HHE': stream1, 'HHN': stream2, 'HHZ': stream3}.

        If stream is missing, the correponding dict entry should be `None`.
        """
        self.date = date

        self.channels = util.get_channels(station_seed)
        self.sampling_rate = util.get_sampling_rate(station_seed)

        if sorted(streams.keys()) != sorted(self.channels):
            raise ValueError("Channels must match channels defined in config.py")

        self.streams = dict(sorted(streams.items()))

        # create catalog that lists all gaps in the data
        self.create_gap_catalog()
        # merge all traces and trim to full day
        self.prepare_streams(part1=True, part2=False)
        # create main stream from all channels
        self.create_main_stream()
        self.prepare_streams(part1=False, part2=True)

    to_datetime = np.vectorize(lambda v: v.datetime)

    def create_gap_catalog(self) -> None:
        """
        Catalog containing the channel, start and end of the gaps in the streams.
        The result (`pandas.DataFrame`) is stored in `self.gap_catalog`.
        Also includes gaps if stream starts too late or ends too early.
        """
        gaps_list = []

        for channel in self.channels:
            stream = self.streams[channel]

            if stream is None:  # if channel is completely missing
                gaps_list.append([channel, self.date, util.increment_date(self.date)])
            else:
                gaps = np.array(stream.get_gaps())

                not_overlapping_gaps = []
                for i, gap in enumerate(gaps):  # if gap has negative length (overlaps)
                    if (gap[5] - gap[4]) > 0:
                        not_overlapping_gaps.append(i)
                gaps = gaps[not_overlapping_gaps]

                min_start = min([tr.stats.starttime.datetime for tr in stream])
                max_end = max([tr.stats.endtime.datetime for tr in stream])

                if len(gaps) != 0:  # append intermediate gaps
                    gaps = np.array(gaps)[:, [3, 4, 5]]
                    gaps[:, [1, 2]] = self.to_datetime(gaps[:, [1, 2]])
                    gaps_list += gaps.tolist()

                if self.date < min_start:  # if stream starts too late
                    start, end = self.date, min_start
                    gaps_list.append([channel, start, end])

                if util.increment_date(self.date) > max_end:  # if stream ends too early
                    start, end = max_end, util.increment_date(self.date)
                    gaps_list.append([channel, start, end])

        self.gap_catalog = pd.DataFrame(gaps_list, columns=["channel", "start", "end"])

    def prepare_streams(self, part1: bool = True, part2: bool = True) -> None:
        """
        Merges, trims and sets masked values of stream for preparation for further processing (inplace).
        - part 1: merge traces, trim start end end to whole day,
        - part 2: fill masked values with 0
        Part 1 has to be applied first.
        """

        def merge_and_trim(stream):
            stream.taper(**config.settings["EventCatalog"]["taper"])
            stream.merge()
            start, end = stream[0].stats.starttime, stream[0].stats.endtime
            stream.trim(
                obspy.UTCDateTime(self.date),
                obspy.UTCDateTime(util.increment_date(self.date)),
                pad=True,
                nearest_sample=False,
            )

        def fill_masked(stream):
            for trace in stream:
                if isinstance(trace.data, np.ma.masked_array):
                    trace.data = np.ma.filled(trace.data, fill_value=0)

        for channel in self.channels:
            stream = self.streams[channel]
            if stream is not None:
                if part1:
                    merge_and_trim(stream)
                if part2:
                    fill_masked(stream)

        if hasattr(self, "main_stream"):
            if self.main_stream is not None:
                if part1:
                    merge_and_trim(self.main_stream)
                if part2:
                    fill_masked(self.main_stream)

    def _get_existing_channels(self) -> list:
        """returns a list containing the channel-names of existant streams"""

        channels = []
        for channel in self.channels:
            if self.streams[channel] is not None:
                channels.append(channel)

        return channels

    def create_main_stream(self) -> None:
        """
        complements the main channel with the all available channels as best as possible
        and stores it in `self.main_stream`
        """
        # after merge, after gap catalog creation

        def fill_with_stream(main_trace, other, gaps):
            # fills the gaps of the main_trace with the information from other
            for slc in gaps:
                other_chunk = other.data[slc]

                if len(other_chunk) != slc.stop - slc.start:
                    print(f"Warning: Problem with main stream: Gap could not be filled")
                    slc = slice(slc.start, slc.stop - 1)  # rare error correction

                # set data
                main_trace.data[slc] = other_chunk

                # set mask to 0 where gap could be filles sucessfully
                if isinstance(other_chunk, np.ma.masked_array):
                    main_trace.data.mask[slc] = other.data.mask[slc]
                else:
                    main_trace.data.mask[slc] = 0

        # get all channels that are not None
        channels = self._get_existing_channels()

        if not channels:  # if nothing is there at all
            self.main_stream = None
            return

        main_trace = self.streams[channels[0]][
            0
        ]  # using channel order defined in config.py
        gaps = np.ma.clump_masked(main_trace.data)  # extract gaps (from masked array)

        if gaps:
            # fill gaps in main_trace with other available channels (`channels[1:]`)
            for channel in channels[1:]:
                fill_with_stream(main_trace, self.streams[channel][0], gaps)

        self.main_stream = obspy.Stream([main_trace])

    def __iter__(self):
        """iterates over all channels if existant"""
        self.iteration_index = self.streams.keys
        return self

    def __next__(self):
        try:
            return self.streams[self.iteration_index.pop(0)]
        except:
            raise StopIteration

    def __len__(self):
        # count non-empty streams
        return sum([0 if self.streams[s] is None else 1 for s in self.streams])

    def __repr__(self):
        num_streams = sum([0 if self.streams[s] is None else 1 for s in self.streams])
        string = (
            f"Stream ({self.date.strftime('%Y-%m-%d')}) with {num_streams} channels"
        )
        return string


def create_catalog(
    global_start: dt.datetime,
    global_end: dt.datetime,
    read_directory: str,
    write_directory: str,
    name: str,
    station_seed: str,
    checkpoint=False,
) -> None:
    """
    Generates catalogs for seismic data between `global_start` and `global_end`. `name` is the identifier for the final catalogs.
    Use MassDownload, to get the mseed files in the correct folder structure and naming convention.
    If `checkpoint` is `True`, already generated files (if existing) are used.

    Generated catalogs:
    - `event_catalog`: Containing the amplitude and time of each event in the seismic data
    - `gaps_catalog`: Lists all gaps in the seimic data (start and end time)
    """

    if not checkpoint:
        global_event_catalog = []
        global_gaps_catalog = []

        date, i = global_start, 0
    else:
        checkpoint_info = _check_checkpoint_files(
            name,
            write_directory,
            global_start,
            util.get_station_from_seed(station_seed),
        )
        global_event_catalog = checkpoint_info[1]
        global_gaps_catalog = checkpoint_info[2]

        date, i = checkpoint_info[0], 0

    while date <= global_end:

        event_catalog, gap_catalog = _load_data_and_create_catalogs(
            date, read_directory, station_seed
        )

        if event_catalog is not None:
            global_event_catalog.append(event_catalog)
            global_gaps_catalog.append(gap_catalog)
        else:
            global_gaps_catalog.append(gap_catalog)
        if (i + 1) % 20 == 0:  # save checkpoints
            _save_catalogs(
                global_event_catalog,
                global_gaps_catalog,
                write_directory,
                name,
                station_seed.split(".")[1],
            )

        date += dt.timedelta(days=1)
        i += 1

    _save_catalogs(
        global_event_catalog,
        global_gaps_catalog,
        write_directory,
        name,
        station_seed.split(".")[1],
    )


def _load_data_and_create_catalogs(
    date: dt.datetime, read_directory: str, station_seed: str
) -> tuple:
    """
    Loads one day of seismic data (in .mseed format, with obspy) from the specified directory.
    The naming convention of the .mseed-file must follow the one defined in MassDownload.
    """
    filename = os.path.join(
        read_directory,
        str(date.year),
        date.strftime("%Y-%m-%d")
        + "_"
        + (date + dt.timedelta(days=1)).strftime("%Y-%m-%d"),
    )

    streams = {}
    channels = util.get_channels(station_seed)

    for channel in channels:
        try:
            streams[channel] = obspy.read(filename + f"_{channel}.mseed")
        except FileNotFoundError:
            streams[channel] = None
        except Exception as e:
            print(
                "FILE READ ERROR (try to redownload): " + filename + f"_{channel}.mseed"
            )
            raise e

    # custom preparation function
    for channel in streams:
        if streams[channel] is not None:
            streams[channel] = config.settings["EventCatalog"]["preparation_function"](
                streams[channel]
            )
    streams = Signal(date, streams, station_seed)

    if len(streams) != 0:
        event_catalog = _create_event_catalog(streams, date)
        return event_catalog, streams.gap_catalog
    else:
        empty_event_catalog = pd.DataFrame(
            [], columns=["duration", "time"] + channels + ["amplitude"]
        )
        return empty_event_catalog, streams.gap_catalog


def _create_event_catalog(
    streams: Signal, date: dt.datetime
) -> typing.Optional[pd.DataFrame]:
    """
    Event catalog containing time and amplitude of all events in the seismic data.
    Returns `None` if all streams are missing.
    """

    # part 1: detect time of events
    stream = streams.main_stream
    if stream is not None:
        stream = stream.copy()  # copy main stream
        if config.settings["EventCatalog"]["event_detection_filter"] is not None:
            stream.filter(**config.settings["EventCatalog"]["event_detection_filter"])
        event_times, widths = EventDetection.find_peaks(
            stream[0].data,
            config.settings["EventDetection"]["kernel_factor"],
            config.settings["EventDetection"]["downsampling"],
            config.settings["EventDetection"]["threshold_factor"],
            config.settings["EventDetection"]["threshold_window_size"],
            config.settings["EventDetection"]["min_max_kernel_size"],
        )

        # part 2: get amplitude and duration of events
        catalog = {}
        catalog["time"] = None  # declaration
        catalog["duration"] = widths
        for channel in streams.channels:
            if streams.streams[channel] is not None:
                if (
                    config.settings["EventCatalog"]["amplitude_detection_filter"]
                    is not None
                ):
                    data = (
                        streams.streams[channel]
                        .filter(
                            **config.settings["EventCatalog"][
                                "amplitude_detection_filter"
                            ]
                        )[0]
                        .data
                    )
                else:
                    data = streams.streams[channel][0].data

                catalog[channel] = EventDetection.find_amplitudes(
                    data,
                    event_times,
                    config.settings["EventDetection"]["downsampling"],
                    config.settings["EventDetection"]["maxfilter_kernel"],
                )
            else:  # if channel does not exist
                catalog[channel] = np.full(len(event_times), 0)

        catalog["time"] = util.index_to_time(event_times, date, streams.sampling_rate)

        catalog = pd.DataFrame(catalog)

        catalog["amplitude"] = _get_amplitude_from_channels(catalog, streams.channels)
        return catalog
    else:
        return None


def _get_amplitude_from_channels(catalog: pd.DataFrame, channels) -> pd.Series:
    """compute mean amplitude over all available channels"""
    c = catalog[channels].copy()
    c.replace(0, np.nan, inplace=True)  # 0 stands for channel not available
    return catalog[channels].mean(axis=1)


def _save_catalogs(
    event_catalog: typing.List[pd.DataFrame],
    gaps_catalog: typing.List[pd.DataFrame],
    write_directory: str,
    name: str,
    station: str,
) -> None:
    """
    Saves the event_catalog and gap_catalog as csv.
    """

    for catalog, cname in zip(
        (event_catalog, gaps_catalog), ("_events.csv", "_gaps.csv")
    ):
        catalog = pd.concat(catalog)
        catalog.reset_index(inplace=True, drop=True)
        catalog.to_csv(
            os.path.join(write_directory, name + "_" + station.lower() + cname),
            index=False,
        )


def _check_checkpoint_files(
    name: str, directory: str, start_date: dt.datetime, station: str
):
    """
    Check disk if any checkpoint file already exists. If yes the new start date is determined
    and the existing catalogs are returned.
    """
    try:
        df = pd.read_csv(os.path.join(directory, f"{name}_{station}_events.csv"))

        if len(df) == 0:
            raise FileNotFoundError  # although it is not a file not found error
    except FileNotFoundError:
        return start_date, [], [], []
    else:
        last_event = pd.to_datetime(df.iloc[-1]["time"])
        start_date = dt.datetime(
            last_event.year, last_event.month, last_event.day
        ) + dt.timedelta(days=1)
        logger.info(f"Using catalog checkpoint from disk. New start date: {start_date}")

        event_catalog, gap_catalog = _load_catalogs(name, directory, station)

        return start_date, [event_catalog], [gap_catalog]


def _load_catalogs(name: str, directory: str, station: str):
    """Loads event- and gap- catalog"""
    event_catalog = pd.read_csv(os.path.join(directory, f"{name}_{station}_events.csv"))
    gap_catalog = pd.read_csv(os.path.join(directory, f"{name}_{station}_gaps.csv"))

    event_catalog["time"] = pd.to_datetime(event_catalog["time"])
    gap_catalog["start"] = pd.to_datetime(gap_catalog["start"])
    gap_catalog["end"] = pd.to_datetime(gap_catalog["end"])

    return event_catalog, gap_catalog
