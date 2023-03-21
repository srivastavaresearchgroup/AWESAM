import os
import numpy as np
import pandas as pd
import datetime as dt
from awesam import config
import types


def index_to_time(
    a: np.ndarray,
    start: dt.datetime,
    sampling_rate: int,
    vectorize=np.vectorize(
        lambda x: dt.timedelta(seconds=float(x)), otypes=[dt.timedelta]
    ),
):
    """
    Converts an array of integer or a single integer to an array
    of `dt.datetime`, using `config.EventDetection.sampling_rate`
    """
    return start + vectorize(a) / (
        sampling_rate / config.settings["EventDetection"]["downsampling"]
    )


def time_to_index(
    a: pd.Series,
    start: dt.datetime,
    sampling_rate: int,
    vectorize=np.vectorize(lambda x: x.total_seconds(), otypes=[int]),
):
    """
    Converts an array of `dt.datetime` objects or a single datetime object
    to an array of indices.
    """
    new_sampling_rate = (
        sampling_rate / config.settings["EventDetection"]["downsampling"]
    )
    try:
        return vectorize(a - start) * new_sampling_rate
    except Exception:
        to_pydt = np.vectorize(lambda x: pd.Timedelta(x))
        return vectorize(to_pydt(a - start)) * new_sampling_rate


def increment_date(date: dt.datetime, timedelta: dt.timedelta = dt.timedelta(days=1)):
    return date + timedelta


# def stations_from_identifiers(identifiers: list) -> list:
#    return [i.split('.')[1] for i in identifiers]


def get_station_seeds(seed_codes: list) -> list:
    return [s["station"] for s in seed_codes]


def get_station_from_seed(station_code: str) -> str:
    return station_code.split(".")[1]


def get_station_names(seed_codes: list) -> list:
    return [s["station"].split(".")[1] for s in seed_codes]


def get_station_name_from_seed(seed: str) -> list:
    return seed.split(".")[1]


def get_channels(station_code: str) -> list:
    station = [
        s
        for s in config.settings["general"]["stations"]
        if s["station"] == station_code
    ][0]
    return station["channels"]


def get_sampling_rate(station_code: str) -> int:
    station = [
        s
        for s in config.settings["general"]["stations"]
        if s["station"] == station_code
    ][0]
    return station["sampling_rate"]


def check_settings_integrity():
    """
    Checks if the dictionary config.settings is valid:
    1. Raises error, if key is missing or redundant
    2. Raises error, if key has wrong type
    3. Raises error, if directories do not exist
    """

    settings_types = {
        "general.pipeline": dict,
        "general.service": str,
        "general.stations": list,
        "general.coordinates": np.ndarray,
        "general.mseed_directory": str,
        "general.catalog_directory": str,
        "general.earthquake_catalog_directory": str,
        "EventDetection.threshold_window_size": int,
        "EventDetection.downsampling": (float, int),
        "EventDetection.kernel_factor": (float, int),
        "EventDetection.min_max_kernel_size": list,
        "EventDetection.maxfilter_kernel": int,
        "EventDetection.threshold_factor": (float, int),
        "EventDetection.duration_height": (float, int),
        "EventCatalog.preparation_function": types.FunctionType,
        "EventCatalog.taper": dict,
        "EventCatalog.event_detection_filter": (dict, type(None)),
        "EventCatalog.amplitude_detection_filter": (dict, type(None)),
        "CatalogConsolidation.default_metric": np.ndarray,
        "CatalogConsolidation.window_length": int,
        "EarthquakeClassification.intensity_metric": types.FunctionType,
        "EarthquakeClassification.amplitude_metric": types.FunctionType,
        "EarthquakeClassification.travel_time_metric": types.FunctionType,
        "EarthquakeClassification.intensity_threshold": (float, int),
        "EarthquakeClassification.earthquake_metric": np.ndarray,
    }
    try:
        use_catalog_consolidation = config.settings["general"]["pipeline"][
            "use_catalog_consolidation"
        ]
        use_earthquake_classification = config.settings["general"]["pipeline"][
            "use_earthquake_classification"
        ]
    except:
        raise ValueError(
            "The keys settings['general']['pipeline']['use_catalog_consolidation']"
            + "or settings['general']['pipeline']['use_earthquake_classification'] are missing."
        )

    settings = pd.json_normalize(config.settings, sep=".", max_level=1).to_dict(
        orient="records"
    )[0]

    keys_settings = set(settings.keys())
    keys_ref = set(settings_types.keys())

    if not use_catalog_consolidation:
        keys_ref = set([k for k in keys_ref if k.find("CatalogConsolidation") == -1])
    if not use_earthquake_classification:
        keys_ref = set(
            [k for k in keys_ref if k.find("EarthquakeClassification") == -1]
        )

    # CHECK IF KEYS ARE COMPLETE
    if len(keys_ref.difference(keys_settings)):
        raise ValueError(
            f"In config.settings the keys {keys_ref.difference(keys_settings)} are missing "
            + f"and {keys_settings.difference(keys_ref)} are redundant"
        )

    # CHECK IF KEYS HAVE CORRECT TYPE
    for key in keys_ref:
        if not isinstance(settings[key], settings_types[key]):
            raise ValueError(
                f"The key '{key}' in config.settings has an invalid type. Found: {type(settings[key])}, expected: {settings_types[key]} "
            )

    # SPECIFIC CHECKS
    if len(settings["general.coordinates"]) != 2:
        raise ValueError(
            f"settings['general']['coordinates'] must be a np.ndarray with two floats."
        )
    if not os.path.isdir(settings["general.mseed_directory"]):
        raise ValueError(
            f"settings['general']['mseed_directory'] must be an existing directory"
        )
    if not os.path.isdir(settings["general.catalog_directory"]):
        raise ValueError(
            f"settings['general']['catalog_directory'] must be an existing directory"
        )
    if use_earthquake_classification and not os.path.isdir(
        settings["general.earthquake_catalog_directory"]
    ):
        raise ValueError(
            f"settings['general']['earthquake_catalog_directory'] must be an existing directory"
        )
