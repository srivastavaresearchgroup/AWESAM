import numpy as np
import obspy


def stromboli_preparation_function(stream: obspy.Stream) -> obspy.Stream:
    return stream


stromboli_settings = {
    "general": {
        "pipeline": {
            "use_catalog_consolidation": True,
            "use_earthquake_classification": False,
        },
        "service": "INGV",
        "stations": [
            {
                "station": "IV.IST3.--",
                "channels": ["HHN", "HHE", "HHZ"],
                "sampling_rate": 100,
            },
            {
                "station": "IV.ISTR.--",
                "channels": ["HHN", "HHE", "HHZ"],
                "sampling_rate": 100,
            },
        ],
        "coordinates": np.array([38.793315, 15.211588]),
        "mseed_directory": "/path/to/...",
        "catalog_directory": "/path/to/...",
        "earthquake_catalog_directory": "/path/to/...",
    },
    "EventDetection": {
        "threshold_window_size": 5 * 60 * 100,
        "downsampling": 100,
        "kernel_factor": 0.35,
        "min_max_kernel_size": [100, 1_000_000],
        "maxfilter_kernel": 1600,
        "threshold_factor": 1.0,
        "duration_height": 0.5,
    },
    "EventCatalog": {
        "preparation_function": stromboli_preparation_function,
        "taper": {"max_length": 5, "type": "hann", "max_percentage": 0.1},
        "event_detection_filter": {"type": "bandpass", "freqmin": 0.7, "freqmax": 5},
        "amplitude_detection_filter": {
            "type": "bandpass",
            "freqmin": 0.7,
            "freqmax": 10,
        },
    },
    "CatalogConsolidation": {
        "default_metric": np.array([200, 0, 0, 0.1], dtype=np.double),
        "window_length": 500,
    },
    "EarthquakeClassification": {
        "travel_time_metric": (lambda distance: distance / 6.6),
        "intensity_metric": lambda magnitude, dist: magnitude - np.log(dist),
        "amplitude_metric": lambda intensity: 10 ** (0.83 * intensity + 5.36),
        "intensity_threshold": -3.0,
        "earthquake_metric": np.array([20, 0, 0, 0.01], dtype=np.double),
    },
}


def whakaari_preparation_function(stream: obspy.Stream) -> obspy.Stream:
    return stream


whakaari_settings = {
    "general": {
        "pipeline": {
            "use_catalog_consolidation": False,
            "use_earthquake_classification": False,
        },
        "service": "GEONET",
        "stations": [
            {
                "station": "NZ.WIZ.10",
                "channels": ["HHN", "HHE", "HHZ"],
                "sampling_rate": 100,
            }
        ],
        "coordinates": np.array([-37.518056, 177.181389]),
        "mseed_directory": "/path/to/...",
        "catalog_directory": "/path/to/...",
        "earthquake_catalog_directory": "",
        "delete_temporary_files": True,
    },
    "EventDetection": {
        "threshold_window_size": 150 * 60 * 100,
        "threshold_factor": 1.0,
        "maxfilter_kernel": 5000,
        "downsampling": 1000,
        "kernel_factor": 1.0,
        "duration_height": 0.5,
        "min_max_kernel_size": [100, 1_000_000],
    },
    "EventCatalog": {
        "preparation_function": whakaari_preparation_function,
        "taper": {"max_length": 5, "type": "hann", "max_percentage": 0.1},
        "event_detection_filter": {"type": "bandpass", "freqmin": 0.7, "freqmax": 5},
        "amplitude_detection_filter": {
            "type": "bandpass",
            "freqmin": 0.7,
            "freqmax": 10,
        },
    },
    "CatalogConsolidation": {
        "default_metric": np.array(
            [200, 0, 0, 0.1], dtype=np.double
        ),  # lambda y: [200/y, 0.1/y],
        "window_length": 500,  # seconds
    },
    "EarthquakeClassification": {
        "travel_time_metric": (lambda distance: distance / 6.6),
        "intensity_metric": lambda magnitude, dist: magnitude - np.log(dist),
        "amplitude_metric": lambda intensity: 10 ** (0.83 * intensity + 5.36),
        "intensity_threshold": -3.0,
        "earthquake_metric": np.array([20, 0, 0, 0.01], dtype=np.double),
    },
}


def yasur_preparation_function(stream: obspy.Stream) -> obspy.Stream:
    """correct for amplitude difference"""
    if stream[0].stats.station == "Y31":
        for tr in stream:
            tr.data *= 4
    return stream


yasur_settings = {
    "general": {
        "pipeline": {
            "use_catalog_consolidation": True,
            "use_earthquake_classification": False,
        },
        "service": "RESIF",
        "stations": [
            {
                "station": "ZO.Y32.00",
                "channels": ["HHN", "HHE", "HHZ"],
                "sampling_rate": 100,
            },
            {
                "station": "ZO.Y31.00",
                "channels": ["HHN", "HHE", "HHZ"],
                "sampling_rate": 100,
            },
        ],
        "coordinates": np.array([-19.529281, 169.448113]),
        "mseed_directory": "/path/to/...",
        "catalog_directory": "/path/to/...",
        "earthquake_catalog_directory": "",
    },
    "EventDetection": {
        "threshold_window_size": 10 * 60 * 100,
        "threshold_factor": 0.5,
        "maxfilter_kernel": 1600,
        "downsampling": 100,
        "kernel_factor": 0.05,
        "duration_height": 0.5,
        "min_max_kernel_size": [100, 100_000],
    },
    "EventCatalog": {
        "preparation_function": yasur_preparation_function,
        "taper": {"max_length": 5, "type": "hann", "max_percentage": 0.1},
        "event_detection_filter": {"type": "bandpass", "freqmin": 0.7, "freqmax": 5},
        "amplitude_detection_filter": {
            "type": "bandpass",
            "freqmin": 0.7,
            "freqmax": 10,
        },
    },
    "CatalogConsolidation": {
        "default_metric": np.array(
            [200, 0, 0, 0.5], dtype=np.double
        ),  # lambda y: [200/y, 0.1/y],
        "window_length": 500,  # seconds
    },
    "EarthquakeClassification": {
        "intensity_metric": lambda magnitude, dist: magnitude - np.log(dist),
        "amplitude_metric": lambda intensity: 10 ** (0.83 * intensity + 5.36),
        "travel_time_metric": (lambda distance: distance / 6.6),
        "intensity_threshold": -3.0,
        "earthquake_metric": np.array(
            [20, 0, 0, 0.01], dtype=np.double
        ),  # lambda y: [20/y, 0.01/y],
    },
}


def etna_preparation_function(stream: obspy.Stream) -> obspy.Stream:
    return stream


etna_settings = {
    "general": {
        "pipeline": {
            "use_catalog_consolidation": True,
            "use_earthquake_classification": False,
        },
        "service": "INGV",
        "stations": [
            {
                "station": "IV.ECPN.--",
                "channels": ["HHN", "HHE", "HHZ"],
                "sampling_rate": 100,
            },
            {
                "station": "IV.ECNE.--",
                "channels": ["HHN", "HHE", "HHZ"],
                "sampling_rate": 100,
            },
        ],
        "coordinates": np.array([37.753898, 14.995958]),
        "mseed_directory": "/path/to/...",
        "catalog_directory": "/path/to/...",
        "earthquake_catalog_directory": "/path/to/...",
    },
    "EventDetection": {
        "threshold_window_size": 5 * 60 * 100,  # samples
        "downsampling": 100,
        "kernel_factor": 0.35,
        "min_max_kernel_size": [100, 1_000_000],
        "maxfilter_kernel": 1600,  # samples
        "threshold_factor": 1.0,
        "duration_height": 0.5,
    },
    "EventCatalog": {
        "preparation_function": etna_preparation_function,
        "taper": {"max_length": 5, "type": "hann", "max_percentage": 0.1},
        "event_detection_filter": {"type": "bandpass", "freqmin": 0.7, "freqmax": 5},
        "amplitude_detection_filter": {
            "type": "bandpass",
            "freqmin": 0.7,
            "freqmax": 10,
        },
    },
    "CatalogConsolidation": {
        "default_metric": np.array([200, 0, 0, 0.02], dtype=np.double),
        "window_length": 500,  # seconds
    },
    "EarthquakeClassification": {
        "travel_time_metric": (lambda distance: distance / 6.6),
        "intensity_metric": lambda magnitude, dist: magnitude - np.log(dist),
        "amplitude_metric": lambda intensity: 10 ** (0.83 * intensity + 5.36),
        "intensity_threshold": -3.0,
        "earthquake_metric": np.array(
            [20, 0, 0, 0.01], dtype=np.double
        ),  # lambda y: [20/y, 0.01/y],
    },
}
