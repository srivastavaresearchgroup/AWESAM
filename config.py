import numpy as np
import obspy


def preparation_function(stream: obspy.Stream) -> obspy.Stream:
    # ... prepare stream
    return stream


settings = {
    "general": {
        "pipeline": {
            # Which steps in CatalogPipeline.multiprocessing_pipeline should be performed?
            "use_catalog_consolidation": True,
            "use_earthquake_classification": True,
        },
        # service used to download data
        "service": "INGV",
        # codes for all stations, in preferred order
        # the first two stations are used for catalog consolidation
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
        # coordinates of the volcano (latitude, longitude)
        "coordinates": np.array([38.793315, 15.211588]),
        # directory to save the downloaded data
        "mseed_directory": "/path/to/...",
        # directory to save the final catalogs
        "catalog_directory": "/path/to/...",
        # directory with the earthquake catalog (only required if general.pipeline.use_earthquake_classification == True)
        "earthquake_catalog_directory": "/path/to/...",
    },
    "EventDetection": {
        # window size used to compute the prominence thresholds and the adaptive kernel sizes. Window in both directions.
        #   Rule of thumb: Should be some multiple of the average event time (e.g. 10x) but not so large,
        #   that changes in noise level are overlooked
        "threshold_window_size": 5 * 60 * 100,  # samples
        # downsampling-factor in maxpooling step.
        #   it controls the maximum time resoluton of the catalog. e.g. with a sampling rate of 100 Hz
        #   and a downsampling-factor of 100, the processed data has a sampling rate of 1 Hz.
        #   The uncertanty of the time of an event is in consequence 1Hz
        #   The higher the value, the better the performance. If events are rare, increasing this value is recommended.
        "downsampling": 100,  # do not change
        # Controls the average kernel size used in adaptive maxpooling
        #   Rule of thumb: the kernels should be small enough, so each event can be distinguished, but large enough,
        #   so no event is detected multiple times.
        "kernel_factor": 0.35,
        # minimum and maximum kernel size in adaptive max filter
        #   This setting is only for extreme cases to prevent memory crashes.
        #   In normal cases the maxfilter kernel size should be in between [min, max]
        #   The minimum value must be positive and cannot be zero.
        "min_max_kernel_size": [100, 1_000_000],
        # kernel size for determining the amplitude (probably no changes needed)
        #   Used in adaptive maxfilter with constant kernel size.
        #   Should be larger than the downsampling factor.
        "maxfilter_kernel": 1600,  # samples
        # factor the prominence threshold is multiplied with
        #   this factor controls the minimum (relative) amplitude of detected events
        "threshold_factor": 1.0,
        # When computing the duration of events, the width of each peak is computed at a specific height.
        #   This setting sets the relative peak height used in scipy.signal.peak_widths,
        "duration_height": 0.5,
    },
    "EventCatalog": {
        # custom function to prepare Stream, before anything is done to them (e.g. response removal)
        #   function signature must be: (stream: obspy.Stream) -> obspy.Stream
        #   function is applied per day to the raw downloaded data.
        #   Recommendation: Use unit counts and convert to physical units afterwards. Because the data
        #   is squared, using different units (e.g. m/s) can lead to numerical errors.
        "preparation_function": preparation_function,
        # tapering at the edge of traces to avoid filtering artifacts
        #   Dictionary will be passed to obspy.Trace.taper.
        "taper": {"max_length": 5, "type": "hann", "max_percentage": 0.1},
        # filter used for detection of events. It should remove local interferences, without suppressing events.
        #   Set to None if data should be used unfiltered.
        #   Dictionary will be passed to obspy.Stream.filter.
        "event_detection_filter": {"type": "bandpass", "freqmin": 0.7, "freqmax": 5},
        # filter used for detection of amplitudes. It should preserve the amplitude as best as possible.
        "amplitude_detection_filter": {
            "type": "bandpass",
            "freqmin": 0.7,
            "freqmax": 10,
        },
    },
    "CatalogConsolidation": {  # (only required if general.pipeline.use_catalog_consolidation == True)
        # metric used for determining the distance between two events in different catalogs
        #   The array [a, b, c, d] is mapped to the matrix ((a,b), (c,d)). The event_probability is computed with:
        #   p = exp( - || ((a,b), (c,d))/y * (Δt, Δy) ||)
        #   where y and Δy are the amplitude and amplitude difference. Usually b=c=0.
        "default_metric": np.array([200, 0, 0, 0.1], dtype=np.double),
        # Maximum temporal distance between two events to be considered as same events.
        #   Should not be too large for performance.
        "window_length": 500,  # seconds
    },
    "EarthquakeClassification": {  # (only required if general.pipeline.use_earthquake_classification == True)
        # estimate the travel time of a given earthquake-distance.
        #   In this case: 6.6 km/s is the average/estimated p-wave velocity.
        "travel_time_metric": (lambda distance: distance / 6.6),
        # estimate the intensity of a given earthquake magnitude and distance
        #   at the volcano.
        "intensity_metric": lambda magnitude, dist: magnitude - np.log(dist),
        # estimate the amplitude of the waveform (at the volcano) from the earthquake intensity
        "amplitude_metric": lambda intensity: 10 ** (0.83 * intensity + 5.36),
        # earthquakes with intensity lower than the threshold are not considered
        "intensity_threshold": -3.0,
        # distance metric used for determining if event is an earthquake (as used in CatalogConsolidation)
        "earthquake_metric": np.array(
            [20, 0, 0, 0.01], dtype=np.double
        ),  # equivalent to: lambda y: [20/y, 0.01/y],
    },
}
