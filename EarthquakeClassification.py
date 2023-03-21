"""
With `get_and_save_earthquake_catalog` the earthquake catalog needed for the earthquake classification is downloaded.
"""
import os
import numpy as np
import pandas as pd
import datetime as dt
from awesam import config


def add_earthquake_probabilities(
    names: list, eq_catalog: pd.DataFrame, max_distance: float = 2000
) -> None:
    """
    Loads catalogs, adds earthquake_probability and overwrites them.
    (Not used in CatalogPipeline.)
    """
    from awesam import CatalogPipeline
    from awesam import CatalogConsolidation

    coords = np.array([[*config.settings["general"]["coordinates"], 0]])
    distances = _distance_with_depth(
        eq_catalog[["latitude", "longitude", "depth"]].to_numpy(), coords
    )
    eq_catalog_query = eq_catalog[distances < max_distance]
    eq_catalog_localized = _earthquake_catalog_to_local(eq_catalog_query.copy())
    directory = config.settings["general"]["catalog_directory"]
    metric = config.settings["EarthquakeClassification"]["earthquake_metric"]
    window_length = config.settings["CatalogConsolidation"]["window_length"]

    for name in names:
        print(name)
        catalog = CatalogPipeline._load_catalog(
            directory, name, station="", suffix="FINAL.csv"
        )
        start, end = catalog["time"].min(), catalog["time"].max()

        eq_catalog_localized_query = eq_catalog_localized[
            (eq_catalog_localized["time"] > start - dt.timedelta(days=1))
            & (eq_catalog_localized["time"] < end)
        ]

        p = CatalogConsolidation.compute_earthquake_probabilities(
            catalog,
            eq_catalog_localized_query,
            start=start,
            metric=metric,
            window_length=window_length,
        )
        catalog["earthquake_probability"] = p

        catalog.to_csv(os.path.join(directory, f"{name}_FINAL.csv"), index=False)


def _earthquake_catalog_to_local(catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Converts an (official) earthquake catalog to a local catalog (reflecting travel time and attenuation).
    The catalog should be a pd.DataFrame with the following columns: ['latitude', 'longitude', 'depth', 'magnitude']
    It returns the same catalog (no copy) with additional columns: ['arrival_time', 'intensity', 'amplitude']

    - 'arrival_time': Estimated from the distance with config.settings['EarthquakeClassification']['travel_time_metric']
    - 'intensity': Estimated from the magnitude and the distance with config.settings['EarthquakeClassification']['intensity_metric']
    - 'amplitude': Estimated from the intensity with config.settings['EarthquakeClassification']['amplitude_metric']
    """
    coords = np.array([[*config.settings["general"]["coordinates"], 0]])
    catalog["dist"] = _distance_with_depth(
        catalog[["latitude", "longitude", "depth"]].to_numpy(), coords
    )

    catalog["arrival_time"] = catalog["time"] + pd.to_timedelta(
        config.settings["EarthquakeClassification"]["travel_time_metric"](
            catalog["dist"]
        ),
        unit="seconds",
    )
    catalog["intensity"] = config.settings["EarthquakeClassification"][
        "intensity_metric"
    ](catalog["magnitude"], catalog["dist"])
    catalog["amplitude"] = config.settings["EarthquakeClassification"][
        "amplitude_metric"
    ](catalog["intensity"])

    return catalog


def _distance_with_depth(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes distance between array of coordinates a and a single coordinate b, where a and b are coordinate vectors in the following form:
    a: [[latitude[degrees], longitude[degrees], depth[km]], ...]
    b: [[latitude[degrees], longitude[degrees], depth[km]]] or same shape as a
    Note: Depth must be given in km!
    """
    to_cartesian = lambda x: np.vstack(
        [
            x[:, 2] * np.sin(x[:, 0]) * np.cos(x[:, 1]),
            x[:, 2] * np.sin(x[:, 0]) * np.sin(x[:, 1]),
            x[:, 2] * np.cos(x[:, 0]),
        ]
    )

    to_spherical = lambda a: a * [-1 / 180 * np.pi, 1 / 180 * np.pi, -1] + [
        np.pi / 2,
        0,
        6371.009,
    ]  # km (earth radius)
    a = to_cartesian(to_spherical(a.astype(np.double))).T
    b = to_cartesian(to_spherical(b.astype(np.double))).T
    return np.linalg.norm(a - b, axis=1)
