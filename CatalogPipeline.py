"""
Allows the execution of the catalog creation pipeline, including:
- event detection (for all stations)
- gap complementation
- catalog consolidation
- earthquake classification
"""

from awesam import EarthquakeClassification, EventCatalog, CatalogConsolidation
from awesam import config, util
import datetime as dt
import pandas as pd
import multiprocessing
import traceback
import logging
import os


def execute_catalog_pipeline(
    name: str,
    start: dt.datetime,
    end: dt.datetime,
    csv_directory: str = None,
    checkpoint: bool = False,
) -> None:
    """
    Note: Use multiprocessing_pipeline instead of this function. If multiprocessing is unwanted,
    you can disable it with use_multiprocessing=False.

    The CatalogPipeline includes:
    - Creation of event- and gap- catalogs for all stations.
    - Catalog consolidation (transfer gaps, event-probability)
    - Earthquake classification (earthquake-probability)
    - Saving final catalogs.

    The seismic data must alredy exist in the mseed_directory.

    - `name`: Name used to identify catalog
    - `start`: start day
    - `end`: end day (inclusive)
    - `csv_directory`: output directory of catalogs
    - `checkpoint`: While creating the catalog, checkpoint files are generated,
    to return to the current progress after interruption. If `True` these files are used.
    """
    try:

        if csv_directory is None:
            csv_directory = config.settings["general"]["catalog_directory"]

        # Logging
        logger = _get_logger()
        logger.name = f"pipeline.{start.strftime('%Y-%m-%d')}"
        logger.info(
            f'Starting CatalogPipeline from {start.strftime("%Y-%m-%d")} to {end.strftime("%Y-%m-%d")}'
        )

        # get station codes from settings
        station_seeds = util.get_station_seeds(config.settings["general"]["stations"])

        # Create catalog (catalogs are generated and saved in csv_directory)
        # besides the event_catalog, a gap_catalog is also created.
        for station_seed in station_seeds:
            EventCatalog.create_catalog(
                start,
                end,
                os.path.join(
                    config.settings["general"]["mseed_directory"],
                    util.get_station_name_from_seed(station_seed).lower(),
                    "mseed",
                ),
                csv_directory,
                name,
                station_seed,
                checkpoint=checkpoint,
            )
            logger.info(f"Finished generating EventCatalog for {station_seed}.")

        logger.info("Finished generating EventCatalog for all stations.")

        # Load catalogs for all stations
        catalogs, gaps = {}, {}
        for station_seed in station_seeds:
            station_name = util.get_station_name_from_seed(station_seed)
            catalogs[station_seed] = _load_catalog(
                csv_directory, name, station_name.lower()
            )
            gaps[station_seed] = _load_gaps_catalog(
                csv_directory, name, station_name.lower()
            )

        # Catalog Consolidation
        if config.settings["general"]["pipeline"]["use_catalog_consolidation"]:
            if len(station_seeds) < 2:
                raise ValueError("Two stations are needed for catalog consolidation")

            # defining station identifiers
            principal_station = station_seeds[0]
            complementary_station = station_seeds[1]

            logger.info(
                f"Start CatalogConsolidation with data from {principal_station} and {complementary_station}"
            )

            catalog_final = CatalogConsolidation.consolidate_catalog(
                catalogs[principal_station],
                catalogs[complementary_station],
                gaps[principal_station],
                gaps[complementary_station],
                util.get_channels(principal_station),
                util.get_channels(complementary_station),
                config.settings["CatalogConsolidation"]["default_metric"],
                config.settings["CatalogConsolidation"]["window_length"],
                start,
            )
        else:
            catalog_final = catalogs[list(catalogs)[0]]

        # Earthquake Classification
        if config.settings["general"]["pipeline"]["use_earthquake_classification"]:
            catalog_earthquake = _load_earthquake_catalog(
                config.settings["general"]["earthquake_catalog_directory"], start, end
            )
            catalog_final[
                "earthquake_probability"
            ] = CatalogConsolidation.compute_earthquake_probabilities(
                catalog_final,
                catalog_earthquake,
                start,
                config.settings["EarthquakeClassification"]["earthquake_metric"],
                config.settings["CatalogConsolidation"]["window_length"],
            )

        # Saving
        _save_catalog(catalog_final, name, csv_directory)
        logger.info(
            f'Finished from {start.strftime("%Y-%m-%d")} to {end.strftime("%Y-%m-%d")}'
        )
    except Exception as e:
        print(traceback.format_exc())
        raise e


def multiprocessing_pipeline(
    name: str,
    start: dt.datetime,
    end: dt.datetime,
    checkpoint: bool = False,
    use_multiprocessing=True,
) -> None:
    """
    CatalogPipeline with multiprocessing.
    See CatalogPipeline.execute_catalog_pipeline for details.
    """
    util.check_settings_integrity()

    csv_directory = config.settings["general"]["catalog_directory"]
    chunk_length = dt.timedelta(days=max(4, min(50, (end - start).days // 10)))
    chunks = pd.date_range(start, end, freq=chunk_length).to_pydatetime()

    # make directory for temporary files of individual processes (if not exists)
    if not os.path.isdir(os.path.join(csv_directory, "tmp")):
        os.mkdir(os.path.join(csv_directory, "tmp"))

    # compile parameter list
    parameter_list = []
    for i, (date_start) in enumerate(chunks):

        if date_start == chunks[-1]:  # if last chunk
            date_end = end
        else:
            date_end = date_start + chunk_length - dt.timedelta(days=1)

        parameter_list.append(
            [
                f"{name}_{i}",
                date_start,
                date_end,
                os.path.join(csv_directory, "tmp"),
                checkpoint,
            ]
        )

    # start catalog pipeline
    if use_multiprocessing:
        p = multiprocessing.Pool(32)
        p.starmap(execute_catalog_pipeline, parameter_list)
        p.close()
        p.join()
    else:
        for i, params in enumerate(parameter_list):
            execute_catalog_pipeline(*params)

    # merge and delete all files that were created temporarily.
    _join_temporary_files(name, start, end, csv_directory)


def _join_temporary_files(
    name: str, start: dt.datetime, end: dt.datetime, csv_directory: str
) -> None:
    """
    When using multiprocessing_pipeline, temporary files are generated.
    This method merges everything into one catalog.
    """
    station_names = [
        s.lower() + "_events"
        for s in util.get_station_names(config.settings["general"]["stations"])
    ]

    # get and filter files in the tmp directory by `name`
    files = os.listdir(os.path.join(csv_directory, "tmp"))
    files = [file for file in files if file.find(name) != -1]

    # compose catalog for each station and FINAL
    for catalog_type in ["FINAL", *station_names]:
        final_files = [f for f in files if f.find(f"{catalog_type}.csv") != -1]
        final_catalogs = [
            pd.read_csv(os.path.join(os.path.join(csv_directory, "tmp"), f))
            for f in final_files
        ]
        final_df = pd.concat(final_catalogs)
        final_df.sort_values("time", inplace=True)
        final_df.to_csv(
            os.path.join(csv_directory, f"{name}_{catalog_type}.csv"), index=False
        )

    # gap catalog (using all stations)
    gap_catalogs = []
    for station in util.get_station_names(config.settings["general"]["stations"]):
        # filter by station and gap-catalog
        station_files = [
            file
            for file in files
            if (file.find(station.lower()) != -1) and (file.find("gaps.csv") != -1)
        ]
        for f in station_files:
            c = pd.read_csv(os.path.join(os.path.join(csv_directory, "tmp"), f))
            c["station"] = station
            gap_catalogs.append(c)
    gap_df = pd.concat(gap_catalogs)
    gap_df.to_csv(os.path.join(csv_directory, f"{name}_gaps.csv"), index=False)

    # delete temporary files
    for f in files:
        os.remove(os.path.join(csv_directory, "tmp", f))


def _get_logger():
    logger = logging.getLogger("Pipeline")
    if not len(logger.handlers):
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
    return logger


def _load_catalog(
    csv_directory: str, name: str, station: str, suffix="_events.csv"
) -> pd.DataFrame:
    """
    Loads the event-catalog.
    """
    df = pd.read_csv(os.path.join(csv_directory, name + "_" + station + suffix))
    df["time"] = pd.to_datetime(df["time"])
    return df


def _load_gaps_catalog(csv_directory: str, name: str, station: str) -> pd.DataFrame:
    """
    Loads the gap-catalog.
    """
    df = pd.read_csv(os.path.join(csv_directory, name + "_" + station + "_gaps.csv"))
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])

    return df


def _load_earthquake_catalog(
    directory: str, start: dt.datetime, end: dt.datetime, return_full: bool = False
) -> pd.DataFrame:
    """
    Reads the specified earthquake catalogs.
    Also computes the intensity for each earthquake and only returns earthquakes over the threshold.
    The catalog is localized for the volcano.
    """
    # load all necessary years
    eq_catalog = []
    for year in range(start.year, end.year + 1):
        try:
            c = pd.read_csv(os.path.join(directory, str(year) + ".csv"))
        except FileNotFoundError:
            raise FileNotFoundError(f"Earthquake Catalog for {year} does not exist.")
        else:
            if "date" in c.columns:
                c.rename(columns={"date": "time"}, inplace=True)
            eq_catalog.append(c)

    eq_catalog = pd.concat(eq_catalog)
    eq_catalog["time"] = pd.to_datetime(eq_catalog["time"])

    # filter catalog
    eq_catalog = eq_catalog[(eq_catalog["time"] > start) & (eq_catalog["time"] < end)]

    # localize catalog
    eq_catalog = EarthquakeClassification._earthquake_catalog_to_local(eq_catalog)
    eq_catalog = eq_catalog[
        eq_catalog["intensity"]
        > config.settings["EarthquakeClassification"]["intensity_threshold"]
    ]

    if return_full:
        return eq_catalog
    else:
        return eq_catalog[["arrival_time", "amplitude"]]


def _save_catalog(catalog: pd.DataFrame, name: str, directory: str) -> None:
    """
    Writes the final catalog to disk, with the suffix '_FINAL'.
    """
    catalog.to_csv(os.path.join(directory, name + "_FINAL.csv"), index=False)
