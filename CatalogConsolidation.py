"""
Computes a probability for each event in a principal catalog, that this event's source is volcanic.
The principal and complementary catalog are generated independently at two stations.
Core functions: `execute_catalog_consolidation`
"""
import numpy as np
import pandas as pd
import datetime as dt
from awesam import util
from awesam import awesamlib


class Gaps:
    """
    Collection of functions that perform the gap transfer step.
    Core Functions: _transfer_gaps, create_simultaneous_gap_catalog
    """

    @staticmethod
    def create_simultaneous_gap_catalog(
        gap_catalog: pd.DataFrame, channels: list
    ) -> np.ndarray:
        """Generates a list that contains only the time intervals where all channels are missing"""
        gap_catalog_melted = Gaps._melt_gap_catalog(gap_catalog)
        gap_catalog_melted = Gaps._merge_adjacent_gaps(gap_catalog_melted, channels)
        gaps = Gaps._simultaneous_gaps(gap_catalog_melted, channels)

        return Gaps._melt_back_catalog(gaps)

    @staticmethod
    def complement_gaps(
        pc: pd.DataFrame, cc: pd.DataFrame, pc_gaps: np.ndarray, cc_gaps: np.ndarray
    ):
        """copies events, that lie in a gap, from the other catalog."""
        pc = Gaps._transfer_gaps(pc, cc, pc_gaps)
        cc = Gaps._transfer_gaps(cc, pc, cc_gaps)
        return pc, cc

    @staticmethod
    def set_gaps_probabilities(
        catalog: pd.DataFrame, pc_gaps: np.ndarray, cc_gaps: np.ndarray, filler=np.nan
    ) -> pd.DataFrame:
        """event_probabilities for events during gaps are set to np.nan"""
        for gap_catalog in [pc_gaps, cc_gaps]:
            for gap in gap_catalog:
                catalog.loc[
                    (catalog["time"] > gap[0]) & (catalog["time"] < gap[1]),
                    "event_probability",
                ] = filler
        return catalog

    @staticmethod
    def _transfer_gaps(
        pc: pd.DataFrame, cc: pd.DataFrame, gaps: np.ndarray
    ) -> pd.DataFrame:
        """
        Uses the gap-catalog to transfer the corresponding peaks in the complementary catalog to the principal catalog.

        Parameters:
        - pc: Catalog that is completed
        - cc: Catalog used to complete pc
        - pc_gaps: Gaps in pc

        Returns:
        - new pc catalog
        """
        if len(gaps) != 0:
            events = []
            for i in range(len(gaps)):
                ev = cc[(cc["time"] > gaps[i, 0]) & (cc["time"] < gaps[i, 1])]

                if len(ev) > 0:
                    events.append(ev)

            if len(events) > 0:
                print(f"Transfer of {sum([len(c) for c in events])} gaps.")
                pc = pd.concat([pc] + events).sort_values("time")

                return pc
        return pc

    @staticmethod
    def _melt_gap_catalog(gap_catalog: pd.DataFrame) -> pd.DataFrame:
        """converts the gap catalog to a different repesentation, where each gap has two entries (start and end)"""
        gap_catalog["index"] = gap_catalog.index
        # melts the catalog so each start and end has its own row
        gap_catalog_melted = gap_catalog.melt(
            value_vars=["start", "end"],
            id_vars=["index", "channel"],
            value_name="time",
            var_name="start",
        ).sort_values(["time", "index"])
        # start column: 1 marks the beginning of a gap, 0 marks the end of a gap
        gap_catalog_melted["start"] = gap_catalog_melted["start"].map(
            lambda x: 1 if x == "start" else 0
        )
        gap_catalog_melted.reset_index(inplace=True, drop=True)
        return gap_catalog_melted

    @staticmethod
    def _merge_adjacent_gaps(gap_catalog: pd.DataFrame, channels) -> pd.DataFrame:
        """merges gaps that have the same end and start time"""
        gap_catalog = gap_catalog.copy()
        gap_catalog.reset_index(inplace=True, drop=True)
        for channel in channels:
            c = gap_catalog[gap_catalog["channel"] == channel].copy()
            c.sort_values("time", inplace=True)
            t = c["time"]

            for i in range(len(c) - 1):
                if t.iloc[i] == t.iloc[i + 1]:
                    gap_catalog.drop([c.iloc[i].name, c.iloc[i + 1].name], inplace=True)
        gap_catalog.reset_index(inplace=True, drop=True)
        return gap_catalog

    @staticmethod
    def _simultaneous_gaps(catalog: pd.DataFrame, channels: list) -> list:
        """returns only gaps in catalog, where all streams are missing at the same time"""

        # initialize dict, for example: {'HHE': None, 'HHN': None, 'HHZ': None}
        gaps_started = dict(zip(channels, [0] * len(channels)))
        new_catalog = []
        waiting_for_gap_end = False

        for i in range(len(catalog)):
            gap = catalog.iloc[i]

            gaps_started[gap["channel"]] = gap["start"]

            if not waiting_for_gap_end:
                if sum(gaps_started.values()) == len(channels):  # if all channels
                    new_catalog.append([gap["time"], 1])
                    waiting_for_gap_end = True
            else:
                if sum(gaps_started.values()) == len(channels):
                    raise ValueError(
                        "There is some odd problem with gaps in this catalogs."
                    )
                new_catalog.append([gap["time"], 0])
                waiting_for_gap_end = False

        return new_catalog

    @staticmethod
    def _melt_back_catalog(lst: list) -> np.ndarray:
        """converts gap catalog back to usual representation"""
        if len(lst) % 2:
            raise ValueError("Length of list is uneven")

        new_catalog = []
        for i in range(0, len(lst), 2):
            if lst[i][1] != 1 or lst[i + 1][0] != 0:
                ValueError("Gap start/end error")
            new_catalog.append([lst[i][0], lst[i + 1][0]])
        return np.array(new_catalog)


def consolidate_catalog(
    principal_catalog: pd.DataFrame,
    complementary_catalog: pd.DataFrame,
    principal_catalog_gaps: pd.DataFrame,
    complementary_catalog_gaps: pd.DataFrame,
    principal_channels: list,
    complementary_channels: list,
    metric,
    window_length: int,
    start: dt.datetime,
    gap_filler=np.nan,
) -> pd.DataFrame:
    """
    Transfer gaps and compute event & earthquake (optional) probability for catalog_ist3.
    Returns the main catalog with complemented gaps and probabilities.

    The goal of this algorithm is to identify local disturbances and earthquakes in a volcanic event catalog.
    Uses a consolication catalog to fill gaps and resolve conflicts in the principal catalog.
    Both catalogs have to span over the same time span.

    Returns the principal catalog with the probabilities.
    """

    # prepare gap catalogs (only gap where all channels are missing)
    principal_catalog_gaps_np = Gaps.create_simultaneous_gap_catalog(
        principal_catalog_gaps, principal_channels
    )
    complementary_catalog_gaps_np = Gaps.create_simultaneous_gap_catalog(
        complementary_catalog_gaps, complementary_channels
    )

    # complement gaps
    principal_catalog, complementary_catalog = Gaps.complement_gaps(
        principal_catalog,
        complementary_catalog,
        principal_catalog_gaps_np,
        complementary_catalog_gaps_np,
    )

    # catalogs to np
    principal_catalog_np = np.array(
        [
            util.time_to_index(principal_catalog["time"], start, 100),
            principal_catalog["amplitude"],
        ]
    ).T
    complementary_catalog_np = np.array(
        [
            util.time_to_index(complementary_catalog["time"], start, 100),
            complementary_catalog["amplitude"],
        ]
    ).T

    # Compute event probability
    principal_catalog["event_probability"] = awesamlib.compute_probabilities(
        principal_catalog_np,
        complementary_catalog_np,
        metric=metric,
        window_length=window_length,
    )

    # set probabilities during gaps to nan and append to principal_catalog
    principal_catalog = Gaps.set_gaps_probabilities(
        principal_catalog,
        principal_catalog_gaps_np,
        complementary_catalog_gaps_np,
        filler=gap_filler,
    )

    return principal_catalog


def compute_earthquake_probabilities(
    catalog: pd.DataFrame,
    earthquake_catalog: pd.DataFrame,
    start: dt.datetime,
    metric,
    window_length: int,
) -> np.ndarray:
    """
    Using CatalogConsolidation to get an earthquake probability
    """
    # convert to numpy array
    catalog_np = np.array(
        [util.time_to_index(catalog["time"], start, 100), catalog["amplitude"]]
    ).T
    earthquake_catalog_np = np.array(
        [
            util.time_to_index(earthquake_catalog["arrival_time"], start, 100),
            earthquake_catalog["amplitude"],
        ]
    ).T

    # compute probabilities
    earthquake_probabilities = awesamlib.compute_probabilities(
        catalog_np, earthquake_catalog_np, metric=metric, window_length=window_length
    )

    return earthquake_probabilities
