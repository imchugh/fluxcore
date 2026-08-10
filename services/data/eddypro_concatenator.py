#!/usr/bin/env python3
"""EddyPro file concatenator.

Splices a master EddyPro (LI-COR SmartFlux) file with one or more slave
files, patching gaps in the master with records from the slaves. Master
data always takes priority: for any timestamp present in both master and a
slave the master record is kept. Slave files are consumed in order; the
first slave to provide a record for a missing timestamp wins.

Mirrors ``toa5_concatenator.py``, adapted for EddyPro's two-row header
(variable, units — no sampling row) and the fact that EddyPro's
DATAH/filename/date/time columns survive loading as ordinary data columns
rather than being collapsed into the DatetimeIndex.

Note on deduplication: as in ``toa5_concatenator``, the sort-and-dedup
logic here intentionally differs from
``infrastructure.data_conditioning.condition_dataframe``, which pads the
output to a uniform time index. The concatenator preserves only timestamps
that appear in at least one input file — no NaN rows are inserted for gaps
that no file covers.
"""

###############################################################################
### BEGIN IMPORTS ###
###############################################################################

import os
import pathlib

import pandas as pd

from services.data import eddypro_writer, raw_data_loader
from services.data.concat_common import validate_headers, validate_interval

_HEADER_LABELS = ("variable", "units")

###############################################################################
### END IMPORTS ###
###############################################################################


###############################################################################
### BEGIN CONSTANTS ###
###############################################################################

# Bytes read from the end of the master file to recover its last record's
# timestamp without loading the whole file. Real files average ~1.5 KB/line;
# this is a wide safety margin for a single data line.
_TAIL_PEEK_BYTES = 16384

###############################################################################
### END CONSTANTS ###
###############################################################################


###############################################################################
### BEGIN FUNCTIONS ###
###############################################################################


# -----------------------------------------------------------------------------
def _load_header(file_path: pathlib.Path) -> list[list]:
    """Return [variable, units] for an EddyPro file."""
    raw = raw_data_loader.load_raw_header(file_path=file_path, file_format="EddyPro")
    return [raw["variable"], raw["units"]]


# -----------------------------------------------------------------------------
def _peek_last_timestamp(file_path: pathlib.Path) -> pd.Timestamp:
    """Read the last data row's timestamp without loading the full file.

    Relies on the concatenator always writing the master pre-sorted, so the
    last non-empty line is the most recent record. Assumes the fixed
    DATAH/filename/date/time column layout (date at field index 2, time at
    field index 3).
    """
    with open(file_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - _TAIL_PEEK_BYTES))
        tail = f.read().decode(errors="ignore")

    lines = [line for line in tail.splitlines() if line.strip()]
    fields = lines[-1].split("\t")
    return pd.Timestamp(f"{fields[2]} {fields[3]}")


# -----------------------------------------------------------------------------
def select_new_slaves(
    master: str | pathlib.Path,
    candidates: list[str | pathlib.Path],
) -> list[pathlib.Path]:
    """Filter candidate slave files down to those that might contain new data.

    Cheap pre-filter for the common case of individually date-named daily
    slave files (e.g. SmartFlux summary files: `YYYY-MM-DD_*.txt`) accumulating
    against a master that already covers most of them. Reads only the
    master's last timestamp (a tail-read, not a full parse — see
    ``_peek_last_timestamp``) and keeps any candidate whose filename-encoded
    date is on or after that day.

    This is purely a performance optimisation, not a correctness
    requirement: ``concatenate_eddypro`` already discards any record whose
    timestamp collides with an existing master record, so passing extra
    (already-covered) files through unfiltered is always safe — just wasted
    work, since each additional slave costs a full sort/dedup pass over the
    combined frame. A candidate whose filename doesn't start with a
    parseable `YYYY-MM-DD` date is kept rather than silently dropped.

    Args:
        master: path to the master file (must already exist).
        candidates: candidate slave file paths to filter.

    Returns:
        Subset of `candidates` (as Path objects) that could contain records
        newer than the master's last one.
    """
    master = pathlib.Path(master)
    master_date = _peek_last_timestamp(master).normalize()

    kept = []
    for candidate in candidates:
        candidate = pathlib.Path(candidate)
        try:
            candidate_date = pd.Timestamp(candidate.name[:10])
        except ValueError:
            kept.append(candidate)
            continue
        if candidate_date >= master_date:
            kept.append(candidate)
    return kept


# -----------------------------------------------------------------------------
def concatenate_eddypro(
    master: str | pathlib.Path,
    slaves: list[str | pathlib.Path],
    output: str | pathlib.Path,
) -> dict:
    """Splice EddyPro files, patching gaps in master with data from slaves.

    Master data takes priority: for any timestamp present in both master and
    a slave the master record is kept. Slaves are consumed in order; the
    first slave to supply a record for a missing timestamp wins.

    Args:
        master:
            Path to the authoritative EddyPro file.
        slaves:
            Paths to supplementary (e.g. daily SmartFlux summary) files
            used to fill gaps in the master.
        output:
            Destination path for the spliced output file.

    Returns:
        Report dict with keys:
            'master_records'     — records contributed by the master file.
            'slave_contributions'— {filename: n_records} for each slave
                                   that added at least one new record.
            'total_records'      — total records in the output file.

    Raises:
        ValueError: if any slave has incompatible headers or a different
            time step from the master.
        TypeError: if any loaded DataFrame does not have a DatetimeIndex.
    """
    master = pathlib.Path(master)
    slaves = [pathlib.Path(s) for s in slaves]

    # --- load master -----------------------------------------------------

    master_headers = _load_header(master)
    combined = raw_data_loader.load_raw_data(file_path=master, file_format="EddyPro")
    combined = combined.sort_index(kind="stable")
    combined = combined[~combined.index.duplicated(keep="first")]

    report = {"master_records": len(combined), "slave_contributions": {}}

    # --- load and merge slaves ---------------------------------------------

    for slave_path in slaves:
        slave_headers = _load_header(slave_path)
        validate_headers(master_headers, slave_path, slave_headers, _HEADER_LABELS)

        slave_df = raw_data_loader.load_raw_data(
            file_path=slave_path, file_format="EddyPro"
        )
        validate_interval(combined, slave_df, slave_path)

        before = len(combined)
        combined = (
            pd.concat([combined, slave_df])
            .sort_index(kind="stable")
            .loc[lambda df: ~df.index.duplicated(keep="first")]
        )
        added = len(combined) - before
        if added:
            report["slave_contributions"][slave_path.name] = added

    report["total_records"] = len(combined)

    # --- write output ------------------------------------------------------

    eddypro_writer.write_eddypro(
        data=combined,
        headers=master_headers,
        file_path=output,
    )

    return report


# -----------------------------------------------------------------------------

###############################################################################
### END FUNCTIONS ###
###############################################################################
