#!/usr/bin/env python3
"""TOA5 file concatenator.

Splices a master TOA5 file with one or more slave files, patching gaps in the
master with records from the slaves.  Master data always takes priority: for
any timestamp present in both master and a slave the master record is kept.
Slave files are consumed in order; the first slave to provide a record for a
missing timestamp wins.

Note on deduplication: the sort-and-dedup logic here intentionally differs
from ``infrastructure.data_conditioning.condition_dataframe``, which pads the
output to a uniform time index.  The concatenator preserves only timestamps
that appear in at least one input file — no NaN rows are inserted for gaps
that no file covers.
"""

###############################################################################
### BEGIN IMPORTS ###
###############################################################################

import pathlib

import pandas as pd

from infrastructure import data_diagnostics
from infrastructure.data_conditioning import infer_interval
from services.data import raw_data_loader, toa5_writer
from services.data.concat_common import validate_headers, validate_interval

_HEADER_LABELS = ("variable", "units", "sampling")

###############################################################################
### END IMPORTS ###
###############################################################################


###############################################################################
### BEGIN FUNCTIONS ###
###############################################################################


# -----------------------------------------------------------------------------
def _load_header(file_path: pathlib.Path) -> tuple[list, list[list]]:
    """Return (info, [variable, units, sampling]) for a TOA5 file."""
    raw = raw_data_loader.load_raw_header(file_path=file_path, file_format="TOA5")
    info = raw.pop("info")
    return info, list(raw.values())


# -----------------------------------------------------------------------------
def analyse_gap_fill(
    master: str | pathlib.Path,
    slaves: list[str | pathlib.Path],
) -> dict:
    """Report which gaps in the master file each slave would fill.

    Applies the same header/interval validation as ``concatenate_toa5`` and
    simulates its slave-priority rule (first slave to cover a missing
    timestamp wins), without writing any output.

    Args:
        master:
            Path to the authoritative TOA5 file.
        slaves:
            Paths to supplementary files to check against the master's gaps.

    Returns:
        Report dict with keys:
            'master_gaps'         — number of timestamps missing from the
                                    master's uniform time index (spanning its
                                    first to last record).
            'gap_timestamps'      — sorted list of those missing timestamps.
            'slave_fill'          — {filename: {'fills': int, 'timestamps':
                                    [...]}} for each slave that covers at
                                    least one gap, in the order slaves are
                                    consumed.
            'unfilled_gaps'       — number of gaps no slave covers.
            'unfilled_timestamps' — sorted list of those timestamps.

    Raises:
        ValueError: if any slave has incompatible headers or a different time
            step from the master.
        TypeError: if any loaded DataFrame does not have a DatetimeIndex.
    """
    master = pathlib.Path(master)
    slaves = [pathlib.Path(s) for s in slaves]

    master_headers = _load_header(master)[1]
    master_df = raw_data_loader.load_raw_data(file_path=master, file_format="TOA5")
    master_df = master_df.sort_index(kind="stable")
    master_df = master_df[~master_df.index.duplicated(keep="first")]

    interval = infer_interval(master_df)
    remaining_gaps = data_diagnostics.analyse_data_gaps(
        df=master_df, interval_minutes=interval
    )["missing_timestamps"]

    report = {
        "master_gaps": len(remaining_gaps),
        "gap_timestamps": list(remaining_gaps),
        "slave_fill": {},
    }

    for slave_path in slaves:
        slave_headers = _load_header(slave_path)[1]
        validate_headers(master_headers, slave_path, slave_headers, _HEADER_LABELS)

        slave_df = raw_data_loader.load_raw_data(
            file_path=slave_path, file_format="TOA5"
        )
        validate_interval(master_df, slave_df, slave_path)

        filled = remaining_gaps.intersection(slave_df.index)
        if len(filled):
            report["slave_fill"][slave_path.name] = {
                "fills": len(filled),
                "timestamps": list(filled),
            }
        remaining_gaps = remaining_gaps.difference(filled)

    report["unfilled_gaps"] = len(remaining_gaps)
    report["unfilled_timestamps"] = list(remaining_gaps)

    return report


# -----------------------------------------------------------------------------
def concatenate_toa5(
    master: str | pathlib.Path,
    slaves: list[str | pathlib.Path],
    output: str | pathlib.Path,
    min_timestamp: str | pd.Timestamp | None = None,
) -> dict:
    """Splice TOA5 files, patching gaps in master with data from slave files.

    Master data takes priority: for any timestamp present in both master and a
    slave the master record is kept.  Slaves are consumed in order; the first
    slave to supply a record for a missing timestamp wins.  Slave records may
    extend before or after the master's own date range (e.g. a slave that
    starts logging earlier than the master file).

    Datalogger clock resets (e.g. a program change or power loss that resets
    the clock to the CR1000X default power-on epoch of 1990-01-01) can plant
    a spurious record with a wildly wrong timestamp. There is no reliable
    automatic way to distinguish this from a genuine, correctly-timestamped
    record that happens to be isolated because of real data loss elsewhere
    in the record (the RECORD column is not a safe signal either, since it
    resets on ordinary program changes too). Pass ``min_timestamp`` to
    manually drop everything earlier than a known-good date if a slave file
    turns out to contain one of these artifacts.

    The RECORD column is preserved as-is.  At patch boundaries the record
    numbers will jump or reset — this is intentional and reflects the
    provenance of the data.

    Args:
        master:
            Path to the authoritative TOA5 file.
        slaves:
            Paths to supplementary files used to fill gaps in the master.
        output:
            Destination path for the spliced output file.
        min_timestamp:
            Optional manual floor. Records earlier than this are dropped
            from the final output.

    Returns:
        Report dict with keys:
            'master_records'     — records contributed by the master file.
            'slave_contributions'— {filename: n_records} for each slave that
                                   added at least one new record.
            'dropped_below_min'  — records dropped by ``min_timestamp``, if
                                   provided.
            'total_records'      — total records in the output file.

    Raises:
        ValueError: if any slave has incompatible headers or a different time
            step from the master.
        TypeError: if any loaded DataFrame does not have a DatetimeIndex.
    """
    master = pathlib.Path(master)
    slaves = [pathlib.Path(s) for s in slaves]

    # --- load master ---------------------------------------------------------

    master_info, master_headers = _load_header(master)
    combined = raw_data_loader.load_raw_data(file_path=master, file_format="TOA5")
    combined = combined.sort_index(kind="stable")
    combined = combined[~combined.index.duplicated(keep="first")]

    report = {"master_records": len(combined), "slave_contributions": {}}

    # --- load and merge slaves -----------------------------------------------

    for slave_path in slaves:
        _, slave_headers = _load_header(slave_path)
        validate_headers(master_headers, slave_path, slave_headers, _HEADER_LABELS)

        slave_df = raw_data_loader.load_raw_data(
            file_path=slave_path, file_format="TOA5"
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

    if min_timestamp is not None:
        min_timestamp = pd.Timestamp(min_timestamp)
        before = len(combined)
        combined = combined[combined.index >= min_timestamp]
        report["dropped_below_min"] = before - len(combined)

    report["total_records"] = len(combined)

    # --- write output --------------------------------------------------------

    toa5_writer.write_toa5(
        data=combined,
        headers=master_headers,
        file_path=output,
        info=master_info,
    )

    return report


# -----------------------------------------------------------------------------

###############################################################################
### END FUNCTIONS ###
###############################################################################
