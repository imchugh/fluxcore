#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TOA5 file concatenator.

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

from infrastructure.data_conditioning import infer_interval
from services.data import raw_data_loader, toa5_writer

###############################################################################
### END IMPORTS ###
###############################################################################


###############################################################################
### BEGIN FUNCTIONS ###
###############################################################################

# -----------------------------------------------------------------------------
def _load_header(file_path: pathlib.Path) -> tuple[list, list[list]]:
    """Return (info, [variable, units, sampling]) for a TOA5 file."""

    raw = raw_data_loader.load_raw_header(
        file_path=file_path, file_format='TOA5'
    )
    info = raw.pop('info')
    return info, list(raw.values())


# -----------------------------------------------------------------------------
def _validate_headers(
    master_headers: list[list],
    slave_path: pathlib.Path,
    slave_headers: list[list],
) -> None:
    """Raise ValueError if slave headers are incompatible with master."""

    labels = ('variable', 'units', 'sampling')
    for label, master_row, slave_row in zip(labels, master_headers, slave_headers):
        if master_row != slave_row:
            raise ValueError(
                f'{slave_path.name}: {label} header does not match master.\n'
                f'  master : {master_row}\n'
                f'  slave  : {slave_row}'
            )


# -----------------------------------------------------------------------------
def _validate_interval(
    master_df: pd.DataFrame,
    slave_df: pd.DataFrame,
    slave_path: pathlib.Path,
) -> None:
    """Raise ValueError if slave time step differs from master."""

    master_interval = infer_interval(master_df)
    slave_interval = infer_interval(slave_df)
    if master_interval != slave_interval:
        raise ValueError(
            f'{slave_path.name}: time step ({slave_interval} min) does not '
            f'match master ({master_interval} min).'
        )


# -----------------------------------------------------------------------------
def concatenate_toa5(
    master: str | pathlib.Path,
    slaves: list[str | pathlib.Path],
    output: str | pathlib.Path,
) -> dict:
    """Splice TOA5 files, patching gaps in master with data from slave files.

    Master data takes priority: for any timestamp present in both master and a
    slave the master record is kept.  Slaves are consumed in order; the first
    slave to supply a record for a missing timestamp wins.

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

    Returns:
        Report dict with keys:
            'master_records'     — records contributed by the master file.
            'slave_contributions'— {filename: n_records} for each slave that
                                   added at least one new record.
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
    combined = raw_data_loader.load_raw_data(
        file_path=master, file_format='TOA5'
    )
    combined = combined.sort_index(kind='stable')
    combined = combined[~combined.index.duplicated(keep='first')]

    report = {'master_records': len(combined), 'slave_contributions': {}}

    # --- load and merge slaves -----------------------------------------------

    for slave_path in slaves:
        _, slave_headers = _load_header(slave_path)
        _validate_headers(master_headers, slave_path, slave_headers)

        slave_df = raw_data_loader.load_raw_data(
            file_path=slave_path, file_format='TOA5'
            )
        _validate_interval(combined, slave_df, slave_path)

        before = len(combined)
        combined = (
            pd.concat([combined, slave_df])
            .sort_index(kind='stable')
            .loc[lambda df: ~df.index.duplicated(keep='first')]
            )
        added = len(combined) - before
        if added:
            report['slave_contributions'][slave_path.name] = added

    report['total_records'] = len(combined)

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
