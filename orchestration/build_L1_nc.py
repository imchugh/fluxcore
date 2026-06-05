#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 09:46:11 2026

@author: imchugh
"""

###############################################################################
### BEGIN IMPORTS ###
###############################################################################

import datetime
import gc
import pathlib
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------

from domain import time_utils
from domain.constants import SITE_PLACEHOLDER, DATA_TIME_FORMAT, NC_ENCODING
from infrastructure import file_io, paths
from orchestration import L1_constructor
from services import config_loader

###############################################################################
### END IMPORTS ###
###############################################################################


###############################################################################
### BEGIN INITS ###
###############################################################################

STD_METADATA = config_loader.load_config_file_from_name(name='nc_metadata')
CRS_METADATA = config_loader.load_config_file_from_name(name='nc_dim_attrs')

###############################################################################
### END INITS ###
###############################################################################


###############################################################################
### BEGIN FUNCTIONS ###
###############################################################################

# -----------------------------------------------------------------------------

def get_ds_years(ds):

    return np.unique(ds.time.dt.year).tolist()
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def build(
        site_name: str,
        output_dir: pathlib.Path | str | None = None,
        constrain_to_commission: bool = True,
        year: int | None = None,
        ) -> list[pathlib.Path]:
    """
    Build L1 NetCDF files for all data years at a site.

    Constructs the base L1 dataset via L1_constructor, augments it with
    NetCDF global attributes and spatial dimensions, then writes one file
    per data year to output_dir.

    Args:
        site_name: registered site name.
        output_dir: directory to write files into. Defaults to the
            homogenised_data/nc stream path for this site.
        constrain_to_commission: if True (default), skip years before the
            site's date_commissioned. Prevents pre-flux ancillary-only years
            from appearing in sites with long logger histories.
        year: if provided, only build that calendar year and discard all
            earlier records during data loading.  Use for the 30-min
            operational update cycle to avoid reading full site history.

    Returns:
        List of paths to files written.
    """

    if output_dir is None:
        output_dir = paths.get_local_stream_path('homogenised_data', 'nc') / site_name
    output_dir = pathlib.Path(output_dir)

    start_date = pd.Timestamp(year, 1, 1) if year is not None else None
    ds = L1_constructor.build_dataset_from_site_name(site_name, start_date=start_date)
    ds = build_L1_ds_complete(ds)

    commission_year = None
    if constrain_to_commission and 'date_commissioned' in ds.attrs:
        commission_year = pd.Timestamp(ds.attrs['date_commissioned']).year

    written = []
    for year in get_ds_years(ds):
        if commission_year is not None and year < commission_year:
            continue
        year_ds = build_L1_ds_by_year(ds=ds, year=year)
        file_path = output_dir / f'{site_name}_{year}_L1.nc'
        file_io.write_netcdf(ds=year_ds, file_path=file_path, time_units=NC_ENCODING)
        written.append(file_path)

    del ds
    gc.collect()

    return written
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def build_L1_ds_complete(ds):

    ds = do_dim_ops(ds=ds)
    ds = assign_crs_variable(ds=ds)
    ds = assign_L1_global_generic_attrs(ds=ds)

    return ds
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def build_L1_ds_by_year(ds, year):

    # Get network-specific valid year data bounds
    time_bounds = [
        bound.strftime(DATA_TIME_FORMAT) for bound in
        time_utils.get_data_year_bounds(
            year=year, time_step=ds.attrs['time_step']
            )
        ]

    year_ds = ds.sel(time=slice(*time_bounds))
    year_ds = assign_variable_flags(year_ds)
    year_ds = assign_L1_data_year_attrs(ds=year_ds, year=year)
    year_ds = serialize_inst_history(ds=year_ds, year=year)

    return year_ds
# -----------------------------------------------------------------------------
    
# -----------------------------------------------------------------------------

def do_dim_ops(ds):
    
    # Add spatial coordinates
    ds = (
        ds.assign_coords(
            latitude=ds.attrs['latitude'],
            longitude=ds.attrs['longitude'],
            )
        .expand_dims(['latitude', 'longitude'])
        )
    ds = ds.transpose('time', 'latitude', 'longitude')
    return ds
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def assign_crs_variable(ds):
    
    """
    Assign coordinate reference system variable.

    Args:
        ds: xarray dataset.

    Returns:
        None.

    """

    ds['crs'] = (
        ['time', 'latitude', 'longitude'],
        np.tile(np.nan, (len(ds.time), 1, 1)),
        CRS_METADATA['coordinate_reference_system']
        )
    return ds
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def assign_L1_data_year_attrs(ds, year):
    """
    Augment global attributes.

    Args:
        df: pandas dataframe containing merged data.
        md_mngr: VariableManager class used to access variable attributes.

    Returns:
        Dict containing global attributes.

    """
   
    # Conversion function for dataset start and end times
    func = lambda x: pd.to_datetime(x).strftime(DATA_TIME_FORMAT)
    begin = func(ds.time.values[0])
    end = func(ds.time.values[-1])

    # Make title string
    title_str = (
        f'Flux tower data set from the {ds.site_name} site '
        f'{year}'
        )

    # Assign attrs
    ds.attrs.update(
        {
            'title': title_str,
            'nc_nrecs': len(ds.time),
            'time_coverage_start': begin,
            'time_coverage_end': end,
            }
        )
    
    return ds
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def assign_L1_global_generic_attrs(ds):
    
    # Get and edit the generic global attribute fields
    site_metadata = STD_METADATA.copy()
    site_metadata['metadata_link'] = (
        site_metadata['metadata_link'].replace(SITE_PLACEHOLDER, ds.site_name)
        )
    
    # Add time-sensitive fields
    date = datetime.datetime.now()
    this_year = date.strftime('%Y')
    this_month = date.strftime('%b')
    site_metadata.update(
        {
            'date_created': date.strftime(DATA_TIME_FORMAT),
            'history': f'{this_month} {this_year} processing'
            }
        )
    
    # Update dataset metadata
    ds.attrs.update(site_metadata)
    
    return ds
# -----------------------------------------------------------------------------

#------------------------------------------------------------------------------

def assign_variable_flags(ds):
    """
    Assign the variable QC flags to the existing dataset.

    Args:
        ds: xarray dataset.

    Returns:
        None.

    """

    var_list = [var for var in ds.variables if not var in ds.dims]
    for var in var_list:
        ds[f'{var}_QCFlag'] = (
            ['time', 'latitude', 'longitude'],
            pd.isnull(ds[var]).astype(int),
            {'long_name': f'{var} QC flag', 'units': '1'}
            )
    return ds
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def serialize_inst_history(ds, year):

    year_start, year_end = time_utils.get_data_year_bounds(
        year=year, time_step=ds.attrs['time_step']
        )
    year_start = max(year_start, pd.Timestamp(ds.time.values[0]).to_pydatetime())
    year_end = min(year_end, pd.Timestamp(ds.time.values[-1]).to_pydatetime())

    var_list = [var for var in ds.variables if var not in ds.dims]
    for var in var_list:
        if 'instrument_history' not in ds[var].attrs:
            continue

        serialised = []
        last_inst = None
        last_end = None
        for inst, dates in ds[var].attrs['instrument_history'].items():

            dates = dict(dates)
            start = dates['start_date'] if dates['start_date'] is not None else year_start
            end = dates['end_date'] if dates['end_date'] is not None else year_end

            if end < year_start or start > year_end:
                continue

            use_start = max(year_start, start)
            use_end = min(year_end, end)
            serialised.append(
                f'{inst}: {use_start.strftime(DATA_TIME_FORMAT)}'
                f' - {use_end.strftime(DATA_TIME_FORMAT)}'
                )
            if last_end is None or use_end > last_end:
                last_end = use_end
                last_inst = inst

        if len(serialised) == 1:
            ds[var].attrs['instrument'] = last_inst
            del ds[var].attrs['instrument_history']
        else:
            ds[var].attrs['instrument_history'] = ', '.join(serialised)
            if last_inst is not None:
                ds[var].attrs['instrument'] = last_inst

    return ds
#------------------------------------------------------------------------------

###############################################################################
### END FUNCTIONS ###
###############################################################################
