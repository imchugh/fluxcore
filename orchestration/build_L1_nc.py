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

    Returns:
        List of paths to files written.
    """

    if output_dir is None:
        output_dir = paths.get_local_stream_path('homogenised_data', 'nc') / site_name
    output_dir = pathlib.Path(output_dir)

    ds = L1_constructor.build_dataset_from_site_name(site_name)
    ds = build_L1_ds_complete(ds)

    written = []
    for year in get_ds_years(ds):
        year_ds = build_L1_ds_by_year(ds=ds, year=year)
        file_path = output_dir / f'{site_name}_{year}_L1.nc'
        file_io.write_netcdf(ds=year_ds, file_path=file_path, time_units=NC_ENCODING)
        written.append(file_path)

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
    
    # Subset the dataset
    year_ds = ds.sel(time=slice(*time_bounds))
    
    # Add time-based attrs
    ds = assign_L1_data_year_attrs(ds=year_ds, year=year)
    
    return ds        
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

###############################################################################
### END FUNCTIONS ###
###############################################################################
