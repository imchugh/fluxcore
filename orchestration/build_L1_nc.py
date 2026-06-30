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

from domain.constants import SITE_PLACEHOLDER, DATA_TIME_FORMAT, NC_ENCODING
from infrastructure import file_io, paths
from orchestration import dataset_builder
from services import config_loader

###############################################################################
### END IMPORTS ###
###############################################################################


###############################################################################
### BEGIN INITS ###
###############################################################################

STD_METADATA = config_loader.load_config_file_from_name(name='nc_metadata')
CRS_METADATA = config_loader.load_config_file_from_name(name='nc_dim_attrs')

VARIABLE_NC_ATTRS = {
    'units', 'long_name', 'standard_name',
    'height', 'height_range', 'instrument', 'instrument_history', 
    'instrument_uri', 'statistic_type',
    }

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
        year: int | None = None,
        ) -> list[pathlib.Path]:
    """
    Build L1 NetCDF files for all data years at a site.

    Constructs the base L1 dataset via L1_constructor, augments it with
    NetCDF global attributes and spatial dimensions, then writes one file
    per data year to output_dir.

    The dataset is truncated to the temporal extent of the flux file group,
    so ancillary data (e.g. soil loggers) predating the flux system does not
    produce empty output years.

    Args:
        site_name: registered site name.
        output_dir: directory to write files into. Defaults to the
            homogenised_data/nc stream path for this site.
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
    ds = dataset_builder.build_dataset_from_site_name(site_name, start_date=start_date)
    ds = build_L1_ds_complete(ds)

    written = []
    for ds_year in get_ds_years(ds):
        year_ds = build_L1_ds_by_year(ds=ds, year=ds_year)
        file_path = output_dir / f'{site_name}_{ds_year}_L1.nc'
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
    time_step = ds.attrs['time_step']
    time_bounds = [
        bound.strftime(DATA_TIME_FORMAT) for bound in (
            datetime.datetime(year, 1, 1) + datetime.timedelta(minutes=time_step),
            datetime.datetime(year + 1, 1, 1),
        )
    ]

    year_ds = ds.sel(time=slice(*time_bounds))
    year_ds = assign_variable_flags(year_ds)
    year_ds = assign_L1_data_year_attrs(ds=year_ds, year=year)
    year_ds = filter_variable_attrs(ds=year_ds)
    year_ds = serialize_uri(ds=year_ds)
    year_ds = serialize_inst_history(ds=year_ds, year=year)
    year_ds = serialize_units(ds=year_ds)

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

    ds['crs'] = ([], np.int32(0), CRS_METADATA['coordinate_reference_system'])
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

    var_list = [var for var in ds.variables if var not in ds.dims and var != 'crs']
    for var in var_list:
        ds[f'{var}_QCFlag'] = (
            ['time', 'latitude', 'longitude'],
            pd.isnull(ds[var]).astype(int),
            {'long_name': f'{var} QC flag', 'units': '1'}
            )
    return ds
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def filter_variable_attrs(ds):

    for var in ds.variables:
        if var == 'crs':
            continue
        ds[var].attrs = {
            k: v for k, v in ds[var].attrs.items()
            if k in VARIABLE_NC_ATTRS
            }

    return ds


def serialize_units(ds):

    for var in ds.variables:
        if ds[var].attrs.get('units') == 'dimensionless':
            ds[var].attrs['units'] = '1'

    return ds

def serialize_uri(ds):
    
    var_list = [var for var in ds.variables if var not in ds.dims]
    for var in var_list:
        attrs = ds[var].attrs
        
        if isinstance(attrs.get('instrument_uri'), dict):
            attrs['instrument_uri'] = ','.join(
                f'{uri}' for uri in attrs['instrument_uri'].values()
                )
        
    return ds

def serialize_inst_history(ds, year):

    time_step = ds.attrs['time_step']
    year_start = datetime.datetime(year, 1, 1) + datetime.timedelta(minutes=time_step)
    year_end = datetime.datetime(year + 1, 1, 1)
    year_start = max(year_start, pd.Timestamp(ds.time.values[0]).to_pydatetime())
    year_end = min(year_end, pd.Timestamp(ds.time.values[-1]).to_pydatetime())

    var_list = [var for var in ds.variables if var not in ds.dims]
    for var in var_list:
        attrs = ds[var].attrs

        if 'instrument_history' not in attrs:
            if isinstance(attrs.get('instrument'), dict):
                attrs['instrument'] = ','.join(attrs['instrument'].values())
            continue

        history = attrs['instrument_history']
        first_val = next(iter(history.values()))

        if 'start_date' in first_val:
            # Simple: {inst_name: {start_date, end_date}}
            if isinstance(attrs.get('instrument'), dict):
                attrs['instrument'] = ','.join(attrs['instrument'].values())
            serialised, last_inst = _serialise_simple_history(
                history, year_start, year_end
                )
        else:
            # Compound: {alias: {inst_name: {start_date, end_date}}}
            serialised, last_inst = _serialise_compound_history(
                history, year_start, year_end
                )

        if not serialised:
            del attrs['instrument_history']
        elif 'start_date' in first_val:
            # Simple history
            if len(serialised) == 1:
                attrs['instrument'] = last_inst
                del attrs['instrument_history']
            else:
                attrs['instrument_history'] = '|'.join(serialised)
                if last_inst is not None:
                    attrs['instrument'] = last_inst
        else:
            # Compound history: last_inst is a dict {alias: name}
            attrs['instrument_history'] = ';'.join(serialised)
            inst = attrs.get('instrument')
            current = dict(inst) if isinstance(inst, dict) else {}
            if last_inst:
                current.update(last_inst)
            attrs['instrument'] = ','.join(current.values())

    return ds


def _serialise_simple_history(
        history: dict,
        year_start,
        year_end,
        ) -> tuple[list[str], str | None]:
    """Serialise a simple instrument history to a list of formatted strings."""

    serialised = []
    last_inst = None
    last_end = None
    for inst, dates in history.items():
        dates = dict(dates)
        start = dates['start_date'] if dates['start_date'] is not None else year_start
        end = dates['end_date'] if dates['end_date'] is not None else year_end
        if end < year_start or start > year_end:
            continue
        use_start = max(year_start, start)
        use_end = min(year_end, end)
        serialised.append(f'({inst},{use_start.isoformat()},{use_end.isoformat()})')
        if last_end is None or use_end > last_end:
            last_end = use_end
            last_inst = inst
    return serialised, last_inst


def _serialise_compound_history(
        history: dict,
        year_start,
        year_end,
        ) -> tuple[list[str], dict[str, str]]:
    """Serialise a compound instrument history keyed by alias.

    Produces one segment per alias: alias>(inst,start,end)|(inst,start,end)
    Segments joined with ';' by the caller. Returns last_by_alias as a
    {alias: instrument_name} dict for the most recently used instruments.
    """

    alias_segments = []
    last_by_alias: dict[str, str] = {}
    for alias, inst_history in history.items():
        parts, last_inst = _serialise_simple_history(inst_history, year_start, year_end)
        if parts:
            alias_segments.append(f'{alias}>{"|".join(parts)}')
            if last_inst is not None:
                last_by_alias[alias] = last_inst

    return alias_segments, last_by_alias
#------------------------------------------------------------------------------

###############################################################################
### END FUNCTIONS ###
###############################################################################
