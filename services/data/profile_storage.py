#!/usr/bin/env python3
"""Site-agnostic CO2 storage-term (Sco2) calculation.

Operates on a `(time, height)` xarray Dataset with `CO2` (umol/mol), `Tair`
(degC) and `P` (kPa) data variables — the contract that site-specific
normalizers in `profile_loader.py` must produce. Layer boundaries and depths
are derived from the `height` coordinate, so this generalises to any number
of intake heights without per-site changes.
"""

import numpy as np
import xarray as xr

from domain.constants import CO2_MOL_MASS, TIME_INDEX_NAME
from services.data.transform_service import calculate_molar_density

TIME_DIM = TIME_INDEX_NAME
HEIGHT_DIM = "height"
LAYER_DIM = "layer"
REQUIRED_VARS = ("CO2", "Tair", "P")


def validate_profile_dataset(ds: xr.Dataset) -> None:
    """Check `ds` conforms to the (time, height) profile-dataset contract."""
    missing = [v for v in REQUIRED_VARS if v not in ds.data_vars]
    if missing:
        raise ValueError(f"Profile dataset missing required variable(s): {missing}")

    for var in REQUIRED_VARS:
        dims = ds[var].dims
        if dims != (TIME_DIM, HEIGHT_DIM):
            raise ValueError(
                f"Variable '{var}' has dims {dims}, expected "
                f"({TIME_DIM!r}, {HEIGHT_DIM!r})"
            )

    heights = ds[HEIGHT_DIM].values
    if not np.all(np.diff(heights) > 0):
        raise ValueError(f"{HEIGHT_DIM!r} coordinate must be strictly ascending")


def get_heights(ds: xr.Dataset) -> np.ndarray:
    """Intake heights (m), ascending."""
    return ds[HEIGHT_DIM].values


def get_layer_depths(ds: xr.Dataset) -> np.ndarray:
    """Depth (m) represented by each layer.

    The lowest layer spans ground to the lowest intake; each subsequent
    layer spans between two successive intakes.
    """
    heights = get_heights(ds)
    return heights - np.concatenate([[0], heights[:-1]])


def _layer_names(ds: xr.Dataset) -> list[str]:

    def _fmt(x: float) -> str:
        return str(int(x)) if int(x) == x else str(x)

    labels = [_fmt(h) for h in [0, *get_heights(ds)]]
    return [f"{labels[i - 1]}-{labels[i]}m" for i in range(1, len(labels))]


def get_time_interval_seconds(ds: xr.Dataset) -> float:
    """Median interval (s) between successive timestamps.

    Derived from the dataset's own time index rather than assumed, so the
    storage calculation is correct regardless of the site's sampling
    interval.
    """
    times = ds[TIME_DIM].values
    if times.size < 2:
        raise ValueError("Cannot derive time interval from fewer than 2 timestamps")
    diffs = np.diff(times).astype("timedelta64[s]").astype(float)
    return float(np.median(diffs))


def calculate_CO2_density(ds: xr.Dataset) -> xr.DataArray:
    """CO2 density (mg/m^3) via the ideal gas law.

    From CO2 mole fraction (umol/mol), air temperature (degC) and pressure
    (kPa).
    """
    molar_density = calculate_molar_density(ps=ds["P"], Ta=ds["Tair"])
    da = CO2_MOL_MASS * molar_density * ds["CO2"] / 10**3
    da.name = "CO2_density"
    da.attrs = {"units": "mg/m^3"}
    return da


def calculate_CO2_density_layers(ds: xr.Dataset) -> xr.DataArray:
    """Layer-mean CO2 density (mg/m^3).

    The lowest layer (ground to lowest intake) is assumed constant at the
    lowest intake's density. Every other layer is the simple mean of the
    density at its upper and lower bounding heights.
    """
    density_da = calculate_CO2_density(ds)
    heights = get_heights(ds)

    layers = [
        density_da.sel({HEIGHT_DIM: heights[0]}).reset_coords(HEIGHT_DIM, drop=True)
    ]
    for i in range(1, len(heights)):
        layers.append(
            density_da.sel({HEIGHT_DIM: heights[i - 1 : i + 1]}).mean(HEIGHT_DIM)
        )

    layer_da = xr.concat(layers, dim=LAYER_DIM)
    layer_da[LAYER_DIM] = _layer_names(ds)
    return layer_da.transpose(TIME_DIM, LAYER_DIM)


def calculate_delta_CO2_storage(ds: xr.Dataset) -> xr.DataArray:
    """Per-layer CO2 storage flux (Sco2, umol/m^2/s)."""
    layer_da = calculate_CO2_density_layers(ds)
    layer_da = layer_da / CO2_MOL_MASS * 10**3  # mg/m^3 -> umol/m^3
    interval_s = get_time_interval_seconds(ds)
    diff_da = (layer_da - layer_da.shift(**{TIME_DIM: 1})) / interval_s
    depth_scalar = xr.DataArray(get_layer_depths(ds), dims=LAYER_DIM)
    depth_scalar[LAYER_DIM] = diff_da[LAYER_DIM].data
    diff_da = diff_da * depth_scalar
    diff_da.name = "Sco2"
    diff_da.attrs = {"units": "umol/m^2/s"}
    return diff_da


def calculate_summed_CO2_storage(ds: xr.Dataset) -> xr.DataArray:
    """Total CO2 storage flux (Sco2, umol/m^2/s), summed across layers.

    `skipna=False` deliberately: a missing layer nulls the total rather
    than silently underestimating it.
    """
    da = calculate_delta_CO2_storage(ds).sum(LAYER_DIM, skipna=False)
    da.name = "Sco2"
    da.attrs = {"units": "umol/m^2/s"}
    return da
