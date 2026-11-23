# fluxcore

Network-agnostic processing core for eddy covariance flux tower data:
Campbell Scientific TOA5 / LI-COR EddyPro raw files in, per-site Zarr stores
out at each processing level:

- **L1 (homogenisation):** unit conversion, instrument-period merging,
  renaming to canonical variable names, and derived quantities (RH↔AH, CO2
  mole fraction, net radiation).
- **L2 (QC):** a port of PyFluxPro's L2 checks (range, exclude dates,
  dependency, logger flag, MAD despike), driven by per-site overrides layered
  on shared defaults.

NetCDF is available as an export format.

fluxcore holds no knowledge of any particular network's directory layout,
metadata service or schedule. A network adapter (e.g. the TERN-EP pipeline)
builds fluxcore's contract objects (`SiteContext`, `SiteQCConfig`) from its
own sources and calls the builders, passing locations as plain arguments.
Without an adapter, `run_L1`/`run_L2` do the same from control files.

## Install

```bash
pip install -e .            # editable, for development
pip install -e ".[dask]"    # with dask for chunked Zarr reads
```

## Use

Two front doors, which build identical stores.

**From control files** (standalone, PyFluxPro-style): one file per site per
level. Relative paths resolve against the control file's own folder.

```python
from fluxcore import run_L1, run_L2

run_L1("MySite.yml")  # -> out/MySite_L1.zarr
run_L2("MySite_L2.yml")  # -> out/MySite_L2.zarr
```

The L1 control file is the site config plus three optional sections:

```yaml
site: MySite
# file_formats, flux_system, flux_file, variables: the site config
paths:
  input_data_path: raw        # raw logger files
  input_patch_path: patches   # optional gap-fill .patch files
  output_dir: out
site_metadata:                # site_name defaults to `site`
  latitude: -22.28
  longitude: 133.25
  time_step: 30
  freq_hz: 10
global_attrs:                 # optional NetCDF/Zarr global attributes
  institution: My Institute
```

The L2 control file is the site's QC overrides (empty `variables:` = shared
defaults only):

```yaml
site: MySite
paths:
  l1_dir: out
  output_dir: out
variables:
  Ta_2m:
    range_check: {lower: -10, upper: 50}
```

Any directory, the metadata or the global attributes can also be passed as
an argument, which wins over the file (`run_L1("MySite.yml",
output_dir="elsewhere")`); `update=True` appends new records only.

**From live objects** (what a network adapter uses):

```python
from fluxcore import build_L1_zarr, build_L2_zarr, load_qc_config

build_L1_zarr.build(ctx, output_dir="out")  # ctx: SiteContext
build_L2_zarr.build(
    load_qc_config("MySite", config_dir="configs/L2"),
    output_dir="out",
    l1_dir="out",
)
```

`SiteContext` = `SiteRuntimeConfig` (`build_runtime_config_from_file`) +
`SiteMetadata` (`build_site_metadata_from_file`).

## Tests

```bash
python -m unittest discover -s tests
```
