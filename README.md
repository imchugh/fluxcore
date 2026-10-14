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

## Install

```bash
pip install -e .            # editable, for development
pip install -e ".[dask]"    # with dask for chunked Zarr reads
```

## Use

```python
from fluxcore.orchestration import build_L1_zarr, build_L2_zarr
from fluxcore.services.metadata.core.qc_config_schema import load_qc_config

build_L1_zarr.build(ctx, output_dir="out/zarr/L1")          # ctx: SiteContext
build_L2_zarr.build(
    load_qc_config("MySite", config_dir="configs/L2"),
    output_dir="out/zarr/L2",
    l1_dir="out/zarr/L1",
)
```

`SiteContext` = `SiteRuntimeConfig` (from a site config YAML:
`fluxcore.services.metadata.core.runtime_config_builder.build_runtime_config_from_file`)
\+ `SiteMetadata` (`site_metadata_builder.build_site_metadata_from_file`).

Planned: file-driven entry points, one per level (`run_L1(control_file)`,
`run_L2(control_file)`), PyFluxPro-style.

## Tests

```bash
python -m unittest discover -s tests
```
