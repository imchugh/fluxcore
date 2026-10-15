# CLAUDE.md — fluxcore

## What this is

`fluxcore` is the network-agnostic eddy covariance processing core, extracted
2026-10-14 (with history, via `git filter-repo`) from the TERN-EP flux data
pipeline (`imchugh/TERN-EP_flux_data_pipeline`), which now depends on it via
an editable install. It turns raw logger files into per-site, per-level Zarr
stores: L1 = homogenisation, L2 = QC (a PyFluxPro port), L3 = flux
calculation (future). NetCDF is an export format only.

**Hard rule: fluxcore knows nothing about any particular network.** No
directory-layout templating, no metadata services, no site names resolved
to configs, no scheduling. Every location is a plain argument; every
network-specific input arrives as a contract object or argument built by
an adapter. `tests/test_self_contained.py` enforces that every first-party
import stays inside `fluxcore`.

**Scope rule: fluxcore is a processing core, not a toolbox.** Its public
surface is the entry points and contract objects below, plus introspection of
its own domain (canonical quantities, name parser, range defaults, QC config
resolution, raw-file reading). It only ever *reads* raw files. Writing or
splicing raw files and general-purpose helpers (JSON/YAML writers, file
finders, local-time helpers, gap diagnostics) live outside it — currently in
the TERN pipeline's `fluxtools/` package, which depends on fluxcore and not
the reverse. Don't add a public helper without a caller in fluxcore or a
contract need; fluxcore reads only its own packaged configs, from
`domain.constants.CONFIG_PATH` (resolved from the installed module's location).

## Public API

`fluxcore/__init__.py` re-exports the supported surface (see its `__all__`):
the level builders (`build_L1_zarr`, `build_L1_nc`, `build_L2_zarr`,
`build_L2_nc` modules; `build_dataset_from_context`,
`build_dataframe_from_context`), the contract objects, config assembly
(`build_runtime_config[_from_file]`, `build_site_metadata_from_file`,
`load_qc_config`, `validate_all`, ...), the domain vocabulary (enums,
`DATA_TIME_FORMAT`, `SITE_PLACEHOLDER`), domain introspection (canonical
quantities, `parse_variable_name`, range defaults, `resolve_qc_config`,
`get_calculation`, `profile_storage`) and raw-file reading (`load_raw_data`,
`load_raw_header`, `get_data_loader`, ...). Import these as `from fluxcore
import X`. Anything reached by a deeper import is internal and may change
without notice; `tests/test_public_api.py` checks every exported name
resolves.

Known borrowers of internals: TERN and `fluxtools` still import a few
`fluxcore.infrastructure` utilities directly (`file_io.read_yml` and other
`file_io` readers/writers, `data_conditioning.infer_interval`/
`check_datetime_index`, `datetime_utils.SunTime`/`get_timezone`/
`get_standard_timezone`). They are not API: before changing one, grep those
repos, or copy what they need into `fluxtools` first.

Adding to the API is a decision, not a side effect: a new public name needs a
caller in fluxcore or a contract need (see the scope rule above).

## Development

- Python 3.12; dependencies in `pyproject.toml`. On the TERN VM it is
  installed editable into the `ep_cntl` conda env from `/opt/TERN_EP/fluxcore`
  — edits here take effect in the TERN pipeline immediately.
- Tests: `python -m unittest discover -s tests` (pytest isn't used).
- Lint/format: `ruff` (config in `pyproject.toml`).

## Layout

```
fluxcore/domain/          enums, constants (no I/O)
fluxcore/infrastructure/  file I/O, data conditioning, sun-position/timezone utils
fluxcore/services/        data/ (loaders, conversions, calculations, QC checks,
                          storage term); metadata/ (schemas, config assembly,
                          registries)
fluxcore/orchestration/   dataframe/dataset builders, L1/L2 Zarr builders,
                          incremental-update envelope, QC pipeline, NetCDF export
fluxcore/configs/         package data: canonical_quantities, nc_dim_attrs,
                          nc_global_attrs, qc/ defaults
```

Models live with the module that builds or validates them (e.g. `VariableSpec`
in `variable_registry`, `SiteMetadata` in `site_metadata_builder`), not in a
central models package; public ones are reached through `fluxcore/__init__.py`.
Separate a model out only if several modules share it with no natural owner,
or to break an import cycle.

## Entry points: two front doors (agreed design)

- **Function 1 — live objects in** (what adapters use): `build_dataset_from_context(ctx)`,
  `build_L1_zarr.build/update(ctx, output_dir, global_attrs=None)`,
  `build_L1_nc.build_from_zarr(...)`, `build_L2_zarr.build/update(qc_config, output_dir, l1_dir)`,
  `build_L2_nc.build_from_zarr(...)`. Contract objects: `SiteContext`
  (`SiteRuntimeConfig` + `SiteMetadata`), `SiteQCConfig`, `VariableSpec`,
  `FileGroup`, `CanonicalQuantityMetadata` — all in `fluxcore/services/metadata/`.
- **Function 2 — file in** (planned, not built): one thin wrapper per level,
  `run_L1(control_file)`, `run_L2(control_file)`, PyFluxPro-style (one control
  file per site per level; L1 = the site config, L2 = the QC deltas file). It
  only unpacks files into the same objects and calls function 1 — no
  processing logic, so the two routes can't diverge. Control files get
  optional path fields plus optional `site_metadata:` and `global_attrs:`
  sections (L1 only; inherited downstream via store attrs). An explicit
  argument/live object always wins over the file; adapters such as TERN's
  never set these sections.

## Key modules

| Module | Role |
|---|---|
| `fluxcore/services/metadata/canonical_quantity_registry.py` | Master registry of 100+ canonical quantities (`CanonicalQuantityMetadata`: units and long_name) (no value ranges — those live only in `fluxcore/configs/qc/_range_defaults.yml`, read via `qc_config_schema.resolve_default_range`). Also exports `resolve_variance_units`. |
| `fluxcore/services/metadata/runtime_config_builder.py` | Config assembly: builds `SiteRuntimeConfig` (and the `VariableDefinition`/`RawVariableMetadata` it holds) from a structurally-validated `SiteConfig` — name syntax, quantity/units checks, object assembly. No TERN calls: `instrument_uris` is an optional adapter-supplied enrichment map (defaults to none, leaving every `instrument_uri` as `None`), and `input_data_path` is optional adapter-supplied data for where the site's raw files live (defaults to `None`; required by `file_group_builder.build_file_groups`). Two entry points, both taking `input_data_path`: `build_runtime_config_from_file(file_path, input_data_path=None)` (raw YAML in, zero TERN dependency) and `build_runtime_config(validated_config, instrument_uris=None, input_data_path=None)` (already-validated/adapter-enriched object in) — the file variant also takes `input_data_path` directly so a standalone non-TERN caller has a genuine one-call path to a runnable config, not just the object variant. |
| `fluxcore/services/metadata/site_metadata_builder.py` | `SiteMetadata` and its assembly, the other half of a `SiteContext` — and `SiteContext` itself (`runtime_config: SiteRuntimeConfig` + `metadata: SiteMetadata`), a contract object, so a caller never needs a network adapter's code just to get the type. Strict Pydantic schema (`SiteMetadataSchema`) plus `build_site_metadata_from_file(file_path) -> SiteMetadata` — reads a single-site metadata YAML (the shape of one value out of `configs/site_metadata.yml`'s `{site_name: <record>}` mapping, unwrapped to a standalone file). Deliberately strict (raises on malformed input), unlike `SiteMetadata.__init__`'s own tolerant log-and-keep-raw coercion (an accommodation for adapters fed by messy upstream RDF/yml data) — never exercised by this path since validation happens first. A standalone entry point for callers without an adapter; TERN builds `SiteMetadata` itself (`SiteRegistry.get_metadata()`). |
| `fluxcore/services/metadata/variable_registry.py` | Builds `VariableSpec` objects (flat raw→canonical mapping) from a runtime config and file groups. Use `build_variable_registry(runtime_cfg, file_groups)`. |
| `fluxcore/services/metadata/file_group_builder.py` | Builds `FileGroup` objects (master path, format, backup files) for all file groups in a site config. Use `build_file_groups(runtime_cfg)` when the target file group is not known in advance. Reads the site's raw-data directory off `runtime_cfg.input_data_path` (adapter-populated); raises if unset — this function makes no `infrastructure.paths`/TERN calls itself. |
| `fluxcore/services/metadata/site_config_schema.py` | Pydantic schema models and structural validation for site YAML configs. Entry point: `validate_L1_config_structure(file)`. |
| `fluxcore/services/metadata/variable_name_parser.py` | Parses canonical variable names into components: `parse_variable_name(name) -> ParsedVariableName` (defined here). |
| `fluxcore/services/data/unit_conversions.py` | Raw-units → canonical-units conversion registry (`@register_conversion`, `get_unit_conversion`), keyed by canonical quantity; used at L1 by `dataframe_builder`. |
| `fluxcore/services/data/calculations.py` | Physical calculation registry (`@register_calculation`, `get_calculation`): derived quantities and atmospheric physics (AH↔RH, `es`, `e`, VPD, molar density, dew point, CO2 mole fraction) plus unregistered helpers such as `standard_pressure_kpa`. The home for L3 flux physics as it is added. |
| `fluxcore/services/data/raw_data_loader.py` | Loads TOA5 and EddyPro files (`load_raw_data`, `load_raw_data_since`, `load_raw_header`, `get_data_loader`), keyed by `FileFormat`. A site config's `file_formats:` values are *logger* types (`LoggerType`: CSI, LICOR), not formats — `enums.DEFAULT_FILE_FORMAT` maps logger → format and is the only place that assumption lives (CSI loggers also write TOB1/TOB3, which fluxcore does not parse; TERN converts TOB3 to TOA5 upstream). |
| `fluxcore/orchestration/dataframe_builder.py` | ETL: loads raw data → converts units → merges instrument periods → renames to canonical names. Entry point: `build_dataframe_from_context(ctx, quantities=None, start_date=None)`. Low-level: `build_dataframe(file_groups, specs_by_alias, quantities=None, ...)` for callers that already hold those objects. |
| `fluxcore/orchestration/dataset_builder.py` | High-level API: `build_dataset_from_context(ctx, ...)`. Orchestrates dataframe build → derived quantities (`derive_missing_humidity`, `derive_co2_mole_fraction`, `Fn`, `day_night`) → xarray Dataset with variable and global metadata. The raw-data → dataframe step is `dataframe_builder.build_dataframe_and_specs` (shared with `build_dataframe_from_context`); `DatasetBuildIntermediate` lives in `derived_quantities`. |
| `fluxcore/orchestration/derived_quantities.py` | Derives missing RH↔AH and CO2 mole fraction; maintains metadata lineage. Also always (re)computes `Fn` (net radiation) from `Fsd`/`Fsu`/`Fld`/`Flu` — unlike the RH/CO2 gap-fills; sites don't import a raw `Fn`, which is either redundant with this sum or from a disagreeing secondary "NR Lite"-family radiometer. |
| `fluxcore/orchestration/build_L1_zarr.py` | `build(ctx, output_dir, year=None, global_attrs=None)` / `update(ctx, output_dir, global_attrs=None)` — takes a `SiteContext` and an explicit directory, never a site name to resolve (TERN's site-name wrappers are its `orchestration/build_runner.py`). Writes a per-site whole-history Zarr store (one store per site, no year-splitting) — the canonical L1 intermediate `build_L1_nc.py`'s operational path reads from. `update()` is the cheap incremental path (TERN runs it every 30 min): delegates to `orchestration.incremental_zarr.incremental_update` for the checkpoint/tail-append/fallback envelope, supplying a `produce_tail` callback (raw-data reload from the checkpoint) and a `full_rebuild` callback (`build()`). `build()` is a full rebuild from all raw data — used to seed a new store and for periodic reconciliation (TERN: nightly); it also stamps the store with a hash of the site's assembled `SiteRuntimeConfig`, which `update()` checks every cycle to force a rebuild on config drift (see `incremental_zarr`). `instrument_history` is stored in full structured form (not flattened/year-clipped like NetCDF's) since Zarr attrs are JSON-native; `build_L1_nc.build_from_zarr()` decodes it (`dataset_finishing.decode_instrument_history`) and reproduces the year-clipping itself. |
| `fluxcore/orchestration/incremental_zarr.py` | Shared checkpoint/tail-append/fallback envelope used by both `build_L1_zarr.update()` and `build_L2_zarr.update()` — `incremental_update(store_path, config_hash, produce_tail, full_rebuild, logger, label)`: store missing or `config_hash` mismatch → `full_rebuild()`; else read the checkpoint, call `produce_tail(checkpoint)`, append if non-empty; any exception → `full_rebuild()`. All level-specific logic (raw-data reload, QC + MAD lookback window, tail-attrs refresh) lives in each caller's `produce_tail`/`full_rebuild` closures, not here. `hash_config(*configs)` hashes the assembled config object(s) that drive processing (dataclasses — stdlib or pydantic —, pydantic models, dicts/enums/paths/datetimes), not raw YAML text, so cosmetic edits don't force a rebuild — but field names are part of the hash, so renaming a config field forces one full rebuild per store. Hash only what would make an append *wrong* (site config, QC config); anything that only leaves the store briefly stale (site metadata, physical constants, code changes) is left to the nightly full rebuild. |
| `fluxcore/orchestration/dataset_finishing.py` | Finishing steps shared by the Zarr builders and NetCDF exporters: spatial dims + CRS variable, global attrs (`assign_global_attrs`: core `nc_global_attrs.yml` overlaid by a caller's `global_attrs`), coverage attrs (`assign_coverage_attrs`, incl. the combined range for an appended tail), missing-value `_QCFlag`s, CF attr filtering/serialization, and `encode_instrument_history`/`decode_instrument_history` (the Zarr-safe form of `instrument_history` and its inverse). |
| `fluxcore/orchestration/build_L1_nc.py` | Export step (export-only — see "NetCDF export"): writes a site's L1 Zarr store out as annual NetCDF files. `build_from_zarr(site_name, zarr_dir, output_dir, ...)` — directories required; `site_name` is used only for leaf file naming. Network global attributes are a `global_attrs` dict argument (TERN passes `configs/nc_metadata.yml`; `<site>` placeholders in string values are substituted); only the generic CF `nc_dim_attrs.yml` is loaded here. Adds spatial dims, QC flags, global/variable metadata. Separate from dataset construction by design. Reads only the already-processed Zarr store (`fluxcore/orchestration/build_L1_zarr.py`), never raw data — there is no raw-to-NetCDF path; to export a site without a store, run `build_L1_zarr.build()` first. |
| `fluxcore/services/metadata/qc_config_schema.py` | L2 QC config schema (`VariableQCSpec`: corrections `linear` / `wd_offset`, checks `range_check` / `exclude_dates` / `dependency_check` / `flag_check` / `mad_filter`), loaders (`load_qc_config`, `load_range_defaults`, `load_dependency_defaults`) and `resolve_qc_config`, which merges a site's config with the shared defaults field by field. `load_qc_config(site_name, config_dir)` takes the directory explicitly; the shared defaults load from `QC_DEFAULTS_DIR` (`CONFIG_PATH / "qc"`). (TERN resolves its site-file location in its own `services/metadata/qc_config_loader.py`.) |
| `fluxcore/services/data/qc_service.py` | The L2 checks themselves: small pure functions returning a boolean "bad" mask, called directly by `qc_pipeline.apply_qc`. `mad_filter` is directly callable on pandas Series; its defaults (`MAD_*` constants) are also `MADFilterSpec`'s. |
| `fluxcore/orchestration/qc_pipeline.py` | `apply_qc(ds, qc_config)`: runs the resolved config over an xarray Dataset in dependency order, masks flagged values to NaN and writes `{var}_QCFlag` with PyFluxPro's codes. Knows nothing about defaults. |
| `fluxcore/orchestration/build_L2_zarr.py` / `build_L2_nc.py` | `build(qc_config, output_dir, l1_dir)` / `update(qc_config, output_dir, l1_dir)` (store names from `qc_config.site_name`) and `build_from_zarr(site_name, zarr_dir, output_dir, ...)`. L2 Zarr store (one per site, built from the L1 store — never from raw data or its own output) and its annual NetCDF export. `build()` is a full rebuild — also stamps the store with a hash of the site's assembled QC config (site overrides + shared defaults). `update()` appends new records only, via the same `orchestration.incremental_zarr.incremental_update` envelope `build_L1_zarr.update()` uses; a config-hash mismatch forces a full rebuild instead of appending under stale rules. TERN runs `build()` nightly, right after its L1 rebuild so it reads freshly reconciled L1; `update()` is not scheduled there (L1 runs every 30 min only to keep a dashboard current; L2 has no such consumer). `build_L2_nc` is export-only and deliberately not scheduled (no L2 NetCDF use case; see "NetCDF export"). |

## NetCDF export

NetCDF is purely an export format, never a pipeline input: each level reads
the previous level's **Zarr** store. Global attributes = the core's
`fluxcore/configs/nc_global_attrs.yml` (`Conventions`, `comment`,
`featureType`, `license`, `license_name`) overlaid by a caller's
`global_attrs` (per key), then `processing_level` and date/history set by the
builder.

### L2 QC stage

`build_L2_zarr` reads a site's L1 Zarr store, applies QC, and writes `{site}_L2.zarr`; the config is resolved before `apply_qc` runs:

```
site L2 file  +  configs/qc/_dependency_defaults.yml  +  configs/qc/_range_defaults.yml
        └── qc_config_schema.resolve_qc_config (per variable, per check FIELD) ──> apply_qc
```

- **Field-level merge.** For each variable, each check field is resolved independently: the site's own value wins, else the dependency default, else the range default. A site entry that only sets `dependency_check` therefore still gets the default `range_check`. Site files hold only deltas/overrides.
- **Existence-gating, not context branching.** A default entry, or an individual source name inside one, that is not a variable in the site's L1 store is silently dropped. This is how EddyPro/logger-flux sites are exempted from SONIC/IRGA defaults with no site-name or instrument-type logic — the engine must know as little site context as possible; anything specific to a site or flux system is config data. A site's *own* entries are validated strictly (`validate_qc_config_variables`) and raise on unknown names.
- **Check semantics.** `dependency_check`: a source is "bad" if its own resolved code is non-zero in this run, or (if it has no checks) if it is missing — a variable that is NaN for part of the record therefore flags its dependents there. `flag_check`: PyFluxPro's `EPQCFlagCheck` — `{source: [flag vars], reject: [values]}` on the flux, code 9; flag variables are gate-only (not masked, no `_QCFlag` of their own). Logger QC flag schemes differ by `flux_system` (EasyFlux 1–9, SmartFlux 0/1/2), so `flag_check` is configured **per site, never defaulted**.
- **Corrections** (`linear`: slope/offset; `wd_offset`: wind-direction offset modulo 360) are PyFluxPro's Linear and Wd offset, date-windowed (inclusive, open-ended allowed), configured in the site's L2 file in canonical units. L2, not L1, deliberately: L1 stays the unvarnished homogenised record. They run before every check, so the checks (and other variables' dependency/MAD references) see corrected data; corrected records get code 10 and stay unmasked, and the variable records its windows in a `linear_correction`/`wd_offset` attribute.
- **Flag codes** are PyFluxPro's (1 missing, 2 range, 6 exclude_dates, 9 flag_check, 10 corrected, 23 dependency, 24 MAD), applied in PyFluxPro's order (corrections → range → flag_check → exclude_dates → MAD → dependency); the last to fire wins. Only 0 and 10 count as good: every other code masks the value.
- **`mad_filter`** is a port of PyFluxPro's two-stage despike. It is not idempotent, so it must only ever run on L1 data — which `build()`/`update()` guarantee. Its first and last input records are untestable and pass (a deliberate departure from PyFluxPro); `update()` therefore holds the newest L1 record back at any site with a MAD configured, so the L2 store lags L1 by one record there.
- **Hybrid L3.** PyFluxPro does L3 (flux calculation): it recomputes provisional TERNFLUX logger fluxes from L2 covariances, and for finalised fluxes (EasyFlux/SmartFlux, later EddyPro) only fills points that are missing. So L2 fluxes carry no SONIC/IRGA dependency (those protect the covariances and concentrations), and a flux masked by `flag_check` becomes an L3 fill candidate.
- **Not implemented:** DiurnalCheck (needs climatology state). ExcludeHours was deliberately dropped.

## Variable Naming Conventions

- **Canonical form**: `{quantity}_{statistic_suffix}_{qualifier}` — e.g. `Fco2_Av`, `Ta_Sd_2m`
- **Processing aliases**: `{raw_name}_{group_id}` — e.g. `Fc_gp0`
- **Variance → Stdev**: `_Vr` suffix replaced with `_Sd` in output
- **QC flags**: `{variable}_QCFlag`

### Site config: `instrument` field notation

Within `input_variables`, the `instrument` field has two valid forms:

**Single instrument** (scalar string) — used for all variables measured by one sensor:
```yaml
instrument: Campbell Scientific CSAT3B
```

**Compound instrument** (nested mapping) — used for flux/covariance variables that require two sensor types (sonic + IRGA). Keys must be `sonic_anemometer` and `irga`:
```yaml
instrument:
  sonic_anemometer: Campbell Scientific CSAT3B
  irga: LI-COR LI-7500RS
```

`Fh` (sensible heat flux) uses the **single** form (sonic only). Although air density correction involves humidity, this dependency is not considered significant enough to warrant compound notation. Comma-separated strings (e.g. `CSAT3B, LI-7500RS`) are **not** valid — do not use them.

### Site config: horizontal-location descriptions

The canonical name's horizontal-location qualifier (`Ts_a`, `Sws_b`, …) is a bare letter that groups co-located sensor arrays (e.g. one soil pit) but carries no meaning of its own. A site describes its letters in its own site config, beside the variable names that use them:

```yaml
custom_metadata:
  horizontal_location:
    belowground: {a: bare soil array 1, b: mulga array 1}
    aboveground: {...}
```

Validated by `HorizontalLocationConfig` (single-letter keys), carried onto `SiteRuntimeConfig.custom_metadata`. Deliberately the site config, not TERN's central site database: the letters are defined by the config's own variable names and change with sensor deployments, and non-TERN core users have no central database (the TERN adapter could still fill it from one later, like `instrument_uri`). Set at AliceSpringsMulga, Boyagin, Fletcherview and Ridgefield (migrated 2026-10-13 from the retired `configs/site_custom_metadata.yml`). **Not yet emitted to output** — the follow-up is a per-variable NetCDF attribute (e.g. `Ts_a: horizontal_location = "bare soil array 1"`).
