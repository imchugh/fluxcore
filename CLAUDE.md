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
`build_dataframe_from_context`), the file-in entry points (`run_L1`,
`run_L2`), the contract objects, config assembly
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
- **Function 2 — file in** (`orchestration/run_from_file.py`, built 2026-11-23):
  `run_L1(control_file, ...)`, `run_L2(control_file, ...)`, PyFluxPro-style
  (one control file per site per level; L1 = the site config, L2 = the site's
  QC config). They only unpack files into the same objects and call
  function 1 — no processing logic, so the two routes can't diverge (proved on
  a real TERN site: identical data and attrs, except `instrument_uri`, which
  only an adapter's vocabulary lookup fills). Optional control-file sections,
  never set by an adapter: L1 `paths:` (`input_data_path`, `input_patch_path`,
  `output_dir`), `site_metadata:` (`SiteMetadataSchema`; `site_name` defaults
  to `site`) and `global_attrs:`; L2 `paths:` (`l1_dir`, `output_dir`). L2
  needs no metadata or attrs: it inherits them from the L1 store. Path keys
  match the functions' argument names; an argument always wins over the file
  (`global_attrs` merged key by key); relative paths resolve against the
  control file's folder. `update=True` takes the incremental path. No CLI yet
  (deliberately deferred until a standalone user needs one).

## Key modules

| Module | Role |
|---|---|
| `fluxcore/services/metadata/canonical_quantity_registry.py` | Master registry of 100+ canonical quantities (`CanonicalQuantityMetadata`: units and long_name) (no value ranges — those live only in `fluxcore/configs/qc/_range_defaults.yml`, read via `qc_config_schema.resolve_default_range`). Also exports `resolve_variance_units`. |
| `fluxcore/services/metadata/runtime_config_builder.py` | Config assembly: builds `SiteRuntimeConfig` (and the `VariableDefinition`/`RawVariableMetadata` it holds) from a structurally-validated `SiteConfig` — name syntax, quantity/units checks, object assembly. No TERN calls: `instrument_uris` is an optional adapter-supplied enrichment map (defaults to none, leaving every `instrument_uri` as `None`), and `input_data_path` is optional adapter-supplied data for where the site's raw files live (defaults to `None`; required by `file_group_builder.build_file_groups`). Two entry points, both taking `input_data_path`: `build_runtime_config_from_file(file_path, input_data_path=None)` (raw YAML in, zero TERN dependency) and `build_runtime_config(validated_config, instrument_uris=None, input_data_path=None)` (already-validated/adapter-enriched object in) — the file variant also takes `input_data_path` directly so a standalone non-TERN caller has a genuine one-call path to a runnable config, not just the object variant. |
| `fluxcore/services/metadata/site_metadata_builder.py` | `SiteMetadata` and its assembly, the other half of a `SiteContext` — and `SiteContext` itself (`runtime_config: SiteRuntimeConfig` + `metadata: SiteMetadata`), a contract object, so a caller never needs a network adapter's code just to get the type. Strict Pydantic schema (`SiteMetadataSchema`, defined in `site_config_schema` so the L1 control file's `site_metadata:` section can use it without an import cycle) plus `build_site_metadata(schema)` and `build_site_metadata_from_file(file_path) -> SiteMetadata` — reads a single-site metadata YAML (the shape of one value out of `configs/site_metadata.yml`'s `{site_name: <record>}` mapping, unwrapped to a standalone file). Deliberately strict (raises on malformed input), unlike `SiteMetadata.__init__`'s own tolerant log-and-keep-raw coercion (an accommodation for adapters fed by messy upstream RDF/yml data) — never exercised by this path since validation happens first. A standalone entry point for callers without an adapter; TERN builds `SiteMetadata` itself (`SiteRegistry.get_metadata()`). |
| `fluxcore/services/metadata/variable_registry.py` | Builds `VariableSpec` objects (flat raw→canonical mapping) from a runtime config and file groups. Use `build_variable_registry(runtime_cfg, file_groups)`. |
| `fluxcore/services/metadata/instrument_history.py` | **The one handler for a variable's instrument history** (2026-11-24). `InstrumentHistory` = roles (`None` for a single instrument, else the compound keys in config order), each with its periods in time order (consecutive same-instrument inputs merged). Every variable built from raw inputs carries one, even with a single period, and its `instrument` attr is always `current()` (comma-joined per role). One shape from `dataset_builder` through the Zarr store: the JSON attr form `to_attr()` writes, `[{role, periods: [{instrument, start, end}]}]` — lists and fixed-key records only, since Zarr's attrs writer alphabetises dict keys (a dict keyed by instrument came back out of time order). `from_attr()` parses it and rejects pre-2026-11-24 forms (rebuild the store). `netcdf_attrs(window_start, window_end)` gives NetCDF's per-year `instrument` and `instrument_history` strings (`(inst,start,end)\|...`, compound `role>...;role>...`), unchanged from before. |
| `fluxcore/services/metadata/file_group_builder.py` | Builds `FileGroup` objects (master path, format, backup and patch files) for all file groups in a site config. Use `build_file_groups(runtime_cfg)` when the target file group is not known in advance. Reads the site's raw-data directory off `runtime_cfg.input_data_path` (adapter-populated); raises if unset — this function makes no `infrastructure.paths`/TERN calls itself. Patch files come from `runtime_cfg.input_patch_path` (optional; unset or missing = none). Naming follows LoggerNet's append-only scheme, matched exactly on the master's full name: `<name>.backup`, `<name>.1.backup`, ... (higher = newer; never renumbered), and likewise `<name>.patch`, `<name>.1.patch` (`file_io.get_numbered_files`). **Overlap precedence** (`FileGroup.all_files`, applied in `dataframe_builder`): master, then backups newest→oldest, then patches newest→oldest; each file contributes only timestamps no higher-precedence file covers, whole record from the winner (no per-column merge). Patches are gap-fill records kept outside the raw directory so raw files are never edited. |
| `fluxcore/services/metadata/site_config_schema.py` | Pydantic schema models and structural validation for site YAML configs. Entry point: `validate_L1_config_structure(file)`. |
| `fluxcore/services/metadata/variable_name_parser.py` | Parses canonical variable names into components: `parse_variable_name(name) -> ParsedVariableName` (defined here). |
| `fluxcore/services/data/unit_conversions.py` | Raw-units → canonical-units conversion registry (`@register_conversion`, `get_unit_conversion`), keyed by canonical quantity; used at L1 by `dataframe_builder`. |
| `fluxcore/services/data/calculations.py` | Physical calculation registry (`@register_calculation`, `get_calculation`): derived quantities and atmospheric physics (AH↔RH, `es`, `e`, VPD, molar density, dew point, CO2 mole fraction) plus unregistered helpers such as `standard_pressure_kpa`. Also holds the L3 flux physics, ported from PyFluxPro and not yet called by any orchestration: humidity/density/heat-capacity terms, `Ta` from sonic `Tv`, `rotate_2d`, `massman_correction` (instrument geometry and averaging period are arguments), fluxes from covariances (`ustar`, `Fm`, `Fh` via `Fhv`, `Fe`), WPL (`wpl_correct_Fe`/`wpl_correct_Fco2`), `L`, `ET`, `Fa`, and single-point `Sco2` and soil heat storage. Registered under a canonical quantity where the output is one. Saturation vapour pressure is Buck (1996) throughout, not PyFluxPro's Tetens. |
| `fluxcore/services/data/raw_data_loader.py` | Loads TOA5 and EddyPro files (`load_raw_data`, `load_raw_data_since`, `load_raw_header`, `get_data_loader`), keyed by `FileFormat`. A site config's `file_formats:` values are *logger* types (`LoggerType`: CSI, LICOR), not formats — `enums.DEFAULT_FILE_FORMAT` maps logger → format and is the only place that assumption lives (CSI loggers also write TOB1/TOB3, which fluxcore does not parse; TERN converts TOB3 to TOA5 upstream). |
| `fluxcore/orchestration/dataframe_builder.py` | ETL: loads raw data → converts units → merges instrument periods → renames to canonical names. Entry point: `build_dataframe_from_context(ctx, quantities=None, start_date=None)`. Low-level: `build_dataframe(file_groups, specs_by_alias, quantities=None, ...)` for callers that already hold those objects. |
| `fluxcore/orchestration/dataset_builder.py` | High-level API: `build_dataset_from_context(ctx, ...)`. Orchestrates dataframe build → derived quantities (`derive_missing_humidity`, `derive_co2_mole_fraction`, `Fn`, `day_night`) → xarray Dataset with variable and global metadata. The raw-data → dataframe step is `dataframe_builder.build_dataframe_and_specs` (shared with `build_dataframe_from_context`); `DatasetBuildIntermediate` lives in `derived_quantities`. |
| `fluxcore/orchestration/derived_quantities.py` | Derives missing RH↔AH and CO2 mole fraction; maintains metadata lineage. Also always (re)computes `Fn` (net radiation) from `Fsd`/`Fsu`/`Fld`/`Flu` — unlike the RH/CO2 gap-fills; sites don't import a raw `Fn`, which is either redundant with this sum or from a disagreeing secondary "NR Lite"-family radiometer. |
| `fluxcore/orchestration/build_L1_zarr.py` | `build(ctx, output_dir, year=None, global_attrs=None)` / `update(ctx, output_dir, global_attrs=None)` — takes a `SiteContext` and an explicit directory, never a site name to resolve (TERN's site-name wrappers are its `orchestration/build_runner.py`). Writes a per-site whole-history Zarr store (one store per site, no year-splitting) — the canonical L1 intermediate `build_L1_nc.py`'s operational path reads from. `update()` is the cheap incremental path (TERN runs it every 30 min): delegates to `orchestration.incremental_zarr.incremental_update` for the checkpoint/tail-append/fallback envelope, supplying a `produce_tail` callback (raw-data reload from the checkpoint) and a `full_rebuild` callback (`build()`). `build()` is a full rebuild from all raw data — used to seed a new store and for periodic reconciliation (TERN: nightly); it also stamps the store with a hash of the site's assembled `SiteRuntimeConfig`, which `update()` checks every cycle to force a rebuild on config drift (see `incremental_zarr`). `instrument_history` is stored in its full attr form (see `instrument_history.py`), not flattened/year-clipped like NetCDF's; `build_L1_nc.serialize_inst_history` does the year-clipping at export. |
| `fluxcore/orchestration/incremental_zarr.py` | Shared checkpoint/tail-append/fallback envelope used by both `build_L1_zarr.update()` and `build_L2_zarr.update()` — `incremental_update(store_path, config_hash, produce_tail, full_rebuild, logger, label)`: store missing or `config_hash` mismatch → `full_rebuild()`; else read the checkpoint, call `produce_tail(checkpoint)`, append if non-empty; any exception → `full_rebuild()`. All level-specific logic (raw-data reload, QC + MAD lookback window, tail-attrs refresh) lives in each caller's `produce_tail`/`full_rebuild` closures, not here. `hash_config(*configs)` hashes the assembled config object(s) that drive processing (dataclasses — stdlib or pydantic —, pydantic models, dicts/enums/paths/datetimes), not raw YAML text, so cosmetic edits don't force a rebuild — but field names are part of the hash, so renaming a config field forces one full rebuild per store. Hash only what would make an append *wrong* (site config, QC config); anything that only leaves the store briefly stale (site metadata, physical constants, code changes) is left to the nightly full rebuild. |
| `fluxcore/orchestration/dataset_finishing.py` | Finishing steps shared by the Zarr builders and NetCDF exporters: spatial dims + CRS variable, global attrs (`assign_global_attrs`: core `nc_global_attrs.yml` overlaid by a caller's `global_attrs`), coverage attrs (`assign_coverage_attrs`, incl. the combined range for an appended tail), missing-value `_QCFlag`s, CF attr filtering/serialization. |
| `fluxcore/orchestration/build_L1_nc.py` | Export step (export-only — see "NetCDF export"): writes a site's L1 Zarr store out as annual NetCDF files. `build_from_zarr(site_name, zarr_dir, output_dir, ...)` — directories required; `site_name` is used only for leaf file naming. Network global attributes are a `global_attrs` dict argument (TERN passes `configs/nc_metadata.yml`; `<site>` placeholders in string values are substituted); only the generic CF `nc_dim_attrs.yml` is loaded here. Adds spatial dims, QC flags, global/variable metadata. Separate from dataset construction by design. Reads only the already-processed Zarr store (`fluxcore/orchestration/build_L1_zarr.py`), never raw data — there is no raw-to-NetCDF path; to export a site without a store, run `build_L1_zarr.build()` first. |
| `fluxcore/services/metadata/qc_config_schema.py` | L2 QC config schema (`VariableQCSpec`: corrections `linear` / `wd_offset`, checks `range_check` / `exclude_dates` / `dependency_check` / `flag_check` / `mad_filter`), loaders (`load_qc_config`, `load_range_defaults`, `load_dependency_defaults`) and `resolve_qc_config`, which merges a site's config with the shared defaults field by field. A site file has the L1 config's shape — `site:` + `variables:` (empty = defaults only) + optional `paths:` (changed from a flat `{variable: spec}` mapping 2026-11-23; every TERN file migrated, loading identically); `_dependency_defaults.yml` stays flat (`QCVariablesSchema`). `load_qc_config(site_name, config_dir)` takes the directory explicitly and raises if the file's `site:` differs from `site_name`; the shared defaults load from `QC_DEFAULTS_DIR` (`CONFIG_PATH / "qc"`). (TERN resolves its site-file location in its own `services/metadata/qc_config_loader.py`.) |
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
- **Humidity rule.** With no site or default `dependency_check`, an average AH depends on the Ta and RH sharing its name suffix (`AH_2m` → `Ta_2m`, `RH_2m`; `_humidity_dependency`, existence-gated, so `AH_IRGA` gets nothing). HMPs measure Ta/RH and AH is calculated from them (logger or `derive_missing_humidity`), so a 0 V dropout (Ta ≈ −80 °C) gives an AH of ~0.3 g/m³ that passes AH's own range. Name-based because the names are per-site, unlike `_dependency_defaults.yml`'s exact keys.
- **Existence-gating, not context branching.** A default entry, or an individual source name inside one, that is not a variable in the site's L1 store is silently dropped. This is how EddyPro/logger-flux sites are exempted from SONIC/IRGA defaults with no site-name or instrument-type logic — the engine must know as little site context as possible; anything specific to a site or flux system is config data. A site's *own* entries are validated strictly (`validate_qc_config_variables`) and raise on unknown names.
- **Check semantics.** `dependency_check`: a source is "bad" if its own resolved code is non-zero in this run, or (if it has no checks) if it is missing — a variable that is NaN for part of the record therefore flags its dependents there. `flag_check`: PyFluxPro's `EPQCFlagCheck` — `{source: [flag vars], reject: [values]}` on the flux, code 9; flag variables are gate-only (not masked, no `_QCFlag` of their own). Logger QC flag schemes differ by `flux_system` (EasyFlux 1–9, SmartFlux 0/1/2), so `flag_check` is configured **per site, never defaulted**.
- **Corrections** (`linear`: slope/offset; `wd_offset`: wind-direction offset modulo 360) are PyFluxPro's Linear and Wd offset, date-windowed (inclusive, open-ended allowed), configured in the site's L2 file in canonical units. L2, not L1, deliberately: L1 stays the unvarnished homogenised record. They run before every check, so the checks (and other variables' dependency/MAD references) see corrected data; corrected records get code 10 and stay unmasked, and the variable records its windows in a `linear_correction`/`wd_offset` attribute.
- **Flag codes** are PyFluxPro's (1 missing, 2 range, 6 exclude_dates, 9 flag_check, 10 corrected, 23 dependency, 24 MAD), applied in PyFluxPro's order (corrections → range → flag_check → exclude_dates → MAD → dependency); the last to fire wins. Only 0 and 10 count as good: every other code masks the value.
- **`mad_filter`** is a port of PyFluxPro's two-stage despike. It is not idempotent, so it must only ever run on L1 data — which `build()`/`update()` guarantee. Its first and last input records are untestable and pass (a deliberate departure from PyFluxPro); `update()` therefore holds the newest L1 record back at any site with a MAD configured, so the L2 store lags L1 by one record there.
- **Hybrid L3.** PyFluxPro does L3 (flux calculation): it recomputes provisional TERNFLUX logger fluxes from L2 covariances, and for finalised fluxes (EasyFlux/SmartFlux, later EddyPro) only fills points that are missing. So L2 fluxes carry no SONIC/IRGA dependency (those protect the covariances and concentrations), and a flux masked by `flag_check` becomes an L3 fill candidate.
- **Not implemented:** DiurnalCheck (needs climatology state). ExcludeHours was deliberately dropped.

## Variable Naming Conventions

- **Canonical form**: `{quantity}_{statistic_suffix}_{qualifier}` — e.g. `Fco2_Av`, `Ta_Sd_2m`
- **Processing aliases**: `{canonical_name}#{position}` (the input's index in `input_variables`; `variable_registry.input_alias`) — e.g. `Fco2_Av#0`. Internal to `dataframe_builder`, never written out. `#` because canonical names can end in `_<digit>` (`Precip_1`), so `_<n>` could collide
- **Variance → Stdev**: `_Vr` suffix replaced with `_Sd` in output
- **QC flags**: `{variable}_QCFlag`

### Site config: `input_variables`

A variable's `input_variables` is a **list** of its sources, in time order. Each entry is one column (`name`) read from one `file` over an optional `begin`/`end` period:

```yaml
Ta_15m:
  statistic_type: average
  height: 14.7
  input_variables:
    - name: Ta_HMP_15m_Avg
      instrument: Vaisala HMP155
      units: degC
      file: Site_EC_slow_met
```

A list rather than a mapping keyed by column name, because a source is identified by column *and* file *and* period, not by column alone: a logger change that keeps a column name feeds one variable the same `name` from two files, and a units or instrument change on an unchanged column feeds it the same `name` from the same file over consecutive periods (see the next-but-one section). The one restriction is that inputs sharing a `name` + `file` must not overlap in time (`begin`/`end` inclusive, missing = open-ended; `validate_same_source_periods_disjoint`). Changed from a name-keyed mapping 2026-11-17; every config was migrated with no change to what it builds. Same-file eras allowed 2026-11-24 (registry keyed by `{canonical_name}#{position}` instead of `{raw_name}_gp{n}`; raw columns copied per input, not renamed).

### Site config: `instrument` field notation

Within each input, the `instrument` field has two valid forms:

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

### Site config: stitching inputs across a logger change

A variable can list several inputs, each with `begin:`/`end:` dates; `dataframe_builder` joins them into one series at the dates (instrument or logger changeovers). Converting raw data is per input, so these per-input fields may differ between eras:

- `units` — each input converted from its own units.
- `diag_type` (counters) — e.g. an old `invalid_count` warning count stitched to a new `valid_count` sample count; each is normalised to `invalid_count` separately.
- `instrument` — instrument history records each instrument's period; consecutive inputs from one instrument are one period, and an instrument returning after a changeover (A → B → A) is rejected.
- `statistic_type` (optional) — the raw form when it differs from the variable's, **only** `variance` ↔ `standard_deviation`. A variance input is square-rooted; a standard-deviation input passes through. Defaults to the variable's `statistic_type`. The variable's own `statistic_type` still sets the output name, attrs, QC and derivations; the input's only drives conversion (`VariableSpec.raw_statistic_type`) and unit validation.

`flux_file` names the **current** flux file; a logger changeover doesn't need it changed beyond pointing at the new logger's file. Older flux eras come in through the variables' raw inputs, and a full build's time extent (`dataframe_builder` step 5, which trims ancillary data predating the flux system) is the union of `flux_file` and every file group stitched into the same variables (`_flux_extent_groups`). Incremental tails still end at the last record of `flux_file` itself, the only flux file still growing, so a backfill of an older era (e.g. a comms gap patched from the logger) reaches the store through the nightly full rebuild.

```yaml
Ux_SONIC_Sd:
  statistic_type: standard_deviation
  height: 15.0
  input_variables:
    - name: covUxUx                # old logger stored variance
      instrument: Campbell Scientific CSAT3B
      units: m^2/s^2
      statistic_type: variance
      file: Site_EC_slow_flux
      end: 2026-10-20 11:30
    - name: Ux_stdev               # new logger stores standard deviation
      instrument: Campbell Scientific CSAT3B
      units: m/s
      file: Site_EC_TERNflux
      begin: 2026-10-20 12:00
```

The same applies within one column: when only a column's metadata changes (the logger program switched from percent to a fraction, or a sensor was swapped without renaming the column), list the column again with the same `name` and `file` and non-overlapping periods:

```yaml
Sws_0.05ma1:
  statistic_type: average
  height: -0.05
  input_variables:
    - name: VWC_1_Avg
      instrument: Campbell Scientific CS616
      units: percent
      file: Site_soil
      end: 2026-03-01 09:30
    - name: VWC_1_Avg              # same column, now written as a fraction
      instrument: Campbell Scientific CS616
      units: m^3/m^3
      file: Site_soil
      begin: 2026-03-01 10:00
```

### Site config: horizontal-location descriptions

The canonical name's horizontal-location qualifier (`Ts_a`, `Sws_b`, …) is a bare letter that groups co-located sensor arrays (e.g. one soil pit) but carries no meaning of its own. A site describes its letters in its own site config, beside the variable names that use them:

```yaml
custom_metadata:
  horizontal_location:
    belowground: {a: bare soil array 1, b: mulga array 1}
    aboveground: {...}
```

Validated by `HorizontalLocationConfig` (single-letter keys), carried onto `SiteRuntimeConfig.custom_metadata`. Deliberately the site config, not TERN's central site database: the letters are defined by the config's own variable names and change with sensor deployments, and non-TERN core users have no central database (the TERN adapter could still fill it from one later, like `instrument_uri`). Set at AliceSpringsMulga, Boyagin, Fletcherview and Ridgefield (migrated 2026-10-13 from the retired `configs/site_custom_metadata.yml`). **Not yet emitted to output** — the follow-up is a per-variable NetCDF attribute (e.g. `Ts_a: horizontal_location = "bare soil array 1"`).
