"""Find, download (with caching), and read the MODIS/VIIRS granules needed
to follow a Lagrangian trajectory -- as opposed to the older approach of
pre-building a `blended_modis`/`blended_viirs` archive covering the whole
Arctic domain.

Requires a locally installed, patched er3t (see ../LAGRANGE_ER3T_PATCH.md
-- download_laads_https() has a caching bug that is fixed there) and a
valid `EARTHDATA_TOKEN` in the environment for anything to actually
download.

Both MODIS (Terra/Aqua) and VIIRS (SNPP/NOAA-20/NOAA-21) are wired up.
`find_candidates` looks up the right instrument per satellite automatically
(see `_INSTRUMENT_OF_SATELLITE`), so a mixed `satellites=(...)` list works
without extra plumbing; `fetch_granule`/`read_granule_point`/
`read_granule_region`/`read_granule_rgb` each dispatch on
`candidate.instrument`.

Three VIIRS-specific gotchas, not documented anywhere upstream as of this
writing:

- er3t's own `common.py` dataset registry has the NOAA-21 cloud-property
  tag ('5111/CLDPROP_L2_VIIRS_NOAA21') commented out as "not available
  yet" -- that comment predates the ARCSIX campaign (2024) by some margin,
  and the product should exist by now under the same naming convention
  SNPP/NOAA-20 use (see `_VIIRS_DATASET`), but it is the one tag here that
  hasn't been independently confirmed against a real download. If a
  NOAA-21 L2 fetch comes back empty/404s, that tag is the first thing to
  check against LAADS's current holdings.
- er3t's `viirs_l1b` (in `er3t.util.viirs`) has a bug when called with
  both `keep_dims=True` and a non-None `f03=`: the code path that would
  apply `f03`'s crop mask only defines the `mask` variable in the *other*
  branch, so it raises `NameError: name 'mask' is not defined`. We work
  around it in `read_granule_rgb` by calling `viirs_l1b` with `f03=None`
  (VIIRS's VNP02MOD/VNP03MOD products already share one native 750 m
  grid, so no f03-driven cropping/upscaling is needed the way MODIS's
  HKM-vs-1km-sza mismatch needs one) and reading geolocation/sza
  separately from our own `viirs_03` instance. Not patched into er3t
  itself (unlike LAGRANGE_ER3T_PATCH.md's caching fix) since it's avoidable
  purely by how we call it.
- er3t's `viirs_l1b._mask_flags()` (the default `auto_mask=True` path) is
  broken: its per-flag masking loop rebuilds the mask from the *raw* array
  on every iteration, so only the last flag in the file's `flag_values`
  list actually ends up masked, and the method's own `.filled()` call
  result is never assigned back/returned anyway -- flagged (e.g.
  bow-tie-deleted) pixels leak through as raw, unconverted sentinel
  values. Confirmed visually against a real SNPP granule: VIIRS flags the
  same fixed cross-track sample positions on every scan, so the leaked
  values show up as a dense, regular dot/diagonal-stripe pattern across
  the whole composite (and, worse, can poison read_granule_rgb's
  99th-percentile normalization, crushing real reflectance toward black).
  Worked around the same way as the previous bug -- not by patching
  er3t, but by calling `viirs_l1b(..., auto_mask=False)`, which routes
  through `_remove_flags()` instead; that method has no such bug and
  correctly NaNs out every flagged pixel before scaling.
- er3t's `modis_l2.read()` (in `er3t.util.modis`) has a copy-paste bug:
  it initializes its merged `cwp` array from `cer0_data` (cloud
  *effective radius*) instead of `cwp0_data` (cloud *water path*) --
  `cwp = cer0_data.copy()` where it should read `cwp = cwp0_data.copy()`
  (compare VIIRS's own `viirs_cldprop_l2.read_cop()`, which gets the
  equivalent line right). The invalid/clear-sky/PCL-fill logic that runs
  afterwards only overwrites *some* of `cwp`'s entries (the invalid,
  clear, and PCL-filled pixels), so every "normal" valid retrieval --
  the majority of real cloudy pixels -- silently comes back as an
  effective-radius value (microns) mislabeled as water path (g/m^2) via
  `modis_l2.data['cwp']`. Confirmed by inspecting a real cached MOD06_L2
  granule's HDF4 SDs directly and diffing against er3t 0.2.0's own
  source (2026-11-03). Worked around the same avoid-rather-than-patch
  way as the VIIRS bugs above: `read_granule_point`/`read_granule_region`
  read `Cloud_Water_Path[_<COP_FLAG>]`/`..._PCL` straight off the HDF4 SD
  themselves (`_read_modis_cop_grid`) with their own, independently
  correct merge logic, rather than trusting `modis_l2`'s assembled `cwp`
  at all -- not filed in LAGRANGE_ER3T_PATCH.md since nothing here needs
  changing inside er3t itself.

Both MODIS (MOD06_L2/MYD06_L2) and VIIRS (CLDPROP_L2_VIIRS) carry several
spectral-pair retrievals of cot/cwp/cer side by side in the same granule:
the default two-channel retrieval (2.1um + a visible/NIR band picked per
pixel), and `_16`/`_1621`/`_37` variants (1.6um+visible, 2.1um+1.6um with
NO visible channel at all, and 3.7um+visible, respectively), each with
its own `_PCL`/`_Uncertainty` companions -- confirmed against real cached
granules of both platforms. This module reads the `_1621` variant
(`COP_FLAG` below) for cot/cwp/cer on both platforms rather than the
plain default, since the default's visible channel has well-documented
poor cloud/surface contrast over bright snow/sea ice -- exactly ARCSIX's
domain -- while `_1621` uses no visible channel at all. Sebastian,
2026-11-03.
"""
from __future__ import annotations

import dataclasses
import datetime
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Set LAGRANGE_VAR_LOG to a file path (csh: `setenv LAGRANGE_VAR_LOG
# ./var_reads.log`) to have every low-level L2 reader
# (_read_modis_cop_grid/_read_modis_ctt_cth_grid/_read_viirs_cop_grid)
# append one line per call recording exactly which SDS/variable name(s)
# it read and from which file -- useful whenever it's not obvious which
# physical quantity actually ended up in e.g. PointSample.ctt (MODIS and
# VIIRS don't necessarily store the same quantity under matching names,
# shapes, or fill/valid-range conventions, even when both products call
# it "Cloud_Top_Temperature"). A no-op, and costs nothing, when unset.
_VAR_LOG_PATH = os.environ.get("LAGRANGE_VAR_LOG")


def _log_vars_read(l2_file: str, variables: Sequence[str]) -> None:
    """Append one line to _VAR_LOG_PATH (see its module comment above),
    if set: UTC timestamp, the L2 file path, and the exact variable
    name(s) just read from it, comma-separated in read order."""
    if not _VAR_LOG_PATH:
        return
    line = "%s | %s | %s\n" % (
        datetime.datetime.utcnow().isoformat(timespec="seconds"),
        l2_file,
        ", ".join(variables),
    )
    with open(_VAR_LOG_PATH, "a") as f:
        f.write(line)


MODIS_SATELLITES = ("terra", "aqua")
VIIRS_SATELLITES = ("snpp", "noaa20", "noaa21")
ALL_SATELLITES = MODIS_SATELLITES + VIIRS_SATELLITES

_INSTRUMENT_OF_SATELLITE = {
    "terra": "modis", "aqua": "modis",
    "snpp": "viirs", "noaa20": "viirs", "noaa21": "viirs",
}

# dataset_tag templates per er3t.common._sat_tags_support_ (collection 61).
# {prefix} is 'MOD' (terra) or 'MYD' (aqua).
MODIS_COLLECTION = "61"
_MODIS_DATASET = {
    "geo": MODIS_COLLECTION + "/{prefix}03",
    "l1b_hkm": MODIS_COLLECTION + "/{prefix}02HKM",   # 500 m, has the 555/865/2130 nm-ish bands
    "l1b_qkm": MODIS_COLLECTION + "/{prefix}02QKM",   # 250 m, bands 1-2 only
    "l2_cloud": MODIS_COLLECTION + "/{prefix}06_L2",  # cot/cer/cwp/cth/ctt
}
_MODIS_FILE_PREFIX = {"terra": "MOD", "aqua": "MYD"}

# Unlike MODIS, VIIRS's collection number isn't shared across satellites
# (NOAA-20 is collection 5201, SNPP/NOAA-21 are 5200/5111) -- see this
# module's docstring re: the NOAA-21 l2_cloud tag specifically. All three
# use the 750 m ("MOD"-equivalent) geolocation/L1B products, matching the
# resolution MODIS's HKM product gives us for the false/true-color panels.
_VIIRS_DATASET = {
    "snpp":   {"geo": "5200/VNP03MOD", "l1b": "5200/VNP02MOD", "l2_cloud": "5111/CLDPROP_L2_VIIRS_SNPP"},
    "noaa20": {"geo": "5201/VJ103MOD", "l1b": "5201/VJ102MOD", "l2_cloud": "5111/CLDPROP_L2_VIIRS_NOAA20"},
    "noaa21": {"geo": "5200/VJ203MOD", "l1b": "5200/VJ202MOD", "l2_cloud": "5111/CLDPROP_L2_VIIRS_NOAA21"},
}

# Which MOD06/CLDPROP spectral-pair retrieval to use for cot/cwp/cer,
# project-wide (see this module's docstring) -- '' is the operational
# default (2.1um+visible/NIR), '16'/'1621'/'37' the alternate band-pair
# retrievals carried in the same granule. Empty string for the plain
# vname (e.g. "Cloud_Water_Path"), non-empty appends "_<flag>"
# (e.g. "Cloud_Water_Path_1621").
COP_FLAG = "1621"


def _cop_vname(field: str, cop_flag: str = COP_FLAG, pcl: bool = False) -> str:
    """Build a MOD06/CLDPROP cloud-optical-property variable name, e.g.
    _cop_vname("Cloud_Water_Path", "1621", pcl=True) -> "Cloud_Water_Path_1621_PCL"."""
    name = field if not cop_flag else "%s_%s" % (field, cop_flag)
    return "%s_PCL" % name if pcl else name


@dataclasses.dataclass
class OverpassCandidate:
    """One candidate granule from `find_candidates`."""

    satellite: str       # 'terra', 'aqua', 'snpp', 'noaa20', 'noaa21'
    instrument: str      # 'modis' or 'viirs'
    filename_tag: str    # e.g. 'A2024158.1505' -- what er3t's downloaders match on
    granule_time: Optional[datetime.datetime]
    offset_minutes: Optional[float]  # granule_time - requested center time


@dataclasses.dataclass
class GranuleFiles:
    """Local paths for one downloaded (or cache-hit) granule."""

    candidate: OverpassCandidate
    geo_file: Optional[str] = None
    l1b_file: Optional[str] = None
    l2_file: Optional[str] = None


@dataclasses.dataclass
class PointSample:
    """Satellite-retrieved values at one lon/lat, pulled from one granule.

    cwp/cot are read off the native (1 km) MOD06 grid; cth/ctt off the
    coarser 5 km grid the same product also carries -- see
    LAGRANGE_ER3T_PATCH.md's sibling note in satellite.py's module
    docstring on why these are read separately.
    """

    candidate: OverpassCandidate
    distance_km: float
    cwp: float = np.nan
    cot: float = np.nan
    cth: float = np.nan
    ctt: float = np.nan
    # Viewing (sensor) zenith angle [degrees] at the nearest pixel -- MODIS:
    # from the companion MOD03/MYD03 geolocation file (same native 1 km grid
    # already used for cot/cwp, via er3t's modis_03 reader); VIIRS: from the
    # CLDPROP file's own geolocation_data group (same native grid as
    # cot/cwp/cth/ctt, no separate geo file needed). Added to investigate
    # whether LWP discrepancies between satellites/platforms track with
    # viewing geometry (Sebastian, 2026-11-18).
    vza: float = np.nan
    # 'rgb' is an (ny, nx, 3) reflectance-ish composite plus its own
    # lon/lat grid, for rendering a wide-area frame -- not reduced to a
    # point. None until a renderer asks for it (fetched separately, it's
    # a different, bigger download than the geo/L2 files above).
    rgb: Optional[np.ndarray] = None
    rgb_lon: Optional[np.ndarray] = None
    rgb_lat: Optional[np.ndarray] = None


def _tag_to_datetime(tag: str) -> Optional[datetime.datetime]:
    """'A2024158.1505' (AYYYYDDD.HHMM, as get_satfile_tag returns) ->
    datetime, or None if it doesn't parse."""
    try:
        doy_part, hhmm_part = tag.split(".")
        year = int(doy_part[1:5])
        doy = int(doy_part[5:8])
        hh = int(hhmm_part[:2])
        mm = int(hhmm_part[2:4])
        return datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy - 1, hours=hh, minutes=mm)
    except Exception:
        return None


def find_candidates(
    lon: float,
    lat: float,
    t_center: datetime.datetime,
    *,
    satellites: Sequence[str] = MODIS_SATELLITES,
    instrument: Optional[str] = None,
    window_min: float = 45.0,
    verbose: bool = True,
) -> List[OverpassCandidate]:
    """Search `satellites` for granules overlapping (lon, lat) within
    +/-window_min of t_center. Returns candidates sorted by |offset|
    (closest in time first). Needs er3t importable; does its own import
    so this module can be imported even before er3t is installed (e.g.
    to unit-test the parts that don't need it).

    `instrument=None` (the default) looks up each satellite's instrument
    from `_INSTRUMENT_OF_SATELLITE`, so a mixed `satellites=("terra",
    "snpp", ...)` list works in one call; pass an explicit instrument to
    force it for every satellite in the list instead (e.g. in tests)."""
    from er3t.util import get_satfile_tag

    date = datetime.datetime(t_center.year, t_center.month, t_center.day)
    win_start = t_center - datetime.timedelta(minutes=window_min)
    win_end = t_center + datetime.timedelta(minutes=window_min)

    candidates: List[OverpassCandidate] = []
    for sat in satellites:
        inst = instrument if instrument is not None else _INSTRUMENT_OF_SATELLITE.get(sat)
        if inst is None:
            raise ValueError(
                "find_candidates: don't know the instrument for satellite %r -- "
                "pass instrument= explicitly or add it to _INSTRUMENT_OF_SATELLITE" % sat
            )
        try:
            tags = get_satfile_tag(
                date,
                np.array([lon], dtype=np.float64),
                np.array([lat], dtype=np.float64),
                satellite=sat,
                instrument=inst,
                start_dt_hhmm=win_start,
                end_dt_hhmm=win_end,
            )
        except Exception as exc:
            if verbose:
                print("Message [find_candidates]: %s/%s search failed (%r)" % (sat, inst, exc))
            continue

        for tag in tags:
            gt = _tag_to_datetime(tag)
            offset = None if gt is None else (gt - t_center).total_seconds() / 60.0
            candidates.append(OverpassCandidate(sat, inst, tag, gt, offset))

    candidates.sort(key=lambda c: abs(c.offset_minutes) if c.offset_minutes is not None else 1.0e9)
    return candidates


def fetch_granule(
    candidate: OverpassCandidate,
    out_dir: str,
    *,
    want_l1b: bool = True,
    want_l2: bool = True,
    l1b_product: str = "l1b_hkm",
    verbose: bool = True,
) -> GranuleFiles:
    """Download (or pick up from cache -- see LAGRANGE_ER3T_PATCH.md)
    just the files needed for one candidate granule: geolocation always,
    L1B radiance and/or the L2 cloud product if asked for.

    Dispatches on `candidate.instrument` ('modis' or 'viirs'; see this
    module's docstring). `l1b_product` ('l1b_hkm'/'l1b_qkm') only applies
    to MODIS -- VIIRS always uses the 750 m ("MOD") L1B product, there
    being only the one 750 m-resolution option wired up here."""
    from er3t.util import download_laads_https

    os.makedirs(out_dir, exist_ok=True)
    date = candidate.granule_time or datetime.datetime.now()

    if candidate.instrument == "modis":
        prefix = _MODIS_FILE_PREFIX[candidate.satellite]

        def _dl(dataset_key):
            tag = _MODIS_DATASET[dataset_key].format(prefix=prefix)
            files = download_laads_https(date, tag, candidate.filename_tag, fdir_out=out_dir, verbose=verbose)
            return files[0] if files else None

        geo_file = _dl("geo")
        l1b_file = _dl(l1b_product) if want_l1b else None
        l2_file = _dl("l2_cloud") if want_l2 else None

    elif candidate.instrument == "viirs":
        try:
            tags = _VIIRS_DATASET[candidate.satellite]
        except KeyError:
            raise NotImplementedError(
                "fetch_granule: no VIIRS dataset tags for satellite %r -- known VIIRS "
                "platforms are %s" % (candidate.satellite, sorted(_VIIRS_DATASET))
            )

        def _dl(dataset_key):
            files = download_laads_https(
                date, tags[dataset_key], candidate.filename_tag, fdir_out=out_dir, verbose=verbose
            )
            return files[0] if files else None

        geo_file = _dl("geo")
        l1b_file = _dl("l1b") if want_l1b else None
        l2_file = _dl("l2_cloud") if want_l2 else None

    else:
        raise NotImplementedError(
            "fetch_granule: don't know how to download instrument=%r (only 'modis' "
            "and 'viirs' are wired up)" % candidate.instrument
        )

    return GranuleFiles(candidate=candidate, geo_file=geo_file, l1b_file=l1b_file, l2_file=l2_file)


def _haversine_nearest(lon_grid: np.ndarray, lat_grid: np.ndarray, lon0: float, lat0: float):
    """Index (into the flattened grid) and great-circle distance (km) of
    the nearest point in lon_grid/lat_grid to (lon0, lat0)."""
    lat_rad = np.radians(lat_grid)
    lon_rad = np.radians(lon_grid)
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    dlat = lat0_rad - lat_rad
    dlon = lon0_rad - lon_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_rad) * np.cos(lat0_rad) * np.sin(dlon / 2.0) ** 2
    dist_km = 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    idx_flat = np.nanargmin(dist_km)
    return idx_flat, float(dist_km.flat[idx_flat])


def read_granule_point(
    granule: GranuleFiles,
    lon: float,
    lat: float,
    *,
    max_dist_km: float = 5.0,
) -> PointSample:
    """Read a granule's L2 cloud product at the pixel nearest (lon, lat).

    Returns a PointSample with NaNs for anything that couldn't be read
    (missing file, or nearest pixel farther than max_dist_km -- which
    normally means the granule's swath didn't actually cover the point,
    even though get_satfile_tag said its footprint should).

    MODIS: cwp/cot come from the native (1 km) MOD06 grid (COP_FLAG
    retrieval -- see this module's docstring), cth/ctt from its separate,
    coarser 5 km companion grid (see PointSample's docstring). VIIRS:
    CLDPROP_L2_VIIRS carries cwp/cot/cth/ctt all natively at 750 m, from
    the SAME grid/file section (see `_read_viirs_cop_grid`) -- no
    separate coarse grid the way MODIS has. vza (viewing/sensor zenith
    angle) is read for both platforms off the same native grid as cot/cwp
    (MOD03's own SensorZenith for MODIS, CLDPROP's geolocation_data/
    sensor_zenith for VIIRS) -- see PointSample's docstring.

    Reads cot/cwp/cth/ctt straight off the HDF4/netCDF SDs via
    `_read_modis_cop_grid`/`_read_modis_ctt_cth_grid`/`_read_viirs_cop_grid`
    rather than through er3t's `modis_l2`/`viirs_cldprop_l2` -- see this
    module's docstring (the modis_l2 cwp-assembly bug) and those
    functions' own docstrings."""
    candidate = granule.candidate
    sample = PointSample(candidate=candidate, distance_km=float("nan"))

    if granule.geo_file is None or granule.l2_file is None:
        return sample

    if candidate.instrument == "modis":
        from er3t.util.modis import modis_03

        f03 = modis_03(fnames=[granule.geo_file], keep_dims=True)
        cot_grid, cwp_grid, lon_grid, lat_grid = _read_modis_cop_grid(granule.l2_file, f03)
        ctt_grid, cth_grid, lon_5km, lat_5km = _read_modis_ctt_cth_grid(granule.l2_file)
        # f03's own vza is already at the same native 1 km grid as lon_grid/
        # lat_grid above (modis_03 reads SensorZenith off the same MOD03
        # file), so it shares idx/dist_km with cot/cwp -- no separate
        # nearest-pixel lookup needed.
        vza_grid = np.asarray(f03.data["vza"]["data"], dtype=float)

        idx, dist_km = _haversine_nearest(lon_grid, lat_grid, lon, lat)
        idx_5km, dist_5km = _haversine_nearest(lon_5km, lat_5km, lon, lat)

        sample.distance_km = dist_km
        if dist_km <= max_dist_km:
            sample.cwp = float(cwp_grid.flat[idx])
            sample.cot = float(cot_grid.flat[idx])
            sample.vza = float(vza_grid.flat[idx])
        if dist_5km <= max_dist_km:
            sample.cth = float(cth_grid.flat[idx_5km])
            sample.ctt = float(ctt_grid.flat[idx_5km])

    elif candidate.instrument == "viirs":
        cot_grid, cwp_grid, cth_grid, ctt_grid, vza_grid, lon_grid, lat_grid = _read_viirs_cop_grid(granule.l2_file)

        idx, dist_km = _haversine_nearest(lon_grid, lat_grid, lon, lat)
        sample.distance_km = dist_km
        if dist_km <= max_dist_km:
            sample.cwp = float(cwp_grid.flat[idx])
            sample.cot = float(cot_grid.flat[idx])
            sample.cth = float(cth_grid.flat[idx])
            sample.ctt = float(ctt_grid.flat[idx])
            sample.vza = float(vza_grid.flat[idx])

    else:
        raise NotImplementedError("read_granule_point: unsupported instrument %r" % candidate.instrument)

    return sample


@dataclasses.dataclass
class RegionSample:
    """Satellite LWP (cwp) statistics over a small area around a trajectory
    point, rather than a single nearest pixel.

    A trajectory lon/lat represents a whole model grid cell (the ARCSIX
    tracer files are 5 km resolution), not one ~1 km MODIS pixel -- so
    comparing the model value against the *distribution* of satellite
    pixels actually inside that cell is a more honest comparison than
    picking whichever single pixel happens to be nearest (what
    read_granule_point/PointSample do, and still useful for a quick
    does-this-granule-cover-the-point check, but not for the science
    comparison itself).
    """

    candidate: OverpassCandidate
    radius_km: float
    n_pixels: int
    cwp_values: np.ndarray  # raw per-pixel values within the region, for a histogram
    cwp_mean: float = np.nan
    cwp_median: float = np.nan
    cwp_std: float = np.nan


def _select_within_radius(
    lon_grid: np.ndarray, lat_grid: np.ndarray, lon0: float, lat0: float, radius_km: float
) -> np.ndarray:
    """Boolean mask, same shape as lon_grid/lat_grid, of every pixel within
    `radius_km` (great-circle distance) of (lon0, lat0)."""
    lat_rad = np.radians(lat_grid)
    lon_rad = np.radians(lon_grid)
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    dlat = lat0_rad - lat_rad
    dlon = lon0_rad - lon_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_rad) * np.cos(lat0_rad) * np.sin(dlon / 2.0) ** 2
    dist_km = 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return dist_km <= radius_km


def read_granule_region(
    granule: GranuleFiles,
    lon: float,
    lat: float,
    *,
    radius_km: float = 2.5,
) -> RegionSample:
    """Read every valid MOD06 cwp (LWP) pixel within `radius_km` of
    (lon, lat) -- the actual area one trajectory grid cell covers -- for a
    proper model-vs-satellite comparison and a per-point LWP histogram.

    `radius_km=2.5` (a 5 km-diameter region) matches the ARCSIX tracer
    files' nominal 5 km grid spacing; pass a different value for a
    trajectory source with a different grid.

    Works for both MODIS (MOD06 cwp, ~1 km native) and VIIRS (CLDPROP cwp,
    ~750 m native) granules, dispatching on `candidate.instrument`. Reads
    the COP_FLAG retrieval straight off the HDF4/netCDF SDs via
    `_read_modis_cop_grid`/`_read_viirs_cop_grid` rather than through
    er3t's `modis_l2`/`viirs_cldprop_l2` -- see this module's docstring
    (the modis_l2 cwp-assembly bug) and those functions' own docstrings.
    """
    candidate = granule.candidate
    empty = RegionSample(candidate=candidate, radius_km=radius_km, n_pixels=0, cwp_values=np.array([]))

    if granule.geo_file is None or granule.l2_file is None:
        return empty

    if candidate.instrument == "modis":
        from er3t.util.modis import modis_03

        f03 = modis_03(fnames=[granule.geo_file], keep_dims=True)
        _cot_grid, cwp_grid, lon_grid, lat_grid = _read_modis_cop_grid(granule.l2_file, f03)

    elif candidate.instrument == "viirs":
        _cot_grid, cwp_grid, _cth_grid, _ctt_grid, _vza_grid, lon_grid, lat_grid = _read_viirs_cop_grid(granule.l2_file)

    else:
        raise NotImplementedError("read_granule_region: unsupported instrument %r" % candidate.instrument)

    mask = _select_within_radius(lon_grid, lat_grid, lon, lat, radius_km)
    values = np.asarray(cwp_grid)[mask]
    values = values[np.isfinite(values) & (values >= 0.0)]
    if values.size == 0:
        return empty

    return RegionSample(
        candidate=candidate,
        radius_km=radius_km,
        n_pixels=int(values.size),
        cwp_values=values,
        cwp_mean=float(np.mean(values)),
        cwp_median=float(np.median(values)),
        cwp_std=float(np.std(values)),
    )


# False-color (bands 7-2-1), matching the old trajectory2satellite.py
# scripts -- good for picking out cloud phase/optical thickness. True-color
# approximates natural color (not full MODIS resolution: band 3, the blue
# channel, is only available at 500 m, same as the other HKM bands used
# here -- see read_granule_rgb's docstring on getting to 250 m).
BANDS_FALSE_COLOR = (2130.0, 860.0, 650.0)
BANDS_TRUE_COLOR = (650.0, 555.0, 470.0)
# Snow/ice-vs-cloud false color (bands 3-6-7), matching NASA Worldview's
# "Corrected Reflectance (Bands367)" layer: R=band 3 (470 nm, visible
# blue), G=band 6 (1640 nm, SWIR), B=band 7 (2130 nm, SWIR2) -- all three
# already carried by the same HKM L1B file BANDS_TRUE_COLOR/
# BANDS_FALSE_COLOR read from, so this needs no new download. Snow/ice is
# bright in the visible blue band but strongly absorbing at both SWIR
# wavelengths, so it reads red-dominant here; clouds (especially liquid
# water ones, less absorbing in the SWIR than ice) stay brighter across
# all three bands, reading white-to-yellow instead -- the opposite of
# BANDS_FALSE_COLOR's bands-7-2-1, where snow/ice reads cyan/blue. Sebastian,
# 2026-11-18, requested as an alternative to 7-2-1 for the global panel
# specifically to make snow/ice easier to tell apart from cloud at a glance
# (see run_lagrange_tracking.py's --false-color-scheme).
BANDS_SNOW_ICE = (470.0, 1640.0, 2130.0)
# Keyed lookup for --false-color-scheme/config's false_color_scheme --
# kept here (not in run_lagrange_tracking.py) so any other caller can
# resolve the same short name without re-importing argparse plumbing.
FALSE_COLOR_SCHEMES = {"721": BANDS_FALSE_COLOR, "367": BANDS_SNOW_ICE}


def _nearest_fill_2d(arr) -> np.ndarray:
    """Fill non-finite (NaN/Inf) or masked entries of a 2D array with the
    nearest finite neighbor's value.

    MOD03 geolocation/angle arrays carry real fill values at a handful of
    pixels (bow-tie-deleted scan edges, instrument calibration views) --
    `lon`/`lat` can hold raw out-of-range sentinels (not NaN, since er3t
    only NaN-fills sza/saa/vza/vaa, not lon/lat), and `sza` can genuinely be
    NaN. Left alone, a few bad geolocation pixels poison er3t's
    orthographic-projection-based upscale_modis_lonlat() far beyond their
    own location (linear interpolation treats them as real neighbors), and
    matplotlib's pcolormesh refuses ANY non-finite coordinate outright. This
    is a cheap patch at the point of use, not a fix to er3t itself -- see
    LAGRANGE_ER3T_PATCH.md if a future er3t version NaN-fills/masks lon/lat
    itself and this becomes redundant.
    """
    arr = np.ma.filled(arr, np.nan) if np.ma.isMaskedArray(arr) else np.asarray(arr, dtype=float)
    bad = ~np.isfinite(arr)
    if not bad.any():
        return arr
    if bad.all():
        raise ValueError("_nearest_fill_2d: every value is non-finite, nothing to fill from")
    from scipy import ndimage

    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return arr[tuple(idx)]


def read_granule_rgb(granule: GranuleFiles, bands=BANDS_FALSE_COLOR):
    """Read a reflectance composite for rendering a wide-area frame.
    `bands` is (wavelength_nm, wavelength_nm, wavelength_nm) mapped to
    (R, G, B) by nearest match among the bands the L1B file carries --
    pass BANDS_TRUE_COLOR for a natural-color look instead of the default
    false color (bands 7-2-1, matching the old trajectory2satellite.py
    scripts -- better at distinguishing cloud phase/optical thickness).
    Dispatches on `candidate.instrument` ('modis' or 'viirs').

    Returns (rgb, lon, lat) or (None, None, None) if the L1B/geo files
    aren't available.

    MODIS: three details that are easy to get wrong here (all three bit us
    once against real granules -- see LAGRANGE_ER3T_PATCH.md's sibling note
    in this docstring if this breaks again against a different product):

    - er3t's modis_l1b, with keep_dims=True, returns 'ref'/'rad'/'cnt' as
      (n_bands, ny, nx) -- band is the FIRST axis, not the last.
    - modis_03's sza/geolocation are always at MOD03's native resolution
      (1 km), while the L1B product read here (HKM -> 500 m by default,
      QKM -> 250 m) is finer. l1b.data['lon']/['lat'] are already upscaled
      to match the L1B grid (that's what modis_l1b does internally), but
      sza is not, so it's upscaled here by simple pixel replication --
      precise sub-pixel registration doesn't matter for a cos(sza)
      reflectance normalization used only for display.
    - a handful of real fill/edge pixels in the raw geolocation can carry
      through er3t's upscaling as NaN, which matplotlib's pcolormesh
      refuses outright -- see _nearest_fill_2d().

    Resolution note: true color here is capped at 500 m because the blue
    channel (band 3) only exists in the HKM product. Getting to the full
    250 m MODIS resolution for the zoomed panel would mean also fetching
    the QKM product (l1b_product='l1b_qkm' in fetch_granule -- 250 m
    red/NIR only) and pan-sharpening the 500 m blue/green bands against
    the 250 m red band; not implemented yet.

    VIIRS: uses the 750 m VNP02MOD/VJ1/VJ202MOD L1B product and its own
    VNP03MOD/VJ1/VJ203MOD geolocation, which already share one native grid
    (no MODIS-style sza upscaling needed). `viirs_l1b` is deliberately
    called with `f03=None` -- see this module's docstring for the upstream
    bug that combination of `keep_dims=True` + a given `f03` triggers --
    and with `auto_mask=False`, for an unrelated second upstream bug: with
    the default `auto_mask=True`, flagged/bow-tie-deleted pixels are
    supposed to come back as NaN via er3t's `_mask_flags()`, but that
    method's flag loop discards every prior iteration's masking except the
    last flag checked, and its own `.filled()` call result is never
    assigned/returned -- so flagged pixels leak through as raw,
    unconverted sentinel values instead. Visually this showed up as a
    dense, regular dot/diagonal-stripe texture across the whole composite
    (VIIRS flags the same fixed cross-track sample positions on every
    scan, so the leaked garbage forms a repeating pattern, not isolated
    pixels). `auto_mask=False` routes through `_remove_flags()` instead,
    which has no such bug and correctly replaces every flagged pixel with
    real NaN before scaling -- those pixels then simply don't get drawn by
    pcolormesh (a small gap) rather than rendering as noise.
    """
    if granule.geo_file is None or granule.l1b_file is None:
        return None, None, None

    instrument = granule.candidate.instrument

    if instrument == "modis":
        from er3t.util.modis import modis_03, modis_l1b

        f03 = modis_03(fnames=[granule.geo_file], keep_dims=True)
        # sanitize f03's lon/lat IN PLACE, before modis_l1b ever sees them --
        # modis_l1b upscales these same arrays internally (self.f03.data[...]),
        # so this is the one place to fix fill-value pixels for both readers.
        f03.data["lon"]["data"] = _nearest_fill_2d(f03.data["lon"]["data"])
        f03.data["lat"]["data"] = _nearest_fill_2d(f03.data["lat"]["data"])

        l1b = modis_l1b(fnames=[granule.l1b_file], f03=f03, keep_dims=True)

        ref_all = l1b.data["ref"]["data"]  # (n_bands, ny, nx) at the L1B product's native resolution
        wvl_all = np.asarray(l1b.data["wvl"]["data"])
        sza = _nearest_fill_2d(f03.data["sza"]["data"])  # (ny_1km, nx_1km) -- coarser than ref_all unless l1b is itself 1km

        scale_y = max(1, ref_all.shape[-2] // sza.shape[0])
        scale_x = max(1, ref_all.shape[-1] // sza.shape[1])
        sza_up = np.repeat(np.repeat(sza, scale_y, axis=0), scale_x, axis=1) if (scale_y > 1 or scale_x > 1) else sza
        # upscale-by-replication can land a row/col short of ref_all's native
        # shape (e.g. an odd dimension) -- crop both to whatever's common
        ny = min(sza_up.shape[0], ref_all.shape[-2])
        nx = min(sza_up.shape[1], ref_all.shape[-1])
        sza_up = sza_up[:ny, :nx]

        panels = []
        for wvl in bands:
            i = int(np.argmin(np.abs(wvl_all - wvl)))
            band = ref_all[i, :ny, :nx] / np.cos(np.radians(sza_up))
            vmax = np.nanpercentile(band, 99)
            band = np.clip(band, 0.0, vmax) / vmax
            panels.append(band)

        rgb = np.stack(panels, axis=-1)
        # l1b's own lon/lat are re-derived (upscaled) from f03's already-cleaned
        # geolocation above, but sanitize again defensively -- upscale_modis_lonlat's
        # interpolation can still introduce its own NaN at the extreme edge rows/cols.
        lon = _nearest_fill_2d(l1b.data["lon"]["data"][:ny, :nx])
        lat = _nearest_fill_2d(l1b.data["lat"]["data"][:ny, :nx])
        return rgb, lon, lat

    elif instrument == "viirs":
        from er3t.util.viirs import VIIRS_L1B_MOD_BANDS, viirs_03, viirs_l1b

        f03 = viirs_03(fnames=[granule.geo_file], keep_dims=True)
        lon = _nearest_fill_2d(f03.data["lon"]["data"])
        lat = _nearest_fill_2d(f03.data["lat"]["data"])
        sza = _nearest_fill_2d(f03.data["sza"]["data"])

        # nearest-wavelength VIIRS M-band (by name) for each requested wavelength
        band_names = list(VIIRS_L1B_MOD_BANDS.keys())
        band_wvls = np.array([VIIRS_L1B_MOD_BANDS[n] for n in band_names], dtype=float)
        chosen = [band_names[int(np.argmin(np.abs(band_wvls - wvl)))] for wvl in bands]

        # f03=None sidesteps the keep_dims=True + f03 bug noted in this
        # module's docstring -- VNP02MOD/VNP03MOD already share one native
        # 750 m grid, so no f03-driven cropping is needed to line them up.
        l1b = viirs_l1b(fnames=[granule.l1b_file], f03=None, bands=chosen, keep_dims=True, auto_mask=False)
        ref_all = l1b.data["ref"]["data"]  # (n_bands, ny, nx), same grid as f03's lon/lat/sza

        ny = min(ref_all.shape[-2], lon.shape[0], sza.shape[0])
        nx = min(ref_all.shape[-1], lon.shape[1], sza.shape[1])

        panels = []
        for i in range(len(chosen)):
            band = ref_all[i, :ny, :nx] / np.cos(np.radians(sza[:ny, :nx]))
            vmax = np.nanpercentile(band, 99)
            band = np.clip(band, 0.0, vmax) / vmax
            panels.append(band)

        rgb = np.stack(panels, axis=-1)
        return rgb, lon[:ny, :nx], lat[:ny, :nx]

    else:
        raise NotImplementedError("read_granule_rgb: unsupported instrument %r" % instrument)


def _read_modis_cop_grid(l2_file: str, f03, cop_flag: str = COP_FLAG) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read MOD06/MYD06 cloud optical thickness AND water path as
    native-shape 2D (~1 km) grids, using the companion MOD03 file's own
    native 1 km geolocation for lon/lat (no upscaling needed, unlike
    modis_l2's own f03=None fallback path).

    This duplicates (rather than calls into) er3t's `modis_l2`, for two
    independent reasons (see this module's docstring for both in detail):
    (1) `modis_l2.read()` unconditionally boolean-indexes every cop field
    by a 1 km validity mask (`arr[logic_1km]`) -- there's no `keep_dims`
    knob on `modis_l2` at all, so it always flattens to 1D, useless for a
    mosaic panel that needs the real 2D grid; (2) `modis_l2.read()`'s own
    `cwp` assembly has a copy-paste bug (initializes from `cer0_data`
    instead of `cwp0_data`), so even its flattened `cwp` can't be trusted
    for read_granule_point/read_granule_region either. So this function
    is now the ONE place both read_granule_point/read_granule_region and
    read_granule_field_grid get MODIS cot/cwp from -- it reads the
    underlying HDF4 SDs directly and applies the same invalid/clear-sky/
    PCL-fill logic `modis_l2.read()` intends (cwp merged correctly, unlike
    upstream), just without the shape-destroying index.
    (See `_read_viirs_cop_grid` -- same approach for VIIRS.)

    `cop_flag` selects the spectral-pair retrieval (see `_cop_vname`/this
    module's docstring); `f03` must be a `modis_03` instance constructed
    with `keep_dims=True` (otherwise its own 'lon'/'lat' are already
    flattened to 1D).

    Returns (cot, cwp, lon, lat), each a native-shape 2D array; `cot`/`cwp`
    follow the same 0.0-for-clear/invalid, NaN-only-for-truly-missing
    convention documented on `read_granule_field_grid`.
    """
    from pyhdf.SD import SD, SDC
    from er3t.util import get_data_h4

    var_names = [
        "Cloud_Phase_Optical_Properties",
        _cop_vname("Cloud_Optical_Thickness", cop_flag), _cop_vname("Cloud_Effective_Radius", cop_flag),
        _cop_vname("Cloud_Water_Path", cop_flag),
        _cop_vname("Cloud_Optical_Thickness", cop_flag, pcl=True), _cop_vname("Cloud_Effective_Radius", cop_flag, pcl=True),
        _cop_vname("Cloud_Water_Path", cop_flag, pcl=True),
    ]
    _log_vars_read(l2_file, var_names)

    f = SD(l2_file, SDC.READ)
    try:
        ctp = get_data_h4(f.select(var_names[0]))
        cot0 = get_data_h4(f.select(var_names[1]))
        cer0 = get_data_h4(f.select(var_names[2]))
        cwp0 = get_data_h4(f.select(var_names[3]))
        cot1 = get_data_h4(f.select(var_names[4]))
        cer1 = get_data_h4(f.select(var_names[5]))
        cwp1 = get_data_h4(f.select(var_names[6]))
    finally:
        f.end()

    cot = cot0.copy()
    cwp = cwp0.copy()

    logic_invalid = (cot0 < 0.0) | (cer0 < 0.0) | (cwp0 < 0.0) | (ctp == 0)
    cot[logic_invalid] = 0.0
    cwp[logic_invalid] = 0.0

    logic_clear = ((cot0 == 0.0) | (cer0 == 0.0) | (cwp0 == 0.0)) & (ctp == 1)
    cot[logic_clear] = 0.0
    cwp[logic_clear] = 0.0

    logic_pcl = ((cot0 == 0.0) | (cer0 == 0.0) | (cwp0 == 0.0)) & \
        ((cot1 > 0.0) & (cer1 > 0.0) & (cwp1 > 0.0))
    cot[logic_pcl] = cot1[logic_pcl]
    cwp[logic_pcl] = cwp1[logic_pcl]

    lon = np.asarray(f03.data["lon"]["data"], dtype=float)
    lat = np.asarray(f03.data["lat"]["data"], dtype=float)
    return cot, cwp, lon, lat


def _read_modis_ctt_cth_grid(l2_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read MOD06/MYD06 cloud top temperature AND height as native-shape
    2D (~5 km) grids, together with their own embedded 5 km lon/lat -- see
    `_read_modis_cop_grid`'s docstring for why this reads the HDF4 SD
    directly rather than through er3t's `modis_l2` (same
    shape-destroying unconditional indexing, this time in `read_vars()`).
    Neither field is part of the band-pair retrieval family, so there's no
    `cop_flag` here -- same plain fields regardless of `COP_FLAG`.
    """
    from pyhdf.SD import SD, SDC
    from er3t.util import get_data_h4

    _log_vars_read(l2_file, ["Longitude", "Latitude", "Cloud_Top_Temperature", "Cloud_Top_Height"])

    f = SD(l2_file, SDC.READ)
    try:
        lon = np.asarray(get_data_h4(f.select("Longitude")), dtype=float)
        lat = np.asarray(get_data_h4(f.select("Latitude")), dtype=float)
        ctt = np.asarray(get_data_h4(f.select("Cloud_Top_Temperature")), dtype=float)
        cth = np.asarray(get_data_h4(f.select("Cloud_Top_Height")), dtype=float)
    finally:
        f.end()
    return ctt, cth, lon, lat


def _read_viirs_cop_grid(
    l2_file: str, cop_flag: str = COP_FLAG
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read VIIRS CLDPROP cloud optical thickness, water path, top height,
    AND top temperature as native-shape 2D grids, plus their own 2D
    lon/lat (carried in the CLDPROP file's own `geolocation_data` group).

    This duplicates (rather than calls into) er3t's `viirs_cldprop_l2`,
    for the VIIRS-side analog of both reasons `_read_modis_cop_grid`
    duplicates `modis_l2` (see its docstring): (1) `read_cop()`
    unconditionally boolean-indexes every field by a lon/lat validity
    mask, flattening to 1D regardless of its own `keep_dims` flag; (2)
    `viirs_cldprop_l2` has no `cop_flag`-equivalent parameter at all, so
    there is no way to ask it for the `_1621` (or `_16`/`_37`) retrieval
    the way `modis_l2` supports via `cop_flag=`. So this function is now
    the ONE place both read_granule_point/read_granule_region and
    read_granule_field_grid get VIIRS cot/cwp/cth/ctt from -- it reads the
    same underlying netCDF variables directly and applies the same
    invalid/clear-sky/PCL-fill logic `read_cop()` does (see its source in
    er3t.util.viirs), just without the shape-destroying final index, and
    parameterized by `cop_flag` like the MODIS side.

    Returns (cot, cwp, cth, ctt, vza, lon, lat), each a native-shape 2D
    array. `cot`/`cwp` are 0.0 for a pixel CLDPROP's own quality flags mark
    clear-sky or an invalid retrieval (matching read_cop's logic) and the
    real retrieved value for a valid/PCL-filled one; a pixel masked out by
    the file's own declared `_FillValue` (genuinely missing, e.g. outside
    the scan) comes back as float NaN -- a gap we leave unpainted rather
    than claim as "clear" -- not 0.0. `cth`/`ctt` are each read plain (not
    part of the band-pair retrieval family, no PCL variant) with their own
    NaN-for-fill handling -- unlike MODIS, VIIRS carries cth/ctt in this
    SAME `geophysical_data` group/file as cot/cwp (no separate coarser
    grid the way `_read_modis_ctt_cth_grid` is for MODIS), so reading them
    here in one pass is both correct and the cheaper option. `vza` (viewing/
    sensor zenith angle, degrees) comes from this same file's
    `geolocation_data` group (CLDPROP carries its own copy of the
    geolocation fields, same convention as er3t's viirs_03 reader uses for
    the standalone VNP03/VJ103 geolocation product -- see its source for
    the 'sensor_zenith' variable name), on the same native grid as
    everything else here. lon/lat are raw (not yet nearest-fill cleaned --
    the caller does that).
    """
    from netCDF4 import Dataset
    from er3t.util import get_data_nc

    def _plain(arr):
        return np.ma.filled(np.ma.asarray(arr, dtype="float64"), np.nan) if np.ma.isMaskedArray(arr) \
            else np.asarray(arr, dtype="float64")

    _log_vars_read(l2_file, [
        "geolocation_data/longitude", "geolocation_data/latitude",
        "geolocation_data/sensor_zenith",
        "geophysical_data/Cloud_Phase_Optical_Properties",
        "geophysical_data/Cloud_Top_Height", "geophysical_data/Cloud_Top_Temperature",
        "geophysical_data/" + _cop_vname("Cloud_Optical_Thickness", cop_flag),
        "geophysical_data/" + _cop_vname("Cloud_Effective_Radius", cop_flag),
        "geophysical_data/" + _cop_vname("Cloud_Water_Path", cop_flag),
    ])

    f = Dataset(l2_file, "r")
    try:
        geo = f.groups["geolocation_data"].variables
        cop = f.groups["geophysical_data"].variables
        lon = _plain(get_data_nc(geo["longitude"], replace_fill_value=None))
        lat = _plain(get_data_nc(geo["latitude"], replace_fill_value=None))
        vza = _plain(get_data_nc(geo["sensor_zenith"]))

        ctp = _plain(get_data_nc(cop["Cloud_Phase_Optical_Properties"], replace_fill_value=None))
        cth = _plain(get_data_nc(cop["Cloud_Top_Height"]))
        ctt = _plain(get_data_nc(cop["Cloud_Top_Temperature"]))
        cot0 = _plain(get_data_nc(cop[_cop_vname("Cloud_Optical_Thickness", cop_flag)]))
        cer0 = _plain(get_data_nc(cop[_cop_vname("Cloud_Effective_Radius", cop_flag)]))
        cwp0 = _plain(get_data_nc(cop[_cop_vname("Cloud_Water_Path", cop_flag)]))
        cot1 = _plain(get_data_nc(cop[_cop_vname("Cloud_Optical_Thickness", cop_flag, pcl=True)]))
        cer1 = _plain(get_data_nc(cop[_cop_vname("Cloud_Effective_Radius", cop_flag, pcl=True)]))
        cwp1 = _plain(get_data_nc(cop[_cop_vname("Cloud_Water_Path", cop_flag, pcl=True)]))
    finally:
        f.close()

    cot = cot0.copy()
    cwp = cwp0.copy()

    logic_invalid = (cot0 < 0.0) | (cer0 < 0.0) | (cwp0 < 0.0) | (ctp == 0)
    cot[logic_invalid] = 0.0
    cwp[logic_invalid] = 0.0

    logic_clear = ((cot0 == 0.0) | (cer0 == 0.0) | (cwp0 == 0.0)) & (ctp == 1)
    cot[logic_clear] = 0.0
    cwp[logic_clear] = 0.0

    logic_pcl = ((cot0 == 0.0) | (cer0 == 0.0) | (cwp0 == 0.0)) & \
        ((cot1 > 0.0) & (cer1 > 0.0) & (cwp1 > 0.0))
    cot[logic_pcl] = cot1[logic_pcl]
    cwp[logic_pcl] = cwp1[logic_pcl]

    return cot, cwp, cth, ctt, vza, lon, lat


def read_granule_field_grid(granule: GranuleFiles, field: str):
    """Read a FULL 2D grid of one L2 cloud-property field plus its own
    lon/lat grid -- for a mosaic-style imagery panel (see render.py's COT/
    CTT panels), as opposed to read_granule_point's single nearest pixel
    or read_granule_region's radius-pooled statistics.

    `field` is 'cot' (cloud optical thickness -- native ~1 km MODIS /
    ~750 m VIIRS grid, same one cwp/cot come from in read_granule_region)
    or 'ctt' (cloud top temperature -- MODIS only for this per-frame
    mosaic panel, off its own coarser 5 km grid, mirroring
    read_granule_point's cth/ctt handling. VIIRS's CLDPROP does carry its
    own Cloud_Top_Temperature field -- `_read_viirs_cop_grid` now reads
    it for read_granule_point/read_granule_region's point-sample/
    time-series use -- but it isn't plumbed into this mosaic-grid path,
    so this still returns (None, None, None) for any VIIRS candidate).

    Returns (values, lon, lat), or (None, None, None) if unavailable
    (missing file, or ctt requested for VIIRS). lon/lat are nearest-fill
    cleaned (see _nearest_fill_2d) since this is meant for pcolormesh,
    which refuses on any non-finite coordinate. `values` follows the L2
    product's own convention of 0.0 for a pixel flagged clear-sky or an
    invalid retrieval and the real value for a valid/PCL-filled one;
    float NaN is reserved for a pixel the file's own fill value masks
    outright (e.g. off the edge of the scan) -- a gap left unpainted
    rather than claimed as "clear" -- so pcolormesh just leaves those
    pixels untouched rather than refusing on them.

    Note: this reads the underlying HDF4/netCDF SDs directly rather than
    through er3t's modis_l2/viirs_cldprop_l2 -- see _read_modis_cop_grid/
    _read_viirs_cop_grid's docstrings for why (both of those readers
    unconditionally flatten every field to 1D, with no way to ask for the
    native 2D grid this needs; MODIS additionally works around a
    cwp-assembly bug in modis_l2 itself -- see this module's docstring).
    cot/cwp (both band-pair retrievals) use COP_FLAG (currently '1621');
    ctt/cth are plain fields, unaffected by that choice.
    """
    if field not in ("cot", "ctt"):
        raise ValueError("read_granule_field_grid: field must be 'cot' or 'ctt', got %r" % field)

    candidate = granule.candidate
    if granule.geo_file is None or granule.l2_file is None:
        return None, None, None

    if candidate.instrument == "modis":
        if field == "cot":
            from er3t.util.modis import modis_03

            f03 = modis_03(fnames=[granule.geo_file], keep_dims=True)
            values, _cwp, lon, lat = _read_modis_cop_grid(granule.l2_file, f03)
        else:
            values, _cth, lon, lat = _read_modis_ctt_cth_grid(granule.l2_file)
        lon = _nearest_fill_2d(lon)
        lat = _nearest_fill_2d(lat)
        return values, lon, lat

    elif candidate.instrument == "viirs":
        if field == "ctt":
            return None, None, None
        values, _cwp, _cth, _ctt, _vza, lon, lat = _read_viirs_cop_grid(granule.l2_file)
        return values, _nearest_fill_2d(lon), _nearest_fill_2d(lat)

    else:
        raise NotImplementedError("read_granule_field_grid: unsupported instrument %r" % candidate.instrument)
