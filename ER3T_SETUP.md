# er3t setup for fetch_satellite_overlays.py

`fetch_satellite_overlays.py` (and the `satellite.py` it imports) needs
[er3t](https://github.com/hong-chen/er3t) and a NASA Earthdata token. This
is the one real prerequisite that makes this piece of the repo more than a
`pip install` -- everything else (`compare_insitu_cloud_properties.py`
itself) only needs numpy/pandas/scipy/matplotlib.

## 1. Clone er3t's `libera-l2-fixes` branch

Two independent upstream er3t bugs used to make this unusable against real
data:

- **`download_laads_https()` caching bug** (`er3t/util/daac.py`): silently
  dropped a file from its *returned* list whenever that file was already
  cached on disk -- a caller re-running against previously-downloaded
  granules (every repeat run against the same date/track, once anything
  is cached) saw an incomplete list back with no way to tell a genuine
  cache hit from a failed download.
- **`viirs_cldprop_l2.read_cop()` crash** (`er3t/util/viirs.py`): crashed
  with `IndexError: too many indices for array` against any real VIIRS
  CLDPROP granule (SNPP/NOAA-20/NOAA-21) -- a leftover double-indexing
  bug. Terra/Aqua (MODIS) were unaffected.

Both are now fixed directly on er3t's `libera-l2-fixes` branch (maintained
by the Libera team, also used in production by `libera_cam_l2`), so no
local patching is needed -- just clone that branch instead of `main`:

```bash
cd ~/software   # wherever you keep clones of third-party tools/dependencies
git clone -b libera-l2-fixes https://github.com/hong-chen/er3t.git er3t_arcsix
cd er3t_arcsix
pip install -e .
```

(Run the `pip install -e .` in whichever conda/venv environment you use
for this.) If something looks off, check `git log` on that branch for
anything that's touched `daac.py` or `viirs.py` since -- it's actively
maintained, so later commits may have moved things further.

`satellite.py` (vendored into this repo) also works around three further
MODIS/VIIRS reader issues in er3t entirely from the calling side -- no
patch needed for those, see its own module docstring if you're curious.

## 2. Get a NASA Earthdata token

Register at [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov) if
you don't already have an account, then generate a token under
"Generate Token" in your profile. Set it in the environment before
running `fetch_satellite_overlays.py`:

```bash
export EARTHDATA_TOKEN=...          # bash
setenv EARTHDATA_TOKEN ...          # csh/tcsh
```

## 3. Additional Python dependencies

Beyond `compare_insitu_cloud_properties.py`'s own numpy/pandas/scipy/
matplotlib, reading the actual satellite files needs:

```bash
pip install pyhdf netCDF4
```

(`pyhdf` reads the HDF4 MODIS L2 files; `netCDF4` reads the netCDF4 VIIRS
CLDPROP files.)

## Once this is done

```bash
EARTHDATA_TOKEN=... python3 fetch_satellite_overlays.py \
    --p3-nav ARCSIX-MetNav_P3B_20240607_R0.ict \
    --date 2024-06-07 --t0 16:18:43 --t1 16:46:00 \
    --out-dir satellite_results/
```

writes one `.npz` per covering satellite platform into `satellite_results/`,
ready for `compare_insitu_cloud_properties.py --sat-dir satellite_results/`.
