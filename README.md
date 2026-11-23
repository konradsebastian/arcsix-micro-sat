# arcsix-micro-sat

Standalone reproduction of Sebastian Schmidt's (LASP/CU Boulder) in-situ
LWC/IWC/LWP/IWP/COT computation for the RF07 (2024-06-07) "wall" transect,
16:18:43-16:46 UTC, shared so Paul Lawson and Sebastian Becker can run and
check it directly against their own numbers.

No satellite, AVIRIS, or lidar (HALO/HSRL) data is required. Those are
supported as optional overlays if you want to plug your own along-track
traces in (see the top of the script), but the core comparison -- LWC,
IWC, LWP, IWP, COT -- only needs the three standard ARCSIX P-3 in-situ
files listed below.

## What this computes

- **LWC**: SEA WCM-2000 hot-wire probe, column `Lwc1_wcm42` (2.0 mm
  sensing element).
- **IWC**: SPEC HVPS4 optical-array probe, column `iwc`
  (`CldMicro_IWC_InSitu_Optical_Drop_AMB`) -- a direct area/mass-derived
  IWC from the measured ice size distribution.
- **Cloud mask**: TWC (LWC+IWC) > 0.04 g/m3. This is purely
  microphysical -- there is no independent geometric cloud-base cutoff
  (see the note on `EXCLUDE_LEGS` below).
- **Legs/boxes**: one box per monotonic ascent or descent segment of the
  P-3's own GPS altitude trace (`scipy.signal.find_peaks` on local
  altitude maxima and minima).
- **LWP/IWP**: trapezoidal vertical integral of each leg's cloud-masked
  LWC/IWC, with two selectable methods (`--lwp-method`): `naive`
  (**default**) skips any altitude binning and integrates the raw,
  altitude-sorted 1 Hz samples directly; `binned` (legacy -- matches
  this project's original figure) mean-bins into 6 altitude bins per
  leg first and zero-fills any empty bin before integrating. These
  differ by ~15% on average in our own data (several legs only have
  4-5/6 bins populated, and zero-filling an empty bin pulls the
  integral down) -- `naive` lands much closer to Lawson's own LWP,
  which is why it is now the default. Both are included so this can
  be checked directly rather than taken on faith.
- **COT**: `tau = K_LIQ * LWP + K_ICE * IWP`, with FIXED `K_LIQ = 0.15
  m2/g` (re = 10 micron assumed) and `K_ICE = 0.04 m2/g` (re = 40 micron
  assumed, NOT independently verified). This is a fixed-Reff
  mass-extinction approach, not a retrieved per-layer Reff and not a
  directly-measured extinction -- see the note in the script's own
  docstring on how this differs from Becker's (per-layer retrieved Reff)
  and Lawson's (HVPS4's own directly-measured `ext` column, at least for
  the ice phase) methods. That difference is the main open question
  between the three analyses and is flagged, not resolved, here.

`--exclude-legs`: one leg in this specific window (leg index 12, the last
and lowest-altitude leg of the transect) has an anomalous near-zero LWP
paired with real IWP and the highest low-altitude IWC of any leg --
plausibly a below-cloud-base ice/virga signature that the purely
microphysical cloud mask does not exclude by construction. Pass
`--exclude-legs 12` to reproduce the comparison with it removed (this
also makes the leg count match Lawson's own 12-segment figure exactly).
Default is nothing excluded.

## Required input files

Three standard ARCSIX P-3 data products (ESPO/ASDC archive), for RF07
(2024-06-07):

| CLI flag | File pattern | Instrument |
|---|---|---|
| `--p3-nav` | `ARCSIX-MetNav_P3B_20240607_R*.ict` | P-3 MetNav (position, GPS altitude) |
| `--lwc` | `ARCSIX-Lwc123_P3B_20240607_R*.ict` | SEA WCM-2000 |
| `--iwc` | `ARCSIX-HVPS4V50_P3B_20240607_R*.ict` | SPEC HVPS4 |

```bash
pip install -r requirements.txt

python compare_insitu_cloud_properties.py \
    --p3-nav  ARCSIX-MetNav_P3B_20240607_R0.ict \
    --lwc     ARCSIX-Lwc123_P3B_20240607_R1.ict \
    --iwc     ARCSIX-HVPS4V50_P3B_20240607_R1.ict
```

This prints a per-leg LWP/IWP/tau table to stdout, a summary (with and
without `--exclude-legs`, if given), and writes
`lawson_becker_comparison_rf07.png`: a 4-panel figure (LWC, IWC, stacked
liquid/ice water path, optical thickness) matching the style of Lawson's
own reference figure for this transect, plus a narrow right-hand column
of density-normalized histograms (cloud-top height, water path, COT --
aircraft vs. each satellite platform, where `--sat-dir` is given). The
current run's mean LWP/IWP (the same numbers quoted in the accompanying
email) are also printed directly in the figure title.

If `--sat-dir` is given and the `.npz` files carry a `cth` array
(satellite cloud-top height, m), it's overlaid as a dashed line on the
LWC/IWC panels alongside the in-situ cloud top (implicit as the top edge
of each leg's shaded box) -- this is the comparison Lawson specifically
asked about (satellite vs. in-situ CTH).

`--lwp-method {naive,binned}` switches between the two LWP/IWP
integration methods described above (`naive` is the default) -- both
are provided so the ~15% difference between them can be checked
directly.

## Full pipeline: satellite download -> plot

The comparison figure can include a satellite CTH/CWP/COT overlay, but
getting there is two separate scripts: one to fetch+collocate the
satellite data, one to make the plot. Here's the whole thing in order.

**Step 0 -- one-time setup (only needed if you want the satellite
overlay; skip to Step 3 if you don't).** `pip install -r
requirements.txt` above already covers everything else; on top of that,
set up a patched `er3t` and a NASA Earthdata token -- this is the one
real prerequisite in this repo, so it's broken out on its own in
**`ER3T_SETUP.md`**. Short version: clone `er3t`, apply the two patches
in `patches/` (`git apply`), `pip install -e .` it, generate a token at
urs.earthdata.nasa.gov, and `export EARTHDATA_TOKEN=...` (or `setenv
EARTHDATA_TOKEN ...` in csh/tcsh). Do this once; it's not needed again
after the first successful fetch below.

**Step 1 -- fetch and collocate the satellite data:**

```bash
python3 fetch_satellite_overlays.py \
    --p3-nav  ARCSIX-MetNav_P3B_20240607_R0.ict \
    --date 2024-06-07 --t0 16:18:43 --t1 16:46:00 \
    --out-dir satellite_results/
```

This finds which MODIS/VIIRS granules overpass the RF07 wall transect
(within +-45 min by default, `--window-min` to change that), downloads
them (cached under `satellite_cache/` so re-runs don't re-download),
and nearest-neighbor-matches each pixel onto the P-3's own along-track
distance. It reuses `find_candidates()`/`fetch_granule()`/
`read_granule_point()` from `satellite.py` (vendored alongside it,
unmodified, from Sebastian's `arcsix_lagrange_tracking` project). By
default it tries all five platforms (`--satellites terra aqua snpp
noaa20 noaa21` to restrict); whichever ones actually overpass the
window write one `.npz` file each into `satellite_results/` (none is
possible -- not every window has a usable overpass).

**Step 2 -- check what came out:**

```bash
ls satellite_results/*.npz
```

If this is empty, nothing overpassed the window -- try widening
`--window-min` or `--max-dist-km`, or confirm the time window is right.

**Step 3 -- make the plot**, pointing `--sat-dir` at that same folder
(omit `--sat-dir` entirely to skip satellite overlay and only plot the
in-situ comparison):

```bash
python compare_insitu_cloud_properties.py \
    --p3-nav  ARCSIX-MetNav_P3B_20240607_R0.ict \
    --lwc     ARCSIX-Lwc123_P3B_20240607_R1.ict \
    --iwc     ARCSIX-HVPS4V50_P3B_20240607_R1.ict \
    --sat-dir satellite_results/ \
    --exclude-legs 12 \
    --lwp-method naive \
    --out comparison.png
```

That's the whole pipeline: raw P-3 + satellite inputs in, one PNG out
(`comparison.png`, or `lawson_becker_comparison_rf07.png` if `--out` is
omitted) with the in-situ LWC/IWC/water-path/COT panels, the satellite
CTH overlay (broken-axis, since satellite CTH is so much higher than
the in-situ range), and the right-hand histogram column, plus the
current run's mean LWP/IWP printed in the figure title and a per-leg
table on stdout.

`--sat-dir` itself just expects a folder of `.npz` files (one per
satellite platform, each with `dist_w`/`cwp`/`cot`/`cth`/`satellite`
arrays) -- Step 1 above is how you generate them, but if you already
have them from elsewhere in the same schema, Step 1 is skippable too.

**This needs a patched `er3t` install and a NASA Earthdata token --
see `ER3T_SETUP.md` for the full recipe and `patches/` for the two
patches themselves.** This is a real, non-trivial prerequisite (not a
pip-installable package) -- the patches are bundled here so you don't
need to come back and ask for them, but setting up `er3t` itself is
still a one-time piece of work. One of the two patches isn't optional:
without it, every VIIRS platform (SNPP/NOAA-20/NOAA-21) crashes
outright on a real granule (a genuine upstream `IndexError`, not a
style choice) -- MODIS (Terra/Aqua) is unaffected either way, so you
can still get partial results with just the first patch applied if
you're in a hurry.

AVIRIS and HALO/HSRL are intentionally not wired into either script at
all -- see `--aviris-csv`/`--lidar-csv` below if you want to plug in
your own pre-extracted traces instead; that collocation code (KDTree
nearest-neighbor search, per-instrument file formats) is real but has
nothing to do with the LWC/IWC/LWP/IWP/COT reconciliation this repo
exists for, unlike the satellite step above, which Paul/Parker asked
for directly.

## Optional overlays

Both default to omitted/skipped:

```bash
--aviris-csv aviris.csv   # columns: dist_km,lwp,cot
--lidar-csv  lidar.csv    # columns: dist_km,cth_m
```

Both files' `dist_km` should be resampled onto the script's own
along-track distance axis (printed to stdout as part of the per-leg
table) before being passed in.

## Contact

Sebastian Schmidt, LASP / University of Colorado Boulder.
