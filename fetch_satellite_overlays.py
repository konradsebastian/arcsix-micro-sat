#!/usr/bin/env python3
"""
Fetch MODIS/VIIRS satellite cloud retrievals (cwp/cot/cth/ctt/vza) along a
P-3 flight track and save them as .npz files ready for
compare_insitu_cloud_properties.py's own --sat-dir.

This is the piece that script's README originally said wasn't included:
finding which MODIS/VIIRS granules overpass the track, downloading them,
and nearest-neighbor-matching each pixel onto the P-3's own along-track
distance. It reuses find_candidates()/fetch_granule()/read_granule_point()
from satellite.py -- vendored alongside this script, unmodified, from
Sebastian Schmidt's arcsix_lagrange_tracking project -- see that file's
own module docstring for the MODIS/VIIRS reader details and the upstream
er3t bugs it works around without needing a patch. It does NOT use that
project's trajectory/rendering machinery (run_lagrange_tracking.py,
render.py) -- this script is the "turn a flight nav file into the (lon,
lat, time) point sequence those three functions expect" adapter, standing
on its own.

PREREQUISITES -- see ER3T_SETUP.md in this repo for the full recipe:
  - a patched er3t importable (the caching bug in download_laads_https()
    is patched; the viirs_cldprop_l2.read_cop() crash -- a real
    IndexError, not a style issue -- is also patched. WITHOUT the second
    patch every VIIRS platform (SNPP/NOAA-20/NOAA-21) will crash; MODIS
    (Terra/Aqua) is unaffected either way).
  - EARTHDATA_TOKEN in the environment (generate one at
    urs.earthdata.nasa.gov under "Generate Token").
  - pyhdf and netCDF4, in addition to compare_insitu_cloud_properties.py's
    own numpy/pandas/scipy/matplotlib (`pip install pyhdf netCDF4`).

*** CANNOT be test-run from a chat/sandbox environment -- actually
    downloading anything needs a real patched er3t install plus a real
    EARTHDATA_TOKEN, neither of which exist outside your own machine. ***

SATELLITE PRODUCT: MOD06_L2/MYD06_L2 (MODIS Terra/Aqua, Collection 6.1) or
CLDPROP_L2_VIIRS_SNPP/NOAA20/NOAA21 (VIIRS), reading the "_1621"
spectral-pair retrieval (1.6+2.1 um, no visible channel) for cot/cwp --
see satellite.py's own module docstring for why (poor visible-channel
cloud/surface contrast over bright snow/sea ice, exactly this domain).
Cloud_Water_Path is MOD06/CLDPROP's single dominant-phase water path
(whichever phase the cloud-phase algorithm picked per pixel), not a true
liquid+ice sum the way compare_insitu_cloud_properties.py's own in-situ
LWC+IWC is -- fine for a liquid-dominated scene, worth checking per-scene
otherwise. CTH/CTT come from MOD06's separate 5 km grid for MODIS, the
same native grid as cwp/cot for VIIRS.

Example
-------
    EARTHDATA_TOKEN=... python3 fetch_satellite_overlays.py \\
        --p3-nav  ARCSIX-MetNav_P3B_20240607_R0.ict \\
        --date 2024-06-07 --t0 16:18:43 --t1 16:46:00 \\
        --out-dir satellite_results/

Then:
    python3 compare_insitu_cloud_properties.py \\
        --p3-nav ... --lwc ... --iwc ... --sat-dir satellite_results/
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import satellite as sat  # vendored copy, alongside this script -- see its own docstring


def read_icartt(path):
    """Read a NASA ICARTT (.ict) file, auto-detecting its header-line count
    from the file's own first line (ICARTT's own convention: "N, FFI",
    data starts at line N+1) -- robust to file-revision differences,
    unlike hardcoding one skiprows value per file."""
    with open(path) as f:
        first_line = f.readline()
    n_header = int(first_line.split(",")[0].strip())
    return pd.read_csv(path, header=n_header - 1)


def load_nav(path):
    df = read_icartt(path)
    return dict(utc=df["Time_Start"].values.astype(float) / 3600.0,
                lon=df["Longitude"].values.astype(float),
                lat=df["Latitude"].values.astype(float))


def _hms_to_hours(hms):
    parts = [float(x) for x in hms.split(":")]
    while len(parts) < 3:
        parts.append(0.0)
    h, m, s = parts[:3]
    return h + m / 60.0 + s / 3600.0


def to_datetime(date, utc_hour):
    return datetime.datetime(date.year, date.month, date.day) + datetime.timedelta(hours=float(utc_hour))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--p3-nav", required=True, help="P-3 MetNav ICARTT file")
    p.add_argument("--date", required=True, help="flight date, YYYY-MM-DD")
    p.add_argument("--t0", required=True, help="window start, UTC HH:MM:SS")
    p.add_argument("--t1", required=True, help="window end, UTC HH:MM:SS")
    p.add_argument("--out-dir", default="satellite_results",
                    help="where to write one .npz per covering platform (default: satellite_results/)")
    p.add_argument("--cache-dir", default="satellite_cache",
                    help="downloaded-granule cache dir, reused across runs (default: satellite_cache/)")
    p.add_argument("--satellites", nargs="+", default=list(sat.ALL_SATELLITES),
                    help=f"which platforms to search (default: all of {list(sat.ALL_SATELLITES)})")
    p.add_argument("--window-min", type=float, default=45.0,
                    help="search +/- this many minutes around the window's own midpoint for overpasses")
    p.add_argument("--max-dist-km", type=float, default=5.0,
                    help="max nearest-pixel distance to count a sample as covered")
    p.add_argument("--nav-stride", type=int, default=8,
                    help="use every Nth 1 Hz nav sample for satellite matching (8 -> ~1.2 km along-track spacing)")
    args = p.parse_args(argv)

    if "EARTHDATA_TOKEN" not in os.environ:
        print("WARNING: EARTHDATA_TOKEN is not set -- er3t's downloaders will fail unless "
              f"every needed granule is already cached under {args.cache_dir!r}.")

    date = datetime.date.fromisoformat(args.date)
    t0_h, t1_h = _hms_to_hours(args.t0), _hms_to_hours(args.t1)

    p3 = load_nav(args.p3_nav)
    utc = p3["utc"]
    idxw = np.where((utc >= t0_h) & (utc <= t1_h))[0]
    if len(idxw) == 0:
        raise SystemExit(f"No nav samples found in {t0_h:.4f}-{t1_h:.4f} UTC -- check --date/--t0/--t1 "
                          f"against this file's own time range ({utc.min():.4f}-{utc.max():.4f} UTC).")
    utc_w, lat_w, lon_w = utc[idxw], p3["lat"][idxw], p3["lon"][idxw]

    # same flat-earth along-track distance convention as
    # compare_insitu_cloud_properties.py, so the two scripts' dist_w line up
    lat0, lon0 = np.mean(lat_w), np.mean(lon_w)
    km_per_deg_lon = 111.0 * np.cos(np.deg2rad(lat0))
    tx = (lon_w - lon0) * km_per_deg_lon
    ty = (lat_w - lat0) * 111.0
    dist_w = np.concatenate([[0.0], np.cumsum(np.sqrt(np.diff(tx) ** 2 + np.diff(ty) ** 2))])
    print(f"window {t0_h:.4f}-{t1_h:.4f} UTC, n={len(idxw)}, total distance {dist_w[-1]:.1f} km")

    sub = np.arange(0, len(idxw), args.nav_stride)
    print(f"subsampled to {len(sub)} points (every {args.nav_stride}th 1 Hz sample, "
          f"~{np.median(np.diff(dist_w[sub])) * 1000:.0f} m along-track spacing)")

    mid = len(utc_w) // 2
    t_mid = to_datetime(date, utc_w[mid])
    lon_mid, lat_mid = float(lon_w[mid]), float(lat_w[mid])
    candidates = sat.find_candidates(lon_mid, lat_mid, t_mid, satellites=args.satellites,
                                      window_min=args.window_min)
    print(f"\nfind_candidates: {len(candidates)} candidate(s) within +/-{args.window_min:.0f} min of "
          f"window midpoint {t_mid.isoformat()} (lon={lon_mid:.3f}, lat={lat_mid:.3f})")
    for c in candidates:
        off = f"{c.offset_minutes:+.1f} min" if c.offset_minutes is not None else "? (tag didn't parse)"
        print(f"  {c.satellite:7s} {c.instrument:5s} {c.filename_tag:16s} offset={off}")

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    n_saved = 0
    for c in candidates:
        print(f"\nfetching {c.satellite}/{c.filename_tag} ...")
        granule = sat.fetch_granule(c, args.cache_dir, want_l1b=False, want_l2=True, verbose=True)
        if granule.l2_file is None:
            print("  -> no L2 file (download failed or not available) -- skipping")
            continue

        n = len(sub)
        cwp = np.full(n, np.nan)
        cot = np.full(n, np.nan)
        cth = np.full(n, np.nan)
        ctt = np.full(n, np.nan)
        vza = np.full(n, np.nan)
        dist_km_nn = np.full(n, np.nan)
        for k, i in enumerate(sub):
            ps = sat.read_granule_point(granule, float(lon_w[i]), float(lat_w[i]), max_dist_km=args.max_dist_km)
            cwp[k], cot[k], cth[k], ctt[k], vza[k] = ps.cwp, ps.cot, ps.cth, ps.ctt, ps.vza
            dist_km_nn[k] = ps.distance_km

        n_cov = int(np.isfinite(cwp).sum())
        if n_cov == 0:
            print(f"  covers 0/{n} sampled points -- candidate's swath doesn't actually "
                  f"reach this track (edge clipped it) -- skipping save")
            continue
        print(f"  covers {n_cov}/{n} sampled points (nearest-pixel distance <= {args.max_dist_km:.0f} km), "
              f"NN distance range {np.nanmin(dist_km_nn):.2f}-{np.nanmax(dist_km_nn):.2f} km, "
              f"CWP range {np.nanmin(cwp):.1f}-{np.nanmax(cwp):.1f} g/m2")

        out_path = os.path.join(args.out_dir, f"{c.satellite}_{c.filename_tag.replace('.', '_')}.npz")
        np.savez(
            out_path,
            dist_w=dist_w[sub], lat=lat_w[sub], lon=lon_w[sub],
            cwp=cwp, cot=cot, cth=cth, ctt=ctt, vza=vza, nn_dist_km=dist_km_nn,
            satellite=c.satellite, instrument=c.instrument, filename_tag=c.filename_tag,
            offset_minutes=c.offset_minutes if c.offset_minutes is not None else np.nan,
        )
        print(f"  saved {out_path}")
        n_saved += 1

    print(f"\nDone: {n_saved}/{len(candidates)} candidate(s) actually covered this track and "
          f"were saved under {args.out_dir}")
    if n_saved:
        print(f"\nNow run, e.g.:\n  python3 compare_insitu_cloud_properties.py --p3-nav ... "
              f"--lwc ... --iwc ... --sat-dir {args.out_dir}")


if __name__ == "__main__":
    main()
