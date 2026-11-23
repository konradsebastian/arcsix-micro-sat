"""
RF07 (2024-06-07) 16:18:43-16:46 UTC "wall" transect: shareable in-situ
cloud-property figure, built so Paul Lawson and Sebastian Becker can
reproduce and check OUR (Sebastian Schmidt's) own LWC/IWC/LWP/IWP/COT
computation directly, without needing this project's AVIRIS or HALO/HSRL
lidar collocation code -- that machinery is real but unrelated to the
LWC/IWC/LWP/IWP/COT reconciliation this script is for, and lives in
36_insitu_composite_recreation.py if it's ever needed.

USAGE

  python compare_insitu_cloud_properties.py \\
      --p3-nav  ARCSIX-MetNav_P3B_20240607_R0.ict \\
      --lwc     ARCSIX-Lwc123_P3B_20240607_R1.ict \\
      --iwc     ARCSIX-HVPS4V50_P3B_20240607_R1.ict \\
      [--sat-dir satellite_results/] \\
      [--exclude-legs 12] \\
      [--lwp-method naive|binned] \\
      [--out comparison.png]

Only the first three (--p3-nav/--lwc/--iwc) are required -- standard
ARCSIX P-3 data products, ESPO/ASDC archive, RF07 2024-06-07. Everything
else is optional and off by default.

--sat-dir is OPTIONAL satellite overlay support: a directory of .npz
files, each with arrays "dist_w" (along-track distance, km, same
convention as this script's own dist_w), "cwp" (g/m2), "cot" and
(optionally) "cth" (satellite cloud-top height, m -- skipped cleanly if
not present), plus a string "satellite" name and (optionally)
"filename_tag" (e.g. "A2024159.1700" -> overpass scanline time 17:00
UTC, parsed and shown in the figure legend/stdout so it's clear which
overpass each curve is). If you don't have these, just omit --sat-dir
and the script runs on in-situ data alone.

This script itself does not download or collocate the satellite data --
that's done by the sibling script fetch_satellite_overlays.py (included
in this repo), which writes exactly the .npz files --sat-dir above
expects. It needs a patched er3t install and a NASA Earthdata token --
see ER3T_SETUP.md for the one-time setup. (Unlike AVIRIS/HALO below,
this piece was specifically requested, so it's included here rather
than cut.)

The figure's right-hand column is a set of narrow density-normalized
step histograms, one per main-panel row: cloud-top height (aircraft
per-leg vs. each satellite platform, sharing the LWC/IWC panels'
altitude axis), total water path (aircraft per-leg LWP+IWP vs. each
satellite's CWP), and COT (aircraft per-leg tau vs. each satellite's
COT). The water-path panel also prints the current run's mean LWP/IWP
(the same numbers as the console summary and the email to Paul/Parker)
directly on the figure.

AVIRIS and HALO/HSRL are intentionally NOT built into this script at all
(that collocation/KDTree code is substantially more involved than this
comparison needs, and not something you should have to stand up just to
check this figure). If you want to overlay your own along-track AVIRIS or
lidar trace, pass it in via --aviris-csv / --lidar-csv: a plain CSV with
columns dist_km,lwp,cot (AVIRIS) or dist_km,cth_m (lidar), resampled onto
this script's own distance axis. Both default to None (skipped).

METHOD, in brief (this is the main thing worth checking against your own
numbers -- see the accompanying email / project LOG.md for the fuller
three-way Lawson/Becker/us write-up):
  - LWC: SEA WCM-2000 hot-wire probe, column "Lwc1_wcm42" (the 2.0 mm
    sensing element).
  - IWC: SPEC HVPS4 optical-array probe, column "iwc"
    (CldMicro_IWC_InSitu_Optical_Drop_AMB) -- a direct area/mass-derived
    IWC from the measured ice size distribution, NOT the WCM-2000's own
    hot-wire differential proxy (TWC - LWC).
  - cloud mask: TWC (LWC+IWC) > CLOUD_THRESH g/m3. This is PURELY
    microphysical -- there is no independent geometric cloud-base cutoff,
    so it does not by construction exclude below-cloud-base ice/virga.
    See --exclude-legs below, which is exactly Paul's concern.
  - legs/boxes: one box per monotonic ascent or descent segment of the
    P-3's own altitude trace (scipy.signal.find_peaks on local maxima AND
    minima of GPS altitude -- a descent leg runs max-alt -> min-alt, an
    ascent leg runs min-alt -> max-alt).
  - LWP/IWP, two interchangeable integration methods (--lwp-method):
      "naive" (DEFAULT as of this release): sort the leg's own
      cloud-masked 1 Hz samples by altitude and trapezoidally integrate
      them directly -- no altitude binning at all.
      "binned" (this project's ORIGINAL method, matched
      36_insitu_composite_recreation.py's figure exactly until this
      release): mean-bin the leg's cloud-masked LWC/IWC into NB_Z
      altitude bins first, zero-fill any bin with no samples in it,
      then trapezoidally integrate the binned profile.
    These give noticeably different answers: in our own data, "binned"
    ran ~15% LOWER than "naive" on average, because several legs have
    only 4-5/6 altitude bins with any samples in them, and zero-filling
    an EMPTY bin (no data, not actually zero LWC/IWC) pulled the
    integral down -- a real artifact, not a physical difference.
    Switching to "naive" brings LWP much closer to Lawson's own figure
    (27.8 -> 31.7 g/m2 mean, vs. Lawson's 33.2) while leaving IWP and
    the IWP/LWP ratio essentially unchanged, which is why "naive" is
    now the default. "binned" is kept available (--lwp-method binned)
    for anyone who wants to reproduce the original figure or compare
    the two directly -- it is NOT a decision about which is "more
    "more correct." Both are provided so this can be checked directly.
  - COT: tau = K_LIQ*LWP + K_ICE*IWP, with FIXED K_LIQ/K_ICE below (i.e. a
    FIXED assumed effective radius: re=10 micron liquid, re=40 micron ice
    -- the ice value in particular has NOT been independently verified
    against this flight's own ice PSD). This is NOT a retrieved per-layer
    Reff like Becker's vertically-homogenized LER/IER, and it is NOT the
    directly-measured particle extinction that Lawson appears to use for
    at least the ice phase (see HVPS4's own "ext" column, 1/km,
    "extinction of the particles" -- a direct geometric-optics extinction
    from the measured ice size distribution, independent of any assumed
    Reff). This K_LIQ/K_ICE approach is the single biggest open
    methodological difference between the three analyses and is flagged
    here, not resolved.

--exclude-legs: leg 12 of this specific window (0-indexed, the LAST leg)
is the lowest-altitude leg of the whole transect (its cloudy samples sit
at 220-335 m, versus 450-630 m for most other legs) and has an anomalous
near-zero LWP (0.9 g/m2) paired with a real IWP (~5.0 g/m2) -- plus the
highest low-altitude IWC of any leg in the transect. That is plausibly a
below-cloud-base ice/virga signature slipping through our purely-
microphysical cloud mask (exactly Paul's concern about RF07, where the
P-3 did go below cloud base, unlike RF06). Pass --exclude-legs 12 to
reproduce the comparison with it removed -- with it excluded, our leg
count (12) matches Lawson's own 12-segment reference figure exactly.
Default is to exclude nothing, so the script reproduces the full,
un-edited transect unless you choose otherwise.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import binned_statistic_2d

# np.trapezoid is numpy>=2.0 only (it's the renamed np.trapz); fall back to
# np.trapz on older numpy so this doesn't require a specific numpy version.
_trapz = getattr(np, "trapezoid", None) or np.trapz

T0 = 16 + 18 / 60 + 43 / 3600  # 16:18:43 UTC, confirmed anchor point
T1 = 16 + 46 / 60  # "through 16:46"
CLOUD_THRESH = 0.04  # g/m3 TWC
K_LIQ = 0.15  # m2/g, 3/(2*rho_w*re), re=10 micron
K_ICE = 0.04  # m2/g, assumed re=40 micron -- NOT independently verified
NB_Z = 6  # altitude bins per box, for the "binned" LWP/IWP method

SAT_COLORS = {
    "terra": "crimson", "aqua": "darkviolet", "snpp": "royalblue", "noaa20": "goldenrod",
}


def parse_overpass_time(filename_tag):
    """filename_tag is e.g. "A2024159.1700" (year 2024, day-of-year 159 =
    2024-06-07, scanline time 17:00 UTC). Returns "HH:MM UTC" or None if
    filename_tag isn't in the expected form."""
    try:
        hhmm = str(filename_tag).split(".")[-1]
        if len(hhmm) == 4 and hhmm.isdigit():
            return f"{hhmm[:2]}:{hhmm[2:]} UTC"
    except Exception:
        pass
    return None


def load_p3_nav(path):
    df = pd.read_csv(path, header=74)
    return dict(utc=df["Time_Start"].values / 3600.0, lon=df["Longitude"].values,
                lat=df["Latitude"].values, alt=df["GPS_Altitude"].values)


def load_p3_lwc(path):
    df = pd.read_csv(path, header=34)
    return dict(utc=df["Time_Start"].values / 3600.0, lwc=df["Lwc1_wcm42"].values)


def load_p3_iwc_hvps4(path):
    df = pd.read_csv(path, header=284)
    return dict(utc=df["Time_Start"].values / 3600.0, iwc=df["iwc"].values.astype(float))


def build_legs(p3, lwc, iwc, lwp_method="binned"):
    """Segment the window into monotonic ascent/descent legs and compute
    per-leg LWP/IWP/COT, matching 36_insitu_composite_recreation.py's own
    method when lwp_method="binned" (the project's existing default)."""
    utc = p3["utc"]
    m = (utc >= T0 - 0.0005) & (utc <= T1 + 0.0005)
    idxw = np.where(m)[0]
    if len(idxw) == 0:
        raise SystemExit(f"No P-3 nav samples found in {T0:.4f}-{T1:.4f} UTC -- "
                          f"check this is the RF07 (2024-06-07) MetNav file.")
    utc_w = utc[idxw]
    lat_w, lon_w, alt_w = p3["lat"][idxw], p3["lon"][idxw], p3["alt"][idxw]

    R = 6371.0
    lat1, lat2 = np.radians(lat_w[:-1]), np.radians(lat_w[1:])
    dlat, dlon = np.radians(np.diff(lat_w)), np.radians(np.diff(lon_w))
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist_w = np.concatenate([[0.0], np.cumsum(2 * R * np.arcsin(np.sqrt(a)))])

    lwc_w = np.interp(utc_w, lwc["utc"], lwc["lwc"])
    iwc_w = np.interp(utc_w, iwc["utc"], iwc["iwc"])
    twc_w = lwc_w + iwc_w
    cloud_w = twc_w > CLOUD_THRESH
    print(f"window {utc_w[0]:.4f}-{utc_w[-1]:.4f} UTC, n={len(idxw)}, "
          f"total distance {dist_w[-1]:.1f} km, cloudy samples "
          f"{cloud_w.sum()}/{len(cloud_w)}, LWP method = {lwp_method}")

    peaks, _ = find_peaks(alt_w, distance=60, prominence=100)
    troughs, _ = find_peaks(-alt_w, distance=60, prominence=100)
    bounds = np.unique(np.concatenate([[0], peaks, troughs, [len(alt_w) - 1]]))
    segments = list(zip(bounds[:-1], bounds[1:]))
    print(f"leg segmentation: {len(segments)} monotonic ascent/descent legs")

    legs = []
    for si, (i0, i1) in enumerate(segments):
        sl = slice(i0, i1 + 1)
        kind = "descent" if alt_w[i0] > alt_w[i1] else "ascent"
        cl = cloud_w[sl]
        if cl.sum() < 3:
            continue
        d_seg, a_seg = dist_w[sl][cl], alt_w[sl][cl]
        lwc_seg, iwc_seg = lwc_w[sl][cl], iwc_w[sl][cl]
        d_lo, d_hi = dist_w[i0], dist_w[i1]
        a_lo, a_hi = min(alt_w[i0], alt_w[i1]), max(alt_w[i0], alt_w[i1])
        if d_hi <= d_lo or a_hi <= a_lo:
            continue
        real_cth = a_seg.max()
        a_edges = np.linspace(a_lo, real_cth, NB_Z + 1)
        d_edges = np.linspace(d_lo, d_hi, NB_Z + 1)
        # NOTE: this 2D (distance x altitude) grid is ALWAYS built via
        # binned_statistic_2d, regardless of --lwp-method. It feeds the
        # LWC/IWC color-mesh panels (a)/(b) and -- deliberately -- is
        # kept as a regularized grid rather than switched to a raw
        # scatter, because it is meant to double as a reusable gridded
        # data product later, not just a plot. This is an intentional
        # decoupling from the LWP/IWP *scalar* integration below, which
        # defaults to "naive" (raw-sample, no binning) because that is
        # closer to Lawson's own numbers -- see the lwp_method branch
        # just below and the METHOD docstring at the top of this file.
        lwc_grid, _, _, _ = binned_statistic_2d(d_seg, a_seg, lwc_seg, statistic="mean",
                                                 bins=[d_edges, a_edges])
        iwc_grid, _, _, _ = binned_statistic_2d(d_seg, a_seg, iwc_seg, statistic="mean",
                                                 bins=[d_edges, a_edges])

        if lwp_method == "binned":
            a_centers = 0.5 * (a_edges[:-1] + a_edges[1:])
            lwc_prof = np.nanmean(lwc_grid, axis=0)
            iwc_prof = np.nanmean(iwc_grid, axis=0)
            lwp = _trapz(np.nan_to_num(lwc_prof), a_centers)
            iwp = _trapz(np.nan_to_num(iwc_prof), a_centers)
        elif lwp_method == "naive":
            order = np.argsort(a_seg)
            lwp = _trapz(lwc_seg[order], a_seg[order])
            iwp = _trapz(iwc_seg[order], a_seg[order])
        else:
            raise ValueError(f"unknown lwp_method {lwp_method!r}")
        tau = K_LIQ * lwp + K_ICE * iwp
        legs.append(dict(idx=len(legs), d_lo=d_lo, d_hi=d_hi, a_lo=a_lo, a_hi=a_hi,
                          real_cth=real_cth, lwp=lwp, iwp=iwp, tau=tau, kind=kind,
                          d_edges=d_edges, a_edges=a_edges, lwc_grid=lwc_grid, iwc_grid=iwc_grid))
        print(f"  leg {len(legs)-1} ({kind}): dist {d_lo:.1f}-{d_hi:.1f} km, "
              f"alt {a_lo:.0f}-{real_cth:.0f} m, LWP={lwp:.1f} IWP={iwp:.2f} g/m2, tau={tau:.1f}")
    return legs, dist_w, cloud_w


def load_satellite_overlays(sat_dir, dist_w, cloud_w):
    """Optional: load every *.npz in sat_dir (one file per platform) and
    interpolate cwp/cot onto this script's own dist_w. Each file is
    expected to carry "dist_w" (km, same along-track convention as this
    script), "cwp" (g/m2), "cot", and a string "satellite" name. Returns
    a list of dicts, or [] if sat_dir is None / has no .npz files."""
    if not sat_dir:
        return []
    files = sorted(glob.glob(os.path.join(sat_dir, "*.npz")))
    if not files:
        print(f"WARNING: no .npz files found under {sat_dir} -- no satellite overlay")
        return []
    out = []
    for fpath in files:
        d = np.load(fpath, allow_pickle=True)
        name = str(d["satellite"]) if "satellite" in d else os.path.basename(fpath)
        overpass = parse_overpass_time(d["filename_tag"]) if "filename_tag" in d else None
        sat_dist = d["dist_w"].astype(float)
        cwp = d["cwp"].astype(float)
        cot = d["cot"].astype(float)
        cwp_ok = np.isfinite(cwp) & (cwp >= 0) & (cwp <= 10000)
        cot_ok = np.isfinite(cot) & (cot >= 0)
        cwp_i = (np.interp(dist_w, sat_dist[cwp_ok], cwp[cwp_ok], left=np.nan, right=np.nan)
                 if cwp_ok.sum() > 1 else np.full_like(dist_w, np.nan))
        cot_i = (np.interp(dist_w, sat_dist[cot_ok], cot[cot_ok], left=np.nan, right=np.nan)
                 if cot_ok.sum() > 1 else np.full_like(dist_w, np.nan))
        if "cth" in d:
            cth = d["cth"].astype(float)
            cth_ok = np.isfinite(cth) & (cth >= 0) & (cth <= 20000)
            cth_i = (np.interp(dist_w, sat_dist[cth_ok], cth[cth_ok], left=np.nan, right=np.nan)
                     if cth_ok.sum() > 1 else np.full_like(dist_w, np.nan))
        else:
            cth_i = np.full_like(dist_w, np.nan)
        cwp_cloudy = cwp_i[cloud_w]
        cwp_cloudy = cwp_cloudy[np.isfinite(cwp_cloudy)]
        cot_cloudy = cot_i[cloud_w]
        cot_cloudy = cot_cloudy[np.isfinite(cot_cloudy)]
        cth_cloudy = cth_i[cloud_w]
        cth_cloudy = cth_cloudy[np.isfinite(cth_cloudy)]
        print(f"satellite {name} (overpass {overpass or 'unknown'}): mean CWP (over cloudy "
              f"samples) = {cwp_cloudy.mean() if len(cwp_cloudy) else float('nan'):.1f} g/m2, "
              f"mean COT = {cot_cloudy.mean() if len(cot_cloudy) else float('nan'):.1f}, "
              f"mean CTH = {cth_cloudy.mean() if len(cth_cloudy) else float('nan'):.0f} m")
        out.append(dict(name=name, dist_w=dist_w, cwp=cwp_i, cot=cot_i, cth=cth_i, overpass=overpass,
                         color=SAT_COLORS.get(name, "gray")))
    return out


def load_csv_overlay(path, columns):
    if not path:
        return None
    df = pd.read_csv(path)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing expected column(s) {missing}")
    return {c: df[c].values for c in df.columns}


def summarize(legs, exclude):
    kept = [b for b in legs if b["idx"] not in exclude]
    lwp = np.array([b["lwp"] for b in kept])
    iwp = np.array([b["iwp"] for b in kept])
    print(f"\n=== summary ({len(kept)}/{len(legs)} legs"
          f"{f', excluding {sorted(exclude)}' if exclude else ''}) ===")
    print(f"  LWP: mean={lwp.mean():.1f} median={np.median(lwp):.1f} sum={lwp.sum():.1f} g/m2")
    print(f"  IWP: mean={iwp.mean():.1f} median={np.median(iwp):.1f} sum={iwp.sum():.1f} g/m2")
    print(f"  IWP/LWP ratio (sum-based): {iwp.sum()/lwp.sum():.3f}")
    return dict(n=len(kept), n_total=len(legs), exclude=sorted(exclude),
                lwp_mean=lwp.mean(), lwp_median=float(np.median(lwp)), lwp_sum=lwp.sum(),
                iwp_mean=iwp.mean(), iwp_median=float(np.median(iwp)), iwp_sum=iwp.sum(),
                ratio=iwp.sum() / lwp.sum())


def make_figure(legs, dist_w, cloud_w, exclude, satellites, aviris_overlay, lidar_overlay, stats,
                 outfile, figsize):
    kept = [b for b in legs if b["idx"] not in exclude]
    have_sat_cth = any(np.isfinite(sat.get("cth", np.nan)).any() for sat in satellites)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.3, 1.3, 1, 1], width_ratios=[4, 1.1],
                           hspace=0.3, wspace=0.06)

    # Satellite CTH here (roughly 1-3.5 km) sits far above the in-situ
    # altitude range (0-900 m) -- plotting both on one axis either clips
    # the satellite values off-scale or squashes the in-situ detail (the
    # whole point of these two panels) into a thin sliver at the bottom.
    # Standard fix, same one already used in this project's own
    # 36_insitu_composite_recreation.py: a small "broken axis" strip
    # directly above each main panel, dedicated to the satellite CTH
    # range, with the usual broken-axis diagonal slash marks at the
    # boundary. Only built when there's actually satellite CTH data to
    # need it -- otherwise this would just be an empty strip.
    if have_sat_cth:
        gs_a = gs[0, :].subgridspec(2, 2, height_ratios=[0.45, 1], width_ratios=[4, 1.1],
                                     hspace=0.08, wspace=0.06)
        break_a = fig.add_subplot(gs_a[0, 0])
        ax_lwc = fig.add_subplot(gs_a[1, 0], sharex=break_a)
        pdf_break_a = fig.add_subplot(gs_a[0, 1], sharey=break_a)
        ax_cth_hist = fig.add_subplot(gs_a[1, 1], sharey=ax_lwc)

        gs_b = gs[1, :].subgridspec(2, 2, height_ratios=[0.45, 1], width_ratios=[4, 1.1],
                                     hspace=0.08, wspace=0.06)
        break_b = fig.add_subplot(gs_b[0, 0], sharex=ax_lwc, sharey=break_a)
        ax_iwc = fig.add_subplot(gs_b[1, 0], sharex=ax_lwc)

        plt.setp(break_a.get_xticklabels(), visible=False)
        plt.setp(break_b.get_xticklabels(), visible=False)
        plt.setp(pdf_break_a.get_xticklabels(), visible=False)
        plt.setp(pdf_break_a.get_yticklabels(), visible=False)
    else:
        break_a = break_b = pdf_break_a = None
        ax_lwc = fig.add_subplot(gs[0, 0])
        ax_iwc = fig.add_subplot(gs[1, 0], sharex=ax_lwc)
        ax_cth_hist = fig.add_subplot(gs[0:2, 1], sharey=ax_lwc)

    ax_wp = fig.add_subplot(gs[2, 0], sharex=ax_lwc)
    ax_tau = fig.add_subplot(gs[3, 0], sharex=ax_lwc)
    for ax in (ax_lwc, ax_iwc, ax_wp):
        plt.setp(ax.get_xticklabels(), visible=False)

    # right-hand column: density-normalized step histograms, one "row" per
    # main panel. CTH (split the same way as the main columns, when there's
    # satellite CTH) shares the LWC/IWC panels' altitude axis; water-path
    # and COT each get their own row, sharing THEIR main panel's y-axis so
    # the bin ranges line up visually.
    ax_wp_hist = fig.add_subplot(gs[2, 1], sharey=ax_wp)
    ax_tau_hist = fig.add_subplot(gs[3, 1], sharey=ax_tau)
    for ax in (ax_cth_hist, ax_wp_hist, ax_tau_hist):
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="both", labelsize=7)
    ax_tau_hist.set_xlabel("density", fontsize=7)

    for b in kept:
        extent = [b["d_edges"][0], b["d_edges"][-1], b["a_edges"][0], b["a_edges"][-1]]
        ax_lwc.imshow(b["lwc_grid"].T, origin="lower", aspect="auto", extent=extent,
                      cmap="Greens", vmin=0, vmax=0.3)
        ax_iwc.imshow(b["iwc_grid"].T, origin="lower", aspect="auto", extent=extent,
                      cmap="Blues", vmin=0, vmax=0.3)
        for ax in (ax_lwc, ax_iwc):
            ax.add_patch(plt.Rectangle((b["d_lo"], b["a_lo"]), b["d_hi"] - b["d_lo"],
                                        b["a_hi"] - b["a_lo"], fill=False, edgecolor="0.3", lw=0.8))
        d_mid, d_w = 0.5 * (b["d_lo"] + b["d_hi"]), b["d_hi"] - b["d_lo"]
        ax_wp.bar(d_mid, b["lwp"], width=d_w, color="#62995f", edgecolor="0.3", lw=0.5,
                  label="liquid (LWP)" if b["idx"] == kept[0]["idx"] else None)
        ax_wp.bar(d_mid, b["iwp"], width=d_w, bottom=b["lwp"], color="#8fd6f0",
                  edgecolor="0.3", lw=0.5, label="ice (IWP)" if b["idx"] == kept[0]["idx"] else None)
        ax_tau.hlines(b["tau"], b["d_lo"], b["d_hi"], color="teal", lw=2.2,
                      label="aircraft (total)" if b["idx"] == kept[0]["idx"] else None)

    # keep the LWC/IWC main panels keyed to the P-3's own altitude range --
    # this is the content that matters there -- rather than letting
    # matplotlib auto-scale to accommodate satellite CTH values plotted
    # elsewhere (the break strips, below).
    if kept:
        main_top = max(900.0, max(b["a_hi"] for b in kept) * 1.1)
        ax_lwc.set_ylim(0, main_top)
        ax_iwc.set_ylim(0, main_top)

    for sat in satellites:
        lbl = f"satellite ({sat['name']}, {sat['overpass']})" if sat.get("overpass") \
            else f"satellite ({sat['name']})"
        ax_wp.plot(sat["dist_w"], sat["cwp"], "-", color=sat["color"], lw=1, alpha=0.8, label=lbl)
        ax_tau.plot(sat["dist_w"], sat["cot"], "-", color=sat["color"], lw=1, alpha=0.8, label=lbl)
        if have_sat_cth and np.isfinite(sat.get("cth", np.nan)).any():
            # plotted in the broken-axis strip ABOVE each main panel, not
            # on the main panel itself -- see comment above.
            break_a.plot(sat["dist_w"], sat["cth"], "--", color=sat["color"], lw=1.3, alpha=0.85,
                         label=f"{sat['name']} CTH*")
            break_b.plot(sat["dist_w"], sat["cth"], "--", color=sat["color"], lw=1.3, alpha=0.85)

    if have_sat_cth:
        sat_cth_all = np.concatenate([sat["cth"] for sat in satellites
                                       if np.isfinite(sat.get("cth", np.nan)).any()])
        sat_finite = sat_cth_all[np.isfinite(sat_cth_all)]
        if sat_finite.size:
            break_a.set_ylim(max(0, sat_finite.min() - 150), sat_finite.max() + 150)
        break_a.set_ylabel("Satellite\nCTH [m]", fontsize=8)
        break_b.set_ylabel("Satellite\nCTH [m]", fontsize=8)
        break_a.tick_params(axis="y", labelsize=7)
        break_b.tick_params(axis="y", labelsize=7)
        break_a.legend(loc="upper left", fontsize=6.5, framealpha=0.9, ncol=2)
        # standard broken-y-axis convention: small diagonal slash marks at
        # the shared boundary instead of a continuous spine, so the scale
        # discontinuity is signaled rather than hidden.
        dd = 0.6
        slash_kw = dict(marker=[(-1, -dd), (1, dd)], markersize=9, linestyle="none",
                         color="k", mec="k", mew=1, clip_on=False)
        for bax, ax in ((break_a, ax_lwc), (break_b, ax_iwc)):
            for x0 in (0, 1):
                bax.plot([x0], [0], transform=bax.transAxes, **slash_kw)
                ax.plot([x0], [1], transform=ax.transAxes, **slash_kw)
        for x0 in (0, 1):
            pdf_break_a.plot([x0], [0], transform=pdf_break_a.transAxes, **slash_kw)
            ax_cth_hist.plot([x0], [1], transform=ax_cth_hist.transAxes, **slash_kw)

    if aviris_overlay is not None:
        ax_wp.plot(aviris_overlay["dist_km"], aviris_overlay["lwp"], "-", color="darkorange",
                   lw=1, label="AVIRIS LWP")
        ax_tau.plot(aviris_overlay["dist_km"], aviris_overlay["cot"], "-", color="darkorange",
                    lw=1, label="AVIRIS COT")
    if lidar_overlay is not None:
        ax_lwc.plot(lidar_overlay["dist_km"], lidar_overlay["cth_m"], "-", color="black",
                    lw=1, label="lidar CTH")

    # --- right-hand histogram column ---
    def _hstep(ax, values, color, label, ls="-"):
        if ax is None:
            return
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            return
        ax.hist(values, bins=min(8, max(2, len(values) // 2)), orientation="horizontal",
                density=True, histtype="step", color=color, lw=1.4, linestyle=ls, label=label)

    _hstep(ax_cth_hist, [b["real_cth"] for b in kept], "0.25", "aircraft (leg top)")
    for sat in satellites:
        if have_sat_cth and np.isfinite(sat.get("cth", np.nan)).any():
            # same split as the main panels: satellite CTH density goes in
            # the break-strip histogram, not the main (low-range) one.
            _hstep(pdf_break_a, sat["cth"][cloud_w], sat["color"], sat["name"], ls="--")
    _hstep(ax_wp_hist, [b["lwp"] + b["iwp"] for b in kept], "teal", "aircraft (total)")
    for sat in satellites:
        _hstep(ax_wp_hist, sat["cwp"][cloud_w], sat["color"], sat["name"])
    _hstep(ax_tau_hist, [b["tau"] for b in kept], "teal", "aircraft (total)")
    for sat in satellites:
        _hstep(ax_tau_hist, sat["cot"][cloud_w], sat["color"], sat["name"])
    ax_cth_hist.legend(loc="upper right", fontsize=6, frameon=False)

    ax_lwc.set_ylabel("Altitude [m]\n(LWC)")
    ax_iwc.set_ylabel("Altitude [m]\n(IWC)")
    ax_wp.set_ylabel("Water path\n[g/m2]")
    ax_tau.set_ylabel("Optical\nthickness")
    ax_tau.set_xlabel("Along-track distance [km]")
    (pdf_break_a if have_sat_cth else ax_cth_hist).set_title("CTH", fontsize=8)
    ax_wp_hist.set_title("water path", fontsize=8)
    ax_tau_hist.set_title("COT", fontsize=8)
    ax_wp.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    ax_tau.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    if lidar_overlay is not None:
        ax_lwc.legend(loc="lower right", fontsize=8, frameon=False)

    title = ("RF07 2024-06-07 16:18:43-16:46 UTC -- P-3 in-situ composite\n"
              "(Sebastian Schmidt's own computation, for comparison against "
              "Lawson's and Becker's)")
    # --- LWP/IWP totals, same numbers as the console summary / email ---
    if stats is not None:
        excl_note = f", excl. leg{'s' if len(stats['exclude']) != 1 else ''} " \
                    f"{','.join(map(str, stats['exclude']))}" if stats["exclude"] else ""
        title += (f"\nAircraft mean ({stats['n']}/{stats['n_total']} legs{excl_note}): "
                   f"LWP={stats['lwp_mean']:.1f} g/m2, IWP={stats['iwp_mean']:.1f} g/m2, "
                   f"IWP/LWP={stats['ratio']:.3f}")
    (break_a if have_sat_cth else ax_lwc).set_title(title, fontsize=10)
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"\nfigure written to {outfile}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--p3-nav", required=True, help="P-3 MetNav .ict file")
    p.add_argument("--lwc", required=True, help="SEA WCM-2000 Lwc123 .ict file")
    p.add_argument("--iwc", required=True, help="SPEC HVPS4 .ict file")
    p.add_argument("--sat-dir", default=None,
                    help="optional directory of satellite *.npz overlay files (skipped if omitted)")
    p.add_argument("--aviris-csv", default=None,
                    help="optional CSV with columns dist_km,lwp,cot (skipped if omitted)")
    p.add_argument("--lidar-csv", default=None,
                    help="optional CSV with columns dist_km,cth_m (skipped if omitted)")
    p.add_argument("--exclude-legs", default="", help="comma-separated leg indices to drop, e.g. 12")
    p.add_argument("--lwp-method", choices=["binned", "naive"], default="naive",
                    help="naive (default as of this release: raw-sample trapezoidal, no "
                         "altitude binning -- closer to Lawson's own numbers, see docstring) or "
                         "binned (legacy method, matches 36_insitu_composite_recreation.py's "
                         "original figure, kept for comparison)")
    p.add_argument("--out", default="lawson_becker_comparison_rf07.png", help="output figure path")
    p.add_argument("--figsize", default="11,10", help="figure size in inches, e.g. 11,10")
    args = p.parse_args()

    exclude = {int(x) for x in args.exclude_legs.split(",") if x.strip() != ""}
    figsize = tuple(float(x) for x in args.figsize.split(","))

    p3 = load_p3_nav(args.p3_nav)
    lwc = load_p3_lwc(args.lwc)
    iwc = load_p3_iwc_hvps4(args.iwc)
    legs, dist_w, cloud_w = build_legs(p3, lwc, iwc, lwp_method=args.lwp_method)

    stats_all = summarize(legs, exclude=set())
    stats = stats_all
    if exclude:
        stats = summarize(legs, exclude=exclude)

    satellites = load_satellite_overlays(args.sat_dir, dist_w, cloud_w)
    aviris_overlay = load_csv_overlay(args.aviris_csv, ["dist_km", "lwp", "cot"])
    lidar_overlay = load_csv_overlay(args.lidar_csv, ["dist_km", "cth_m"])

    make_figure(legs, dist_w, cloud_w, exclude, satellites, aviris_overlay, lidar_overlay, stats,
                args.out, figsize)


if __name__ == "__main__":
    main()
