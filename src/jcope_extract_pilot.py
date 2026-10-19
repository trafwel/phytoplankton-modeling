#!/usr/bin/env python3
"""
jcope_extract_pilot.py — PILOT extraction of JCOPE-EC (data-assimilating reanalysis) for
the crop region, from the 2000-2004 tar. Pulls t / phy / din only, crops to region + upper
ocean, and saves per-day .npy to HD-PGF-A. All-season, daily, observationally constrained.

Design choices:
  * Uses Python `tarfile` on the (uncompressed) tar so unwanted variables are SEEKED PAST,
    not read -> only t/phy/din bytes are actually read.
  * READ-ONLY on the source drive (HD-SHU). WRITE-ONLY-NEW on HD-PGF-A: every output is
    created fresh; if a target file already exists it is SKIPPED (never overwritten), and
    nothing is ever deleted. Honors "can write, cannot overwrite."

JCOPE file: one variable/day, GrADS, big-endian float32, shape (25, 620, 866), stored as
`val`, undef -99.9999. Depth levels are DEEP->SURFACE (index 24 = 0 m surface).

Run on the Mac:
    cd "/Users/lukequasar/Documents/Ocean Project"
    source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate ocean
    python src/jcope_extract_pilot.py
"""
from __future__ import annotations
import os, glob, gzip, tarfile, time, numpy as np

# ---- source tar (auto-locate HD-SHU under any /Volumes name) ----
def find_tar():
    for c in sorted(glob.glob("/Volumes/*/jcope-ec2000-4.tar")):
        return c
    raise SystemExit("ERROR: jcope-ec2000-4.tar not found — is HD-SHU connected?")
TAR = find_tar()

# ---- destination (write-only-new) ----
# HD-PGF-A is NTFS and mounted READ-ONLY on macOS (fskit, no safe native write), so we
# write to the WRITABLE laptop disk instead. Surface-only keeps the pilot to ~2.8 GB.
OUTBASE = "/Users/lukequasar/Documents/Ocean Project/JCOPE Data/jcope_pilot"

# ---- what to pull ----
VARS = {"t", "phy", "din"}                 # temperature, phytoplankton, nutrients(N)
NZ, NY, NX = 25, 620, 866
UNDEF = -99.9999
# region crop (JCOPE grid: lon = 107.9583 + i*0.08333 ; lat = 10.4583 + j*0.08333)
LON0, LON1 = 144, 506      # ~120-150E   (362 pts)
LAT0, LAT1 = 0, 356        # ~10.46-40N  (356 pts)  [extend LAT1->620 later for full Oyashio]
ZK0, ZK1 = 24, 25          # SURFACE only (index 24 = 0 m). ~2.8 GB for 5 yrs x 3 vars.
                           #   -> set to 15,25 for upper 200 m (10 levels, ~28 GB) later.

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

def safe_save(path, arr):
    """Create-only: never overwrite an existing file."""
    if os.path.exists(path):
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".part{os.getpid()}"          # write to a NEW temp name, then rename
    np.save(tmp, arr)                            # np.save appends .npy -> tmp+.npy
    os.replace(tmp + ".npy", path)               # rename onto the (nonexistent) target
    return True

def main():
    log(f"source : {TAR}")
    log(f"dest   : {OUTBASE}  (write-only-new)")
    for v in sorted(VARS):
        os.makedirs(os.path.join(OUTBASE, v), exist_ok=True)
    saved_ctl = set()
    n_saved = n_skip = 0
    with tarfile.open(TAR, "r") as tf:           # plain tar -> members are seekable/skippable
        for m in tf:
            if not m.isfile():
                continue
            base = os.path.basename(m.name)      # e.g. phy.20001015.dat.gz
            var = base.split(".")[0]
            if var not in VARS:
                continue                         # tarfile seeks past this member's data
            # save one .ctl per variable (units / metadata reference)
            if base.endswith(".ctl.gz") and var not in saved_ctl:
                txt = gzip.decompress(tf.extractfile(m).read())
                cp = os.path.join(OUTBASE, var, f"{var}.SAMPLE.ctl")
                if not os.path.exists(cp):
                    with open(cp + f".part{os.getpid()}", "wb") as fh: fh.write(txt)
                    os.replace(cp + f".part{os.getpid()}", cp)
                saved_ctl.add(var)
                continue
            if not base.endswith(".dat.gz"):
                continue
            date = base.split(".")[1]            # YYYYMMDD
            outp = os.path.join(OUTBASE, var, f"{var}.{date}.npy")
            if os.path.exists(outp):             # never overwrite
                n_skip += 1; continue
            raw = gzip.decompress(tf.extractfile(m).read())
            a = np.frombuffer(raw, dtype=">f4").reshape(NZ, NY, NX)
            crop = a[ZK0:ZK1, LAT0:LAT1, LON0:LON1].astype("<f4").copy()
            crop[crop <= UNDEF + 1e-3] = np.nan
            if safe_save(outp, crop):
                n_saved += 1
                if n_saved % 200 == 0:
                    log(f"  saved {n_saved} files (last {var} {date}, shape {crop.shape})")
    log(f"DONE: saved {n_saved} new files, skipped {n_skip} existing. Out -> {OUTBASE}")
    log(f"crop shape per file: ({ZK1-ZK0}, {LAT1-LAT0}, {LON1-LON0}) depth x lat x lon "
        f"(surface = last depth index)")

if __name__ == "__main__":
    main()
