#!/usr/bin/env python3
"""
jcope_extract_full.py — FULL-RECORD extraction of JCOPE-EC surface t/phy/din for the crop
region, across ALL year-tars (1993-2018). Extends the 2000-2004 pilot to the whole record so
we can build a proper multi-decadal, all-season Hobday marine-heatwave climatology.

Same guarantees as the pilot:
  * READ-ONLY on the source drive (HD-SHU); uses `tarfile` so unwanted variables are seeked
    past (only t/phy/din bytes are read).
  * WRITE-ONLY-NEW on the laptop: every output created fresh (temp-name then rename); if a
    target already exists it is SKIPPED -> the run is RESUMABLE and re-running is safe. The
    2000-2004 files from the pilot are simply skipped.
  * Nothing is ever overwritten or deleted; HD-PGF-A (NTFS/read-only) is untouched.

Output: /Volumes/HD-PGF-A/JCOPE_full/{t,phy,din}/<var>.YYYYMMDD.npy  (upper 200 m,
region-cropped (10,356,362)). ~147 GB for the full 26 years x 3 vars (4.6 TB drive).
NOTE: writes to HD-PGF-A (reformatted exFAT). Set ZK0,ZK1=24,25 for a ~15 GB surface run.

Run on the Mac:
    cd "/Users/lukequasar/Documents/Ocean Project"
    source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate ocean
    python src/jcope_extract_full.py
"""
from __future__ import annotations
import os, sys, glob, gzip, tarfile, time, numpy as np

# all top-level year-tars (the old/ duplicates live in a subdir and are NOT matched),
# sorted chronologically: 1993-4, 1995-9, 2000-4, 2005-9, 2010-4, 2015-8
TARS = sorted(glob.glob("/Volumes/*/jcope-ec*.tar"))
if not TARS:
    raise SystemExit("ERROR: no jcope-ec*.tar found — is HD-SHU connected?")

# Optional selection so you can run in chunks (resumable across runs):
#   python src/jcope_extract_full.py                -> all six tars
#   python src/jcope_extract_full.py first|second   -> older/recent HALF (3 tars each)
#   python src/jcope_extract_full.py 2010 2015      -> only tars matching '2010' or '2015'
#                                                      (= recent THIRD, 2010-2018)
#   python src/jcope_extract_full.py 2000 2005      -> middle third (2000-2009), etc.
args = sys.argv[1:]
h = len(TARS) // 2
if not args or args == ["all"]:
    pass
elif args == ["first"]:
    TARS = TARS[:h]
elif args == ["second"]:
    TARS = TARS[h:]
else:                                   # one or more year substrings -> match any
    TARS = [t for t in TARS if any(s in os.path.basename(t) for s in args)]
if not TARS:
    raise SystemExit(f"ERROR: selection {args} matched no tars.")
# target the reformatted (exFAT) HD-PGF-A — 4.6 TB, so we can afford depth levels
OUTBASE = "/Volumes/HD-PGF-A/JCOPE_full"

VARS = {"t", "phy", "din"}
NZ, NY, NX = 25, 620, 866
UNDEF = -99.9999
LON0, LON1 = 144, 506       # ~120-150E
LAT0, LAT1 = 0, 356         # ~10.46-40N
ZK0, ZK1 = 15, 25           # UPPER OCEAN: depth idx 15..24 = -200 m .. 0 m (10 levels).
                            #   surface is the LAST index (per-file shape (10,356,362)).
                            #   ~147 GB for full 1993-2018 x 3 vars. Set 24,25 for surface only.

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

def save_new(path, arr):
    if os.path.exists(path):
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".part{os.getpid()}"
    np.save(tmp, arr); os.replace(tmp + ".npy", path)
    return True

def process_tar(tar):
    saved = skipped = 0
    with tarfile.open(tar, "r") as tf:
        for m in tf:
            if not m.isfile():
                continue
            base = os.path.basename(m.name)
            var = base.split(".")[0]
            if var not in VARS:
                continue
            if base.endswith(".ctl.gz"):
                cp = os.path.join(OUTBASE, var, f"{var}.SAMPLE.ctl")
                if not os.path.exists(cp):
                    os.makedirs(os.path.dirname(cp), exist_ok=True)
                    txt = gzip.decompress(tf.extractfile(m).read())
                    with open(cp + f".part{os.getpid()}", "wb") as fh: fh.write(txt)
                    os.replace(cp + f".part{os.getpid()}", cp)
                continue
            if not base.endswith(".dat.gz"):
                continue
            date = base.split(".")[1]
            outp = os.path.join(OUTBASE, var, f"{var}.{date}.npy")
            if os.path.exists(outp):
                skipped += 1; continue
            raw = gzip.decompress(tf.extractfile(m).read())
            a = np.frombuffer(raw, dtype=">f4").reshape(NZ, NY, NX)
            crop = a[ZK0:ZK1, LAT0:LAT1, LON0:LON1].astype("<f4").copy()
            crop[crop <= UNDEF + 1e-3] = np.nan
            save_new(outp, crop); saved += 1
    return saved, skipped

def main():
    log(f"dest: {OUTBASE}")
    log(f"{len(TARS)} tars: " + ", ".join(os.path.basename(t) for t in TARS))
    tot_s = tot_k = 0
    for tar in TARS:
        s, k = process_tar(tar)
        tot_s += s; tot_k += k
        log(f"  {os.path.basename(tar)}: +{s} new, {k} already-present  (running: {tot_s} new)")
    log(f"DONE: {tot_s} new files, {tot_k} pre-existing. Out -> {OUTBASE}")
    for v in sorted(VARS):
        n = len(glob.glob(os.path.join(OUTBASE, v, f"{v}.*.npy")))
        log(f"  {v}: {n} daily files total")

if __name__ == "__main__":
    main()
