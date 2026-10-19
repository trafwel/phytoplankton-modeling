#!/usr/bin/env python3
"""
copy_ofes_to_drive.py — copy OFES-BGC `.dta` files from a source drive onto the
attached 5 TB drive, organized by variable (matching step1_bgc_series.py's layout).

Both locations are local/mounted drives (on macOS they live under /Volumes/<Name>/).
This COPIES (does not delete the source) and is safe to re-run: it skips files that
already exist at the destination with the same size, so an interrupted run resumes.

Layout produced:   DST_DIR/temp/temp.01.01.2000.dta
                   DST_DIR/phy/phy.01.01.2000.dta   ... etc.

Run DRY_RUN=True first to see what it *would* do; then set DRY_RUN=False.
"""
from __future__ import annotations
import os
import sys
import shutil
import time

# ---------------- CONFIG (set to your mounted drive paths) ----------------
# Find your drive names with:  ls /Volumes
SRC_DIR = "/Volumes/OFES_SOURCE"            # connected drive that HAS the .dta files
DST_DIR = "/Volumes/BigDrive/ofes_bgc"      # the attached 5 TB drive (destination)
VARS = ["temp", "phy", "ntr", "sw", "hmxl", "htflx", "pressur"]  # which to copy
DRY_RUN = True                              # True = report only, no writing


def infer_var(fname: str):
    """Map a filename to its variable by prefix, e.g. 'phy.01.01.2000.dta' -> 'phy'."""
    base = os.path.basename(fname).lower()
    for v in VARS:
        if base.startswith(v + "."):
            return v
    return None


def main() -> int:
    if not os.path.isdir(SRC_DIR):
        sys.exit(f"SOURCE not found: {SRC_DIR}  (check `ls /Volumes`)")

    # 1. scan source recursively for .dta files we recognize
    todo = []            # (src, dst, size)
    per_var = {v: 0 for v in VARS}
    for root, _, files in os.walk(SRC_DIR):
        for f in files:
            if not f.lower().endswith(".dta"):
                continue
            v = infer_var(f)
            if v is None:
                continue
            src = os.path.join(root, f)
            dst = os.path.join(DST_DIR, v, f)
            todo.append((src, dst, os.path.getsize(src)))
            per_var[v] += 1

    total = sum(s for *_, s in todo)
    print(f"found {len(todo)} .dta files ({total/1e9:.1f} GB) under {SRC_DIR}")
    for v in VARS:
        if per_var[v]:
            print(f"   {v:8s}: {per_var[v]} files")

    # 2. destination free-space guard
    if os.path.isdir(DST_DIR) or not DRY_RUN:
        os.makedirs(DST_DIR, exist_ok=True)
        free = shutil.disk_usage(DST_DIR).free
        print(f"destination free space: {free/1e9:.0f} GB")
        if free < total * 1.05:
            print("  WARNING: destination may not have enough free space!")

    # 3. copy (skip files already present with matching size)
    copied = skipped = 0
    bytes_copied = 0
    t0 = time.time()
    for src, dst, size in todo:
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            skipped += 1
            continue
        if DRY_RUN:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)                       # preserves metadata
        if os.path.getsize(dst) != size:            # verify
            sys.exit(f"SIZE MISMATCH after copy: {dst}")
        copied += 1
        bytes_copied += size
        if copied % 50 == 0:
            mins = (time.time() - t0) / 60
            print(f"  copied {copied} files, {bytes_copied/1e9:.1f} GB, {mins:.1f} min")

    tag = "(dry-run) " if DRY_RUN else ""
    print(f"DONE {tag}: to-copy={len(todo)-skipped}, already-present={skipped}, "
          f"copied={copied}")
    if DRY_RUN:
        print("Set DRY_RUN=False to actually copy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
