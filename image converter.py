#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
from pathlib import Path

SRC_DIR = "/home/newin/Projects/warehouse/dataset_clear/dataset_3_new20/all_images_merged_ALL"
DST_DIR = "/home/newin/Projects/warehouse/dataset_clear/dataset_3_new20/all_images_merged_ALL_gray128"

SIZE = (128, 128)  # (W,H) for OpenCV

# Extensions to process
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def main():
    src = Path(SRC_DIR)
    dst = Path(DST_DIR)
    dst.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_ok = 0
    n_skip = 0
    n_fail = 0

    for p in src.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in EXTS:
            continue

        n_total += 1
        out_path = dst / p.name  # same filename (basename)

        # If you want to avoid recomputing:
        if out_path.exists():
            n_skip += 1
            continue

        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[FAIL] unreadable: {p}")
            n_fail += 1
            continue

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)  # matches your ROS callback [web:82]
        resized = cv2.resize(gray, SIZE, interpolation=cv2.INTER_LINEAR)  # matches your ROS callback [web:46]

        ok = cv2.imwrite(str(out_path), resized)  # writes single-channel grayscale [web:119]
        if not ok:
            print(f"[FAIL] cannot write: {out_path}")
            n_fail += 1
            continue

        n_ok += 1
        if (n_ok % 1000) == 0:
            print(f"[PROGRESS] ok={n_ok} / total={n_total}")

    print(f"[DONE] total={n_total} ok={n_ok} skip={n_skip} fail={n_fail}")
    print(f"[OUT]  {DST_DIR}")

if __name__ == "__main__":
    main()
