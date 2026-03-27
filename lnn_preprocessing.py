#!/usr/bin/env python3
import os, re, ast, json
from typing import List
import numpy as np
import pandas as pd

# ======================= CONFIG (EDIT THIS) =======================
INPUT_CSVS = [
    "/home/newin/Projects/warehouse/dataset_clear/dataset_3_new20/combined_dataset_ALL.csv",
]
OUT_DIR = "/home/newin/Projects/warehouse/dataset_clear/dataset_3_new20"
LIDAR_COL = "lidar_points"
LIDAR_MAX_RANGE = 10.0
ROUND_DECIMALS = 4
# ================================================================

# Marker columns we don't need
USE_COLUMNS_DROP = []


def parse_lidar_cell(cell, max_range: float) -> np.ndarray:
    """Turn a string/list cell into a clean 1D float array clamped to [0, max_range]."""
    if isinstance(cell, (list, np.ndarray)):
        arr = np.array(cell, dtype=float)
    else:
        s = str(cell).strip()
        # Replace inf tokens
        s = re.sub(r"\b(inf|Inf|INF|Infinity)\b", str(max_range), s)
        # Ensure brackets
        if not (s.startswith("[") and s.endswith("]")):
            s = "[" + s.strip("[]") + "]"
        try:
            py_list = ast.literal_eval(s)
        except Exception:
            # fallback: split manually
            toks = re.split(r"[,\s]+", s.strip("[]"))
            py_list = []
            for t in toks:
                if t == "":
                    continue
                if re.fullmatch(r"nan|NaN|NAN", t):
                    py_list.append(max_range)
                else:
                    try:
                        py_list.append(float(t))
                    except Exception:
                        py_list.append(max_range)
        arr = np.array(py_list, dtype=float)

    # Replace NaN / ±inf, clamp to [0, max_range]
    arr = np.nan_to_num(arr, nan=max_range, posinf=max_range, neginf=0.0)
    arr = np.clip(arr, 0.0, max_range)
    return arr


def expand_lidar_only(df: pd.DataFrame,
                      lidar_col: str = "lidar_points",
                      lidar_max_range: float = 10.0) -> pd.DataFrame:
    """Expand lidar_points -> lidar_0..N and drop raw lidar + marker cols. No normalization."""
    df_local = df.copy()

    # Drop marker columns if present
    drop_cols = [c for c in USE_COLUMNS_DROP if c in df_local.columns]
    if drop_cols:
        df_local = df_local.drop(columns=drop_cols)

    # Check lidar column
    if lidar_col not in df_local.columns:
        raise ValueError(f"Expected lidar column '{lidar_col}' not found in CSV.")

    # Parse each lidar cell
    lidar_arrays = [parse_lidar_cell(v, max_range=lidar_max_range) for v in df_local[lidar_col]]
    lengths = pd.Series([len(a) for a in lidar_arrays])
    modal_len = int(lengths.mode().iloc[0])

    # Pad/trim to modal length
    fixed_lidar = np.zeros((len(lidar_arrays), modal_len), dtype=float)
    for i, a in enumerate(lidar_arrays):
        if len(a) >= modal_len:
            fixed_lidar[i] = a[:modal_len]
        else:
            pad = np.full(modal_len - len(a), lidar_max_range, dtype=float)
            fixed_lidar[i] = np.concatenate([a, pad], axis=0)

    # LiDAR columns
    lidar_cols = [f"lidar_{i}" for i in range(modal_len)]
    lidar_df = pd.DataFrame(fixed_lidar, columns=lidar_cols, index=df_local.index)

    # Drop original lidar_points and append expanded columns
    df_local = df_local.drop(columns=[lidar_col])
    df_local = pd.concat([df_local, lidar_df], axis=1)

    # Round numeric columns a bit
    num_cols = df_local.select_dtypes(include=[np.number]).columns
    df_local[num_cols] = df_local[num_cols].round(ROUND_DECIMALS)

    return df_local, lidar_cols


def process_file(path: str,
                 lidar_col: str,
                 lidar_max_range: float,
                 out_dir: str):
    df = pd.read_csv(path)

    clean_df, lidar_cols = expand_lidar_only(
        df,
        lidar_col=lidar_col,
        lidar_max_range=lidar_max_range
    )

    base = os.path.splitext(os.path.basename(path))[0]
    out_clean = os.path.join(out_dir, f"{base}__clean.csv")

    clean_df.to_csv(out_clean, index=False)

    # Optional: simple scaler info for LiDAR max range (not used by your train script)
    scaler = {
        "lidar_cols": lidar_cols,
        "lidar_max_range": float(lidar_max_range),
        "note": "Only lidar_points expanded to lidar_0..N. No Z-score normalization. "
                "Your train script does its own scaling."
    }
    out_scal  = os.path.join(out_dir, f"{base}__scaler.json")
    with open(out_scal, "w") as f:
        json.dump(scaler, f, indent=2)

    print(f"[OK] {path}\n -> {out_clean}\n -> {out_scal}")
    return out_clean, out_scal


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for src in INPUT_CSVS:
        if not os.path.isfile(src):
            print(f"[WARN] Skipping missing file: {src}")
            continue
        try:
            process_file(src, LIDAR_COL, LIDAR_MAX_RANGE, OUT_DIR)
        except Exception as e:
            print(f"[ERROR] {src}: {e}")


if __name__ == "__main__":
    main()
