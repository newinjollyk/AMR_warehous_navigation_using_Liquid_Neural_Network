#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===========================================================
THESIS PLOT REGENERATOR
===========================================================

This script regenerates plots from:

1. train_history.csv
2. predictions_vs_gt.csv

Generated plots:
-----------------------------------------------------------
1. Train vs Validation MAE
2. Validation Gap
3. Scatter Plot (Linear)
4. Scatter Plot (Angular)
5. Error Histogram (Linear)
6. Error Histogram (Angular)

Outputs:
-----------------------------------------------------------
Plots are automatically saved in:

/home/newin/Projects/AMR_using_LNN/Generated_plots/<RUN_FOLDER_NAME>/

Formats:
-----------------------------------------------------------
PNG
PDF
SVG

Each plot contains:
-----------------------------------------------------------
Watermark extracted from RUN_FOLDER name:
Example:
IMG0_LID1_STA0

===========================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# USER INPUT
# ==========================================================

# Paste your RUN FOLDER here
RUN_FOLDER = r"/home/newin/Projects/AMR_using_LNN/models/SR03_gA_eps15_IMG0_LID0_STA1_CFC64_OH0"

# ==========================================================
# OUTPUT ROOT DIRECTORY
# ==========================================================

OUTPUT_ROOT = r"/home/newin/Projects/AMR_using_LNN/Generated_plots"

# ==========================================================
# FONT SETTINGS
# ==========================================================

TITLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 14
LEGEND_FONT_SIZE = 14

# ==========================================================
# FIGURE SETTINGS
# ==========================================================

FIG_WIDTH = 8
FIG_HEIGHT = 6
DPI = 300

# ==========================================================
# CREATE OUTPUT DIRECTORY
# ==========================================================

run_name = os.path.basename(os.path.normpath(RUN_FOLDER))

SAVE_DIR = os.path.join(OUTPUT_ROOT, run_name)

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================================
# WATERMARK EXTRACTION
# ==========================================================

parts = run_name.split("_")

watermark_parts = []

for p in parts:
    p_upper = p.upper()

    if (
        p_upper.startswith("IMG")
        or p_upper.startswith("LID")
        or p_upper.startswith("STA")
    ):
        watermark_parts.append(p_upper)

WATERMARK_TEXT = "_".join(watermark_parts)

print("\n===================================================")
print(f"[INFO] RUN NAME  : {run_name}")
print(f"[INFO] SAVE DIR  : {SAVE_DIR}")
print(f"[INFO] WATERMARK : {WATERMARK_TEXT}")
print("===================================================\n")

# ==========================================================
# FILE PATHS
# ==========================================================

history_csv = os.path.join(RUN_FOLDER, "train_history.csv")
pred_csv = os.path.join(RUN_FOLDER, "predictions_vs_gt.csv")

# ==========================================================
# CHECK FILES
# ==========================================================

if not os.path.exists(history_csv):
    raise FileNotFoundError(f"\nMissing file:\n{history_csv}")

if not os.path.exists(pred_csv):
    raise FileNotFoundError(f"\nMissing file:\n{pred_csv}")

# ==========================================================
# LOAD DATA
# ==========================================================

history_df = pd.read_csv(history_csv)
pred_df = pd.read_csv(pred_csv)

print("[INFO] CSV files loaded successfully.\n")

# ==========================================================
# GLOBAL FONT SETTINGS
# ==========================================================

plt.rcParams["font.size"] = LABEL_FONT_SIZE
plt.rcParams["axes.titlesize"] = TITLE_FONT_SIZE
plt.rcParams["axes.labelsize"] = LABEL_FONT_SIZE
plt.rcParams["xtick.labelsize"] = TICK_FONT_SIZE
plt.rcParams["ytick.labelsize"] = TICK_FONT_SIZE
plt.rcParams["legend.fontsize"] = LEGEND_FONT_SIZE

# ==========================================================
# WATERMARK FUNCTION
# ==========================================================

def add_watermark(ax):

    ax.text(
        1.02,
        0.5,
        WATERMARK_TEXT,
        transform=ax.transAxes,
        rotation=90,
        fontsize=10,
        color="gray",
        alpha=0.5,
        va="center",
        ha="left"
    )

# ==========================================================
# SAVE FUNCTION
# ==========================================================

def save_plot(fig, filename):

    png_path = os.path.join(SAVE_DIR, f"{filename}.png")
    pdf_path = os.path.join(SAVE_DIR, f"{filename}.pdf")
    svg_path = os.path.join(SAVE_DIR, f"{filename}.svg")

    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    print(f"[SAVED]")
    print(f"  PNG : {png_path}")
    print(f"  PDF : {pdf_path}")
    print(f"  SVG : {svg_path}\n")

# ==========================================================
# 1. TRAIN VS VALIDATION MAE
# ==========================================================

epochs = range(1, len(history_df) + 1)

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

ax.plot(
    epochs,
    history_df["mae"].to_numpy(),
    label="Train MAE"
)

ax.plot(
    epochs,
    history_df["val_mae"].to_numpy(),
    label="Validation MAE"
)

ax.set_xlabel("Epoch")
ax.set_ylabel("MAE")
ax.set_title("Train vs Validation MAE")

ax.legend()


add_watermark(ax)

save_plot(fig, "train_vs_validation_mae")

plt.close(fig)

# ==========================================================
# 2. VALIDATION GAP
# ==========================================================

gap = (
    history_df["val_mae"].to_numpy()
    - history_df["mae"].to_numpy()
)

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

ax.plot(epochs, gap)

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation MAE - Train MAE")
ax.set_title("Validation Gap")



add_watermark(ax)

save_plot(fig, "validation_gap")

plt.close(fig)

# ==========================================================
# LOAD PREDICTION DATA
# ==========================================================

gt_linear = pred_df["gt_linear"].to_numpy()
gt_angular = pred_df["gt_angular"].to_numpy()

pred_linear = pred_df["pred_linear"].to_numpy()
pred_angular = pred_df["pred_angular"].to_numpy()

# ==========================================================
# 3. SCATTER PLOT - LINEAR
# ==========================================================

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

ax.scatter(gt_linear, pred_linear, s=5, alpha=0.5)

lims = [
    min(gt_linear.min(), pred_linear.min()),
    max(gt_linear.max(), pred_linear.max())
]

ax.plot(lims, lims)

ax.set_xlabel("Ground Truth Linear Velocity")
ax.set_ylabel("Predicted Linear Velocity")
ax.set_title("Prediction vs Ground Truth (Linear)")



add_watermark(ax)

save_plot(fig, "scatter_linear")

plt.close(fig)

# ==========================================================
# 4. SCATTER PLOT - ANGULAR
# ==========================================================

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

ax.scatter(gt_angular, pred_angular, s=5, alpha=0.5)

lims = [
    min(gt_angular.min(), pred_angular.min()),
    max(gt_angular.max(), pred_angular.max())
]

ax.plot(lims, lims)

ax.set_xlabel("Ground Truth Angular Velocity")
ax.set_ylabel("Predicted Angular Velocity")
ax.set_title("Prediction vs Ground Truth (Angular)")



add_watermark(ax)

save_plot(fig, "scatter_angular")

plt.close(fig)

# ==========================================================
# 5. ERROR HISTOGRAM - LINEAR
# ==========================================================

linear_error = pred_linear - gt_linear

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

ax.hist(linear_error, bins=50)

ax.set_xlabel("Prediction Error")
ax.set_ylabel("Frequency")
ax.set_title("Linear Velocity Error Histogram")



add_watermark(ax)

save_plot(fig, "histogram_linear")

plt.close(fig)

# ==========================================================
# 6. ERROR HISTOGRAM - ANGULAR
# ==========================================================

angular_error = pred_angular - gt_angular

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

ax.hist(angular_error, bins=50)

ax.set_xlabel("Prediction Error")
ax.set_ylabel("Frequency")
ax.set_title("Angular Velocity Error Histogram")



add_watermark(ax)

save_plot(fig, "histogram_angular")

plt.close(fig)

# ==========================================================
# FINISHED
# ==========================================================

print("===================================================")
print(" ALL PLOTS GENERATED SUCCESSFULLY")
print("===================================================")

print(f"\nSaved inside:\n{SAVE_DIR}\n")