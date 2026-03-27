import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GOAL_RADIUS = 2.3
VEL_THRESHOLD = 0.05

BASE_DIR = "/home/newin/Projects/AMR_using_LNN/path_cal"

GROUPS = [
    "cfc_no_safety",
    "cfc_safety",
    "gru_no_safety",
    "gru_safety",
    "Goal_C_cfc_safety",
    "Goal_B_cfc_safety",
    "Goal_A_B_cfc_safety"
]
def extract_segment(df):
    # Convert required columns to numeric
    for col in ["x", "y", "t_unix", "dist_to_goal", "v_cmd_lin"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    df = df.sort_values("t_unix").reset_index(drop=True)

    if len(df) < 5:
        return None

    # Check if goal is reached anywhere in the run
    if not (df["dist_to_goal"] < GOAL_RADIUS).any():
        return None  # Not a successful run

    # Find start of movement
    moving_indices = df.index[np.abs(df["v_cmd_lin"]) > VEL_THRESHOLD]
    if len(moving_indices) == 0:
        return None

    start_idx = moving_indices[0]

    # Find first goal reach AFTER movement starts
    goal_indices = df.index[df["dist_to_goal"] < GOAL_RADIUS]
    goal_indices = goal_indices[goal_indices >= start_idx]

    if len(goal_indices) == 0:
        return None

    goal_idx = goal_indices[0]

    return df.iloc[start_idx:goal_idx+1]


for group in GROUPS:
    folder_path = os.path.join(BASE_DIR, group)
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    plt.figure(figsize=(8, 8))

    successful_runs = 0
    total_runs = len(csv_files)

    for file in csv_files:
        df = pd.read_csv(file)
        segment = extract_segment(df)

        if segment is None:
            continue  # Skip unsuccessful runs

        x = segment["x"].values
        y = segment["y"].values

        plt.plot(x, y, linewidth=1, alpha=0.6)
        successful_runs += 1

    if successful_runs == 0:
        print(f"No successful runs in {group}")
        plt.close()
        continue

    # Plot formatting
    plt.title(f"{group} - {successful_runs}/{total_runs} Successful Runs")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.axis("equal")
    plt.grid(False)

    # Save figure
    base_path = os.path.join(BASE_DIR, f"{group}_paths")

    plt.savefig(base_path + ".pdf", bbox_inches="tight")
    plt.savefig(base_path + ".svg", bbox_inches="tight")
    plt.close()

    print(f"{group}: {successful_runs}/{total_runs} successful runs")
    print(f"Saved plot at {base_path}")