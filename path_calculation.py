import os
import glob
import pandas as pd
import numpy as np

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

def compute_path_length_segment(df, start_idx, end_idx):
    segment = df.iloc[start_idx:end_idx+1]
    dx = np.diff(segment["x"].values)
    dy = np.diff(segment["y"].values)
    return np.sum(np.sqrt(dx**2 + dy**2))

summary_results = []
failed_records = []

for group in GROUPS:
    folder_path = os.path.join(BASE_DIR, group)
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    path_lengths = []
    times = []

    for file in csv_files:
        df = pd.read_csv(file)

        # Convert required columns to numeric
        for col in ["dist_to_goal", "x", "y", "t_unix", "v_cmd_lin"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna()
        df = df.sort_values("t_unix").reset_index(drop=True)

        if len(df) < 5:
            failed_records.append([group, os.path.basename(file), "Too short / invalid file"])
            continue

        # 1️⃣ Find movement start
        moving_indices = df.index[np.abs(df["v_cmd_lin"]) > VEL_THRESHOLD]
        if len(moving_indices) == 0:
            failed_records.append([group, os.path.basename(file), "No movement detected"])
            continue

        start_idx = moving_indices[0]

        # 2️⃣ Find first goal reach
        goal_indices = df.index[df["dist_to_goal"] < GOAL_RADIUS]
        goal_indices = goal_indices[goal_indices >= start_idx]

        if len(goal_indices) == 0:
            failed_records.append([group, os.path.basename(file), "Goal not reached"])
            continue

        goal_idx = goal_indices[0]

        # 3️⃣ Compute metrics
        path_len = compute_path_length_segment(df, start_idx, goal_idx)
        time_goal = df["t_unix"].iloc[goal_idx] - df["t_unix"].iloc[start_idx]

        path_lengths.append(path_len)
        times.append(time_goal)

    if len(path_lengths) > 0:
        summary_results.append({
            "Group": group,
            "Total Runs": len(csv_files),
            "Successful Runs": len(path_lengths),
            "Mean Path Length (m)": np.mean(path_lengths),
            "Std Path Length (m)": np.std(path_lengths),
            "Mean Time to Goal (s)": np.mean(times),
            "Std Time to Goal (s)": np.std(times)
        })

# Convert to DataFrame
summary_df = pd.DataFrame(summary_results)
failed_df = pd.DataFrame(failed_records, columns=["Group", "File", "Reason"])

# Save to CSV
summary_path = os.path.join(BASE_DIR, "evaluation_summary.csv")
failed_path = os.path.join(BASE_DIR, "failed_runs_report.csv")

summary_df.to_csv(summary_path, index=False)
failed_df.to_csv(failed_path, index=False)

print("\n✅ Summary saved to:", summary_path)
print("✅ Failed runs report saved to:", failed_path)
print("\nFinal Summary:")
print(summary_df.to_string(index=False))