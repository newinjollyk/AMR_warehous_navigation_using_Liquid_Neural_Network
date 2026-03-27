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

def compute_path_length_segment(df, start_idx, end_idx):
    segment = df.iloc[start_idx:end_idx+1]
    dx = np.diff(segment["x"].values)
    dy = np.diff(segment["y"].values)
    return np.sum(np.sqrt(dx**2 + dy**2))

summary_results = []
failed_records = []

# ✅ Store raw values for box plots
all_path_data = {}
all_time_data = {}

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

        # ✅ Store raw data
        all_path_data[group] = path_lengths
        all_time_data[group] = times

# Convert to DataFrame
summary_df = pd.DataFrame(summary_results)
failed_df = pd.DataFrame(failed_records, columns=["Group", "File", "Reason"])

# Save CSVs
summary_path = os.path.join(BASE_DIR, "evaluation_summary.csv")
failed_path = os.path.join(BASE_DIR, "failed_runs_report.csv")

summary_df.to_csv(summary_path, index=False)
failed_df.to_csv(failed_path, index=False)

print("\n✅ Summary saved to:", summary_path)
print("✅ Failed runs report saved to:", failed_path)
print("\nFinal Summary:")
print(summary_df.to_string(index=False))

# ============================================================
# 📊 BOX PLOT FUNCTIONS (THESIS QUALITY)
# ============================================================

def plot_box(data_dict, groups, title, ylabel, save_name):
    data = [data_dict[g] for g in groups if g in data_dict]

    labels = []
    for g in groups:
        if g == "cfc_no_safety":
            labels.append("CfC\n(No Safety)")
        elif g == "cfc_safety":
            labels.append("CfC\n(Safety)")
        elif g == "gru_no_safety":
            labels.append("GRU\n(No Safety)")
        elif g == "gru_safety":
            labels.append("GRU\n(Safety)")
        elif g == "Goal_B_cfc_safety":
            labels.append("Goal B")
        elif g == "Goal_C_cfc_safety":
            labels.append("Goal C")
        elif g == "Goal_A_B_cfc_safety":
            labels.append("Goal A+B")
        else:
            labels.append("Goal A")

    plt.figure(figsize=(9,6))

    box = plt.boxplot(
        data,
        labels=labels[:len(data)],
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='o', markerfacecolor='black', markersize=5)
    )

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    base_path = os.path.join(BASE_DIR, save_name)
    plt.savefig(base_path.replace(".png", ".pdf"))
    plt.savefig(base_path.replace(".png", ".svg"))
    plt.close()

# ============================================================
# 📊 GROUP 1: MODEL COMPARISON
# ============================================================

group1 = [
    "cfc_no_safety",
    "cfc_safety",
    "gru_no_safety",
    "gru_safety"
]

plot_box(
    all_path_data,
    group1,
    "Path Length Distribution (Model Comparison)",
    "Path Length (m)",
    "boxplot_path_model.png"
)

plot_box(
    all_time_data,
    group1,
    "Time-to-Goal Distribution (Model Comparison)",
    "Time to Goal (s)",
    "boxplot_time_model.png"
)

# ============================================================
# 📊 GROUP 2: GOAL GENERALIZATION
# ============================================================

group2 = [
    "cfc_safety",  # Goal A
    "Goal_B_cfc_safety",
    "Goal_C_cfc_safety",
    "Goal_A_B_cfc_safety"
]

plot_box(
    all_path_data,
    group2,
    "Path Length Distribution (Goal Generalization)",
    "Path Length (m)",
    "boxplot_path_goals.png"
)

plot_box(
    all_time_data,
    group2,
    "Time-to-Goal Distribution (Goal Generalization)",
    "Time to Goal (s)",
    "boxplot_time_goals.png"
)

print("\n📊 Box plots saved successfully.")