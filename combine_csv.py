import os
import pandas as pd
import shutil

# ---------------- CONFIG ----------------
ROOT = "/home/newin/Projects/warehouse/dataset/Gaol_A_B_C"   # your root folder
OUTPUT_CSV = os.path.join(ROOT, "combined_dataset_ALL.csv")
OUTPUT_IMG_DIR = os.path.join(ROOT, "all_images_merged_ALL")
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
# ----------------------------------------

all_rows = []

for folder in sorted(os.listdir(ROOT)):
    folder_path = os.path.join(ROOT, folder)

    # only process directories
    if not os.path.isdir(folder_path):
        continue

    # ---- FIND CSV FILE IN THIS EPISODE FOLDER ----
    csv_file = None
    for f in os.listdir(folder_path):
        if f.lower().endswith(".csv"):
            csv_file = os.path.join(folder_path, f)
            break

    if csv_file is None:
        print(f"[WARN] No CSV found in: {folder}")
        continue

    print(f"[CSV] Loading: {csv_file}")

    # ✅ Read CSV as-is — NO columns dropped, NO changes
    df = pd.read_csv(csv_file)
    all_rows.append(df)

    # ---- COPY IMAGES WITHOUT RENAMING ----
    img_dir = os.path.join(folder_path, "images")
    if os.path.isdir(img_dir):
        print(f"[IMG] Copying images from: {img_dir}")
        for img_name in os.listdir(img_dir):
            src = os.path.join(img_dir, img_name)
            if not os.path.isfile(src):
                continue

            # ❗ Keep original name (no folder prefix)
            dst = os.path.join(OUTPUT_IMG_DIR, img_name)

            # Optional: warn if there *is* a duplicate name
            if os.path.exists(dst):
                print(f"[WARN] Image already exists, skipping: {dst}")
                continue

            shutil.copy2(src, dst)

# ---- SAVE COMBINED CSV ----
if all_rows:
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Combined CSV saved to: {OUTPUT_CSV}")
else:
    print("\n❌ No CSV files found — combined CSV not created.")

print(f"✅ All images copied to: {OUTPUT_IMG_DIR}")
