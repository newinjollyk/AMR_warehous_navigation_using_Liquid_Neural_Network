# AMR Navigation using LNN/CfC

This project implements an Autonomous Mobile Robot (AMR) navigation system using:

- Liquid Neural Networks (CfC)
- GRU baseline models
- LiDAR, Camera, and Goal-State inputs
- Sequence learning for velocity prediction

The framework supports:
- Data collection
- Dataset preprocessing
- Model training
- Evaluation and plot generation

---

# Project Structure

```text
dataset_clear/        -> Dataset and processed images
models/               -> Saved trained models
log_dir/              -> TensorBoard logs
ws_warehouse/         -> ROS2 workspace

train_Experiment.py   -> Main training script
lnn_preprocessing.py  -> Dataset preprocessing
combine_csv.py        -> Combine recorded CSV files
plot_generator.py     -> Thesis plot generator
path_draw.py          -> Path visualization
```

---

# ROS2 Workspace

The ROS2 workspace is located inside:

```text
ws_warehouse/
```

It contains the main runtime nodes:

| Node | Purpose |
|---|---|
| Recorder Node | Records sensor data and robot commands |
| Safety Node | Collision and safety handling |
| Prediction Node | Runs trained LNN/CfC inference for navigation |

---

# Workflow

## 1. Data Collection

Launch the warehouse simulation and run the recorder node.

Recorded data includes:
- Camera images
- LiDAR scans
- Goal-state information
- Velocity commands

Generated outputs:

```text
CSV logs
Image folders
Recorded trajectories
```

---

## 2. Combine CSV Files

Combine all recorded CSV files into a single dataset.

Run:

```bash
python3 combine_csv.py
```

---

## 3. Dataset Preprocessing

Preprocess:
- Images
- LiDAR data
- Goal-state features

Run:

```bash
python3 lnn_preprocessing.py
```

Outputs:

```text
Cleaned dataset CSV
Processed grayscale images
```

---

## 4. Model Training

Train the LNN/CfC or GRU model.

Run:

```bash
python3 train_Experiment.py
```

Outputs:
- Trained models
- Weights
- Training logs
- Prediction CSV files
- Evaluation plots

Saved inside:

```text
models/<RUN_FOLDER>/
```

---

## 5. Plot Generation

Regenerate thesis-quality plots from saved experiment CSV files.

Run:

```bash
python3 plot_generator.py
```

Generated plots:
- Train vs Validation MAE
- Validation Gap
- Scatter plots
- Error histograms

Saved inside:

```text
Generated_plots/<RUN_FOLDER>/
```

Formats:
- PNG
- PDF
- SVG

---

# TensorBoard (Optional)

Monitor training using:

```bash
tensorboard --logdir log_dir
```

Open in browser:

```text
http://localhost:6006
```

---

# Requirements

Main libraries:
- TensorFlow
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- ncps (CfC)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Notes

- Supports multimodal configurations:
  - Image only
  - LiDAR only
  - State only
  - Fusion models

- Experiment configurations are automatically saved for reproducibility.

- Generated plots are thesis/publication ready.
