#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time
from xml.parsers.expat import model
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix
import seaborn as sns
#from tensorflow.keras import regularizers


# ====== Try CfC ======
_HAS_NCPS = False
try:
    from ncps.tf import CfC
    _HAS_NCPS = True
except Exception:
    _HAS_NCPS = False

AUTOTUNE = tf.data.AUTOTUNE

# =============================== CONFIG ===============================
CSV_PATH   = "/home/newin/Projects/AMR_using_LNN/dataset_clear/dataset_3_new20/combined_dataset_ALL__clean.csv"
IMAGE_DIR  = "/home/newin/Projects/AMR_using_LNN/dataset_clear/dataset_3_new20/all_images_merged_ALL_gray128"


ROOT_MODEL_DIR = "/home/newin/Projects/AMR_using_LNN/models"
ROOT_LOG_DIR   = "/home/newin/Projects/AMR_using_LNN/log_dir"

LIDAR_MAX_RANGE = 10.0
MAX_GOAL_DIST   = 50.0

SEQ_LEN        = 32
STRIDE         = 1
BATCH_SIZE     = 32
EPOCHS         = 50
LEARNING_RATE  = 1e-3
#L2            = 1e-4
VAL_SPLIT      = 0.1
MIXED_PRECISION= True
JITTER         = True
SEED           = 42

IMG_SHAPE = (128, 128, 1)
LIDAR_DIM = 67
STATE_BASE_DIM = 5
CFC1_UNITS = 64

# ===================== Experiment switches =====================
USE_IMAGE = True
USE_LIDAR = True
USE_STATE = True

# New:
#   - "A"/"B"/"C" => single goal => NO onehot => STATE_DIM=5
#   - "ALL"       => multi-goal  => USE onehot => STATE_DIM=5+N_GOALS
GOAL_MODE = ["A"]      # later: ["A","B","C"]
# GOAL_MODE = "ALL"


EPISODE_LIMIT = 15
EPISODE_PICK  = "first"
EPISODE_SEED  = 123

RNN_BACKEND = "CFC"   # "CFC" or "GRU"

LIDAR_COLS  = [f"lidar_{i}" for i in range(LIDAR_DIM)]
STATE_COLS  = ["dX", "dY", "sin_dYaw", "cos_dYaw", "dist_to_goal"]
TARGET_COLS = ["cmd_linear_vel", "cmd_angular_vel"]

def make_run_folder(goal_mode: str, use_goal_onehot: bool):
    rnn = "CFC" if (RNN_BACKEND.upper() == "CFC" and _HAS_NCPS) else "GRU"
    tag = "OH1" if use_goal_onehot else "OH0"
    return f"SR03_g{goal_mode}_eps15_IMG{int(USE_IMAGE)}_LID{int(USE_LIDAR)}_STA{int(USE_STATE)}_{rnn}{CFC1_UNITS}_{tag}"

# ---------------- GPU & precision setup ----------------
def setup_gpu(mixed_precision: bool):
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        if gpus:
            print(f"[GPU] Visible GPUs: {len(gpus)}", flush=True)
    except Exception as e:
        print("[GPU] note:", e, flush=True)

    if mixed_precision:
        try:
            from tensorflow.keras import mixed_precision as mp
            mp.set_global_policy("mixed_float16")
            print("[MP] Mixed precision enabled.", flush=True)
        except Exception as e:
            print("[MP] Could not enable mixed precision:", e, flush=True)

def episode_index_matrix(ep_ids: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    N = len(ep_ids)
    out = []  # FIX: was "N, out = len(ep_ids), []" which is a bug
    i = 0
    while i < N:
        j = i + 1
        while j < N and ep_ids[j] == ep_ids[i]:
            j += 1
        L = j - i
        start, made_any = i, False
        while start + seq_len <= j:
            out.append(np.arange(start, start + seq_len, dtype=np.int32))
            made_any = True
            start += stride

        if L < seq_len:
            base = np.arange(i, j, dtype=np.int32)
            pad = np.full(seq_len - L, j - 1, dtype=np.int32)
            out.append(np.concatenate([base, pad]))
        else:
            # ensure last window touches episode end (common for stride=1 too)
            if not made_any or (j - (start - stride + seq_len)) > 0:
                s = max(j - seq_len, i)
                seq = np.arange(s, s + seq_len, dtype=np.int32)
                seq = np.minimum(seq, j - 1)
                if not made_any or s != (start - stride):
                    out.append(seq)
        i = j

    return np.stack(out, axis=0) if out else np.empty((0, seq_len), dtype=np.int32)


def load_and_preprocess_image(path: tf.Tensor, jitter: bool) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=1, expand_animations=False)  # grayscale tensor [web:1]
    img = tf.image.convert_image_dtype(img, tf.float32)  # -> [0,1]
    img = tf.ensure_shape(img, IMG_SHAPE)

    if jitter:
        img = tf.image.random_brightness(img, max_delta=0.05)
        img = tf.image.random_contrast(img, lower=0.95, upper=1.05)
        img = tf.clip_by_value(img, 0.0, 1.0)
    return img

def build_datasets(csv_path: str,
                   image_dir: str,
                   seq_len: int,
                   stride: int,
                   batch_size: int,
                   val_split: float,
                   seed: int,
                   jitter: bool,
                   goal_mode):

    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}

    def col(name: str) -> str:
        k = name.lower()
        if k not in cols:
            raise KeyError(f"Missing column '{name}'. Found: {list(df.columns)}")
        return cols[k]

    img_col  = col("image_file")
    ep_col   = col("episode_id")
    goal_col = col("goal_id")

    # ---------------- GOAL MODE HANDLING ----------------
    if goal_mode is None:
        goal_modes = ["ALL"]

    elif isinstance(goal_mode, str):
        goal_modes = [goal_mode.strip().upper()]

    else:
        goal_modes = [str(g).strip().upper() for g in goal_mode]

    goal_modes = [g for g in goal_modes if g] or ["ALL"]

    goal_series = df[goal_col].astype(str).str.strip().str.upper()

    # Filter dataset by goal
    if "ALL" not in goal_modes:
        before = len(df)
        df = df[goal_series.isin(goal_modes)]
        after = len(df)

        if after == 0:
            raise ValueError(f"GOAL_MODE={goal_modes} but no rows found.")

        print(f"[FILTER] GOAL_MODE={goal_modes}: {before} -> {after} rows", flush=True)

        goal_series = df[goal_col].astype(str).str.strip().str.upper()

    # ---------------- EPISODE LIMIT ----------------
    selected_episode_ids = None

    if EPISODE_LIMIT is not None:

        ep_series = df[ep_col].astype(str).str.strip()
        unique_eps = sorted(ep_series.unique().tolist())

        if EPISODE_LIMIT > len(unique_eps):
            chosen_eps = unique_eps
        else:
            if str(EPISODE_PICK).lower() == "first":
                chosen_eps = unique_eps[:EPISODE_LIMIT]

            elif str(EPISODE_PICK).lower() == "random":
                rng = np.random.default_rng(EPISODE_SEED)
                chosen_eps = rng.choice(unique_eps,
                                        size=EPISODE_LIMIT,
                                        replace=False).tolist()
            else:
                raise ValueError("EPISODE_PICK must be 'first' or 'random'")

        before = len(df)

        df = df[df[ep_col].astype(str).str.strip().isin(chosen_eps)]

        after = len(df)
        selected_episode_ids = chosen_eps

        print(f"[EP_LIMIT] Episodes kept={len(chosen_eps)} | Rows {before} -> {after}", flush=True)

    # ---------------- IMAGE PATH RESOLUTION ----------------
    def resolve_path(fn: str) -> str:
        return os.path.normpath(
            os.path.join(image_dir,
                         os.path.basename(str(fn).strip().replace("\\", "/")))
        )

    image_paths = [resolve_path(p) for p in df[img_col].astype(str).tolist()]

    missing = [p for p in image_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images. Example: {missing[0]}")

    # ---------------- DATA ARRAYS ----------------
    ep_ids = df[ep_col].to_numpy()

    lidar_np = df[[col(c) for c in LIDAR_COLS]].astype(np.float32).to_numpy()
    base_state_np = df[[col(c) for c in STATE_COLS]].astype(np.float32).to_numpy()

    y_np = df[[col(c) for c in TARGET_COLS]].astype(np.float32).to_numpy()

    # Scale goal distances
    base_state_np[:,0] /= MAX_GOAL_DIST
    base_state_np[:,1] /= MAX_GOAL_DIST
    base_state_np[:,4] /= MAX_GOAL_DIST

    # ---------------- GOAL ENCODING ----------------
    goal_series = df[goal_col].astype(str).str.strip().str.upper()

    if "ALL" in goal_modes:
        goal_ids = sorted(goal_series.unique().tolist())
    else:
        goal_ids = sorted(set(goal_modes))

    use_goal_onehot = len(goal_ids) > 1

    n_goals = len(goal_ids)
    goal2idx = {g:i for i,g in enumerate(goal_ids)}

    if use_goal_onehot:
        goal_idx = goal_series.map(goal2idx).to_numpy()
        goal_onehot_np = np.eye(n_goals, dtype=np.float32)[goal_idx]
    else:
        goal_onehot_np = None

    # ---------------- SEQUENCE GENERATION ----------------
    seq_index = episode_index_matrix(ep_ids, seq_len, stride)

    if seq_index.shape[0] == 0:
        raise RuntimeError("No sequences formed. Check SEQ_LEN/STRIDE vs episode length.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(seq_index))
    seq_index = seq_index[perm]

    val_n = int(len(seq_index) * val_split)

    val_idx = seq_index[:val_n]
    train_idx = seq_index[val_n:]

    train_rows = np.unique(train_idx.reshape(-1))
    eps = 1e-6

    # ---------------- LIDAR NORMALIZATION ----------------
    lidar_np = np.clip(lidar_np,0.0,float(LIDAR_MAX_RANGE))
    lidar_np = lidar_np / float(LIDAR_MAX_RANGE)

    lidar_mean = lidar_np[train_rows].mean(0)
    lidar_std  = lidar_np[train_rows].std(0) + eps

    lidar_norm = (lidar_np - lidar_mean) / lidar_std

    # ---------------- STATE NORMALIZATION ----------------
    state_mean = base_state_np[train_rows].mean(0)
    state_std  = base_state_np[train_rows].std(0) + eps

    base_state_norm = (base_state_np - state_mean) / state_std

    # Final state vector
    if use_goal_onehot:
        state_full_np = np.concatenate([base_state_norm, goal_onehot_np], axis=1).astype(np.float32)
    else:
        state_full_np = base_state_norm.astype(np.float32)

    state_dim = int(state_full_np.shape[1])
    state_base_dim = int(base_state_norm.shape[1])

    # ---------------- TF CONSTANTS ----------------
    paths_t  = tf.constant(image_paths)
    lidar_t  = tf.constant(lidar_norm, dtype=tf.float32)
    state_t  = tf.constant(state_full_np, dtype=tf.float32)
    target_t = tf.constant(y_np, dtype=tf.float32)

    # ---------------- DATASET MAPPER ----------------
    def make_mapper(is_training):

        def _map(index_vec):

            seq_paths = tf.gather(paths_t, index_vec)

            img_seq = tf.map_fn(
                lambda p: load_and_preprocess_image(p,
                                                    jitter=is_training and jitter),
                seq_paths,
                fn_output_signature=tf.TensorSpec(shape=IMG_SHAPE,
                                                   dtype=tf.float32),
                parallel_iterations=16
            )

            lidar_seq = tf.gather(lidar_t, index_vec)
            state_seq = tf.gather(state_t, index_vec)
            y_seq     = tf.gather(target_t, index_vec)

            return (img_seq, lidar_seq, state_seq), y_seq

        return _map

    def as_ds(index_mat, shuffle):

        ds = tf.data.Dataset.from_tensor_slices(index_mat)

        if shuffle:
            ds = ds.shuffle(buffer_size=max(1,len(index_mat)),
                            seed=seed,
                            reshuffle_each_iteration=True)

        return ds

    train_ds = (
        as_ds(train_idx, True)
        .map(make_mapper(True), num_parallel_calls=AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        as_ds(val_idx, False)
        .map(make_mapper(False), num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    # ---------------- STATS ----------------
    stats = {
        "num_sequences_total": int(len(seq_index)),
        "num_sequences_train": int(len(train_idx)),
        "num_sequences_val": int(len(val_idx)),
        "seq_len": int(seq_len),
        "batch_size": int(batch_size),
        "lidar_mean": lidar_mean,
        "lidar_std": lidar_std,
        "state_mean": state_mean,
        "state_std": state_std,
        "state_base_dim": state_base_dim,
        "use_goal_onehot": bool(use_goal_onehot),
        "n_goals": int(n_goals),
        "state_dim": int(state_dim),
        "goal_ids": goal_ids,
        "goal2idx": goal2idx,
        "goal_mode": "ALL" if "ALL" in goal_modes else "+".join(goal_ids),
        "episode_limit": EPISODE_LIMIT,
        "episode_pick": EPISODE_PICK if EPISODE_LIMIT is not None else None,
        "episode_seed": EPISODE_SEED if (EPISODE_LIMIT is not None and str(EPISODE_PICK).lower()=="random") else None,
        "selected_episode_ids": selected_episode_ids,
    }

    return train_ds, val_ds, stats

# ---------------- Model blocks ----------------
def build_cnn_encoder(input_shape=(128, 128, 1)):
    img_in = layers.Input(shape=input_shape, name="image_input")
    x = layers.Conv2D(16, 5, strides=2, padding="same", activation="relu")(img_in)
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu", name="cnn_feature")(x)
    return models.Model(img_in, x, name="CNN_Encoder")

def build_lidar_mlp(lidar_dim=67):
    lidar_in = layers.Input(shape=(lidar_dim,), name="lidar_input")
    x = layers.Dense(64, activation="relu")(lidar_in)
    x = layers.Dense(64, activation="relu", name="lidar_feature")(x)
    return models.Model(lidar_in, x, name="LiDAR_MLP")

def build_goal_state_mlp(state_dim, out_dim=32):
    state_in = layers.Input(shape=(state_dim,), name="state_input")
    x = layers.Dense(32, activation="relu")(state_in)
    x = layers.Dense(out_dim, activation="relu", name="state_feature")(x)
    return models.Model(state_in, x, name="GoalState_MLP")

def build_sequence_model(IMG_SHAPE, LIDAR_DIM,
                         STATE_DIM, STATE_BASE_DIM, N_GOALS, SEQ_LEN,
                         CFC1_UNITS,
                         use_image=True, use_lidar=True, use_state=True,
                         use_goal_onehot=True,
                         rnn_backend="CFC"):
    inputs, feats = [], []

    if use_image:
        img_seq = layers.Input(shape=(None,) + IMG_SHAPE, name="image_seq")
        inputs.append(img_seq)
        img_feat = layers.TimeDistributed(build_cnn_encoder(IMG_SHAPE), name="TD_CNN")(img_seq)
        feats.append(img_feat)

    if use_lidar:
        lidar_seq = layers.Input(shape=(None, LIDAR_DIM), name="lidar_seq")
        inputs.append(lidar_seq)
        lidar_feat = layers.TimeDistributed(build_lidar_mlp(LIDAR_DIM), name="TD_LiDAR")(lidar_seq)
        feats.append(lidar_feat)

    goal_seq = None
    if use_state:
        state_seq = layers.Input(shape=(None, STATE_DIM), name="state_seq")
        inputs.append(state_seq)

        cont_seq = layers.Lambda(lambda s: s[..., :STATE_BASE_DIM], name="state_cont")(state_seq)
        state_feat = layers.TimeDistributed(
            build_goal_state_mlp(state_dim=STATE_BASE_DIM, out_dim=32),
            name="TD_State"
        )(cont_seq)
        feats.append(state_feat)

        if use_goal_onehot and (N_GOALS > 1):
            goal_seq = layers.Lambda(lambda s: s[..., STATE_BASE_DIM:], name="goal_onehot")(state_seq)
    else:
        goal_vec = layers.Input(shape=(N_GOALS,), name="goal_vec")
        inputs.append(goal_vec)
        goal_seq = layers.RepeatVector(SEQ_LEN, name="goal_repeat")(goal_vec)

    fused = layers.Concatenate(axis=-1, name="fuse")(feats)
    z = layers.TimeDistributed(layers.Dense(128, activation="relu"), name="pre_rnn")(fused)
    #z = layers.TimeDistributed(layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(L2)),name="pre_rnn")(fused)


    if goal_seq is not None:
        z = layers.Concatenate(axis=-1, name="pre_rnn_with_goal")([z, goal_seq])

    rnn_backend = (rnn_backend or "CFC").upper()
    if rnn_backend == "CFC" and _HAS_NCPS:
        x = CfC(units=CFC1_UNITS, return_sequences=True, name="cfc")(z)
        rnn_used = "CfC"
    else:
        x = layers.GRU(units=CFC1_UNITS, return_sequences=True, name="gru")(z)
        rnn_used = "GRU"

    x = layers.TimeDistributed(layers.Dense(32, activation="relu"), name="head_dense")(x)
    #x = layers.TimeDistributed(layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(L2)),name="head_dense")(x)
    # With mixed precision, keep the final output float32 for numeric stability. [web:9]
    out = layers.TimeDistributed(layers.Dense(2, dtype="float32"), name="vel_output")(x)
    return models.Model(inputs=inputs, outputs=out, name="LNN_modal"), rnn_used

# ---------------- Plots ----------------
def add_right_watermark(ax, text, fontsize=8, alpha=0.5, x=1.02):
    if not text:
        return
    ax.text(
        x, 0.5, str(text),
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="left",
        fontsize=fontsize,
        alpha=alpha,
        color="gray",
        clip_on=False,
    )

# ============================================================
# TRAIN / VAL MAE

def plot_train_val_mae(history, out_dir, run_name, fname="train_val_mae.png"):
    mae  = history.history.get("mae", [])
    vmae = history.history.get("val_mae", [])
    epochs = range(1, len(mae) + 1)

    fig, ax = plt.subplots()
    ax.plot(epochs, mae,  label="train_mae")
    ax.plot(epochs, vmae, label="val_mae")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.set_title("Train vs Val MAE per Epoch")
    ax.legend()
    add_right_watermark(ax, run_name)

    base_path = os.path.join(out_dir, fname.split(".")[0])
    fig.savefig(base_path + ".pdf", bbox_inches="tight")
    fig.savefig(base_path + ".svg", bbox_inches="tight")
    plt.close(fig)

    print(f"[PLOT] {base_path}.pdf / .svg")


# ============================================================
# VAL-TRAIN GAP

def plot_val_train_gap(history, out_dir, run_name, fname="val_train_mae_gap.png"):
    mae  = history.history.get("mae", [])
    vmae = history.history.get("val_mae", [])
    gap  = [v - m for m, v in zip(mae, vmae)]
    epochs = range(1, len(gap) + 1)

    fig, ax = plt.subplots()
    ax.plot(epochs, gap)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val MAE - Train MAE")
    ax.set_title("Val–Train MAE Gap per Epoch")
    add_right_watermark(ax, run_name)

    base_path = os.path.join(out_dir, fname.split(".")[0])
    fig.savefig(base_path + ".pdf", bbox_inches="tight")
    fig.savefig(base_path + ".svg", bbox_inches="tight")
    plt.close(fig)

    print(f"[PLOT] {base_path}.pdf / .svg")


# ============================================================
# PER TIMESTEP MAE

def compute_per_timestep_mae(model, val_ds, seq_len, input_unpack_fn):
    sums   = np.zeros(seq_len, dtype=np.float64)
    counts = np.zeros(seq_len, dtype=np.int64)

    for (img_seq, lidar_seq, state_seq), y_true in val_ds:
        x_in = input_unpack_fn(img_seq, lidar_seq, state_seq)

        if hasattr(y_true, "numpy"):
            y_true = y_true.numpy()

        y_pred = model.predict_on_batch(x_in)
        err = np.abs(y_pred - y_true).mean(axis=2)

        T = err.shape[1]
        sums[:T]   += err.sum(axis=0)
        counts[:T] += err.shape[0]

    return sums / np.maximum(counts, 1)


def plot_per_timestep_mae(model, val_ds, seq_len, out_dir, input_unpack_fn,
                          run_name, fname="per_timestep_val_mae.png"):

    mt = compute_per_timestep_mae(model, val_ds, seq_len, input_unpack_fn)
    steps = range(1, len(mt) + 1)

    fig, ax = plt.subplots()
    ax.plot(steps, mt)
    ax.set_xlabel("Timestep (1 … SEQ_LEN)")
    ax.set_ylabel("Val MAE")
    ax.set_title("Per-timestep Validation MAE")
    add_right_watermark(ax, run_name)

    base_path = os.path.join(out_dir, fname.split(".")[0])
    fig.savefig(base_path + ".pdf", bbox_inches="tight")
    fig.savefig(base_path + ".svg", bbox_inches="tight")
    plt.close(fig)

    print(f"[PLOT] {base_path}.pdf / .svg")


# ============================================================
# SCATTER PLOTS

def scatter_pred_vs_gt(model, val_ds, out_dir, input_unpack_fn,
                       run_name,
                       fname_linear="pred_vs_gt_linear.png",
                       fname_angular="pred_vs_gt_angular.png",
                       max_points=100000):

    y_list, p_list = [], []

    for (img_seq, lidar_seq, state_seq), y_true in val_ds:
        x_in = input_unpack_fn(img_seq, lidar_seq, state_seq)

        if hasattr(y_true, "numpy"):
            y_true = y_true.numpy()

        y_pred = model.predict_on_batch(x_in)

        y_list.append(y_true)
        p_list.append(y_pred)

    y = np.concatenate(y_list, axis=0).reshape(-1, 2)
    p = np.concatenate(p_list, axis=0).reshape(-1, 2)

    if y.shape[0] > max_points:
        idx = np.random.default_rng(42).choice(y.shape[0], size=max_points, replace=False)
        y, p = y[idx], p[idx]

    # -------- Linear --------
    fig, ax = plt.subplots()
    ax.scatter(y[:, 0], p[:, 0], s=2, alpha=0.4)
    lim = [min(y[:, 0].min(), p[:, 0].min()), max(y[:, 0].max(), p[:, 0].max())]
    ax.plot(lim, lim)
    ax.set_xlabel("GT linear")
    ax.set_ylabel("Pred linear")
    ax.set_title("Pred vs GT — Linear")
    add_right_watermark(ax, run_name)

    base_path = os.path.join(out_dir, fname_linear.split(".")[0])
    fig.savefig(base_path + ".pdf", bbox_inches="tight")
    fig.savefig(base_path + ".svg", bbox_inches="tight")
    plt.close(fig)

    print(f"[PLOT] {base_path}.pdf / .svg")

    # -------- Angular --------
    fig, ax = plt.subplots()
    ax.scatter(y[:, 1], p[:, 1], s=2, alpha=0.4)
    lim = [min(y[:, 1].min(), p[:, 1].min()), max(y[:, 1].max(), p[:, 1].max())]
    ax.plot(lim, lim)
    ax.set_xlabel("GT angular")
    ax.set_ylabel("Pred angular")
    ax.set_title("Pred vs GT — Angular Velocity")
    add_right_watermark(ax, run_name)

    base_path = os.path.join(out_dir, fname_angular.split(".")[0])
    fig.savefig(base_path + ".pdf", bbox_inches="tight")
    fig.savefig(base_path + ".svg", bbox_inches="tight")
    plt.close(fig)

    print(f"[PLOT] {base_path}.pdf / .svg")


# ============================================================
# ERROR HISTOGRAM

def plot_error_histogram(model, val_ds, out_dir, input_unpack_fn, run_name,
                         fname_linear="error_hist_linear.png",
                         fname_angular="error_hist_angular.png"):

    y_list, p_list = [], []

    for (img_seq, lidar_seq, state_seq), y_true in val_ds:
        x_in = input_unpack_fn(img_seq, lidar_seq, state_seq)

        if hasattr(y_true, "numpy"):
            y_true = y_true.numpy()

        y_pred = model.predict_on_batch(x_in)

        y_list.append(y_true)
        p_list.append(y_pred)

    y = np.concatenate(y_list, axis=0).reshape(-1, 2)
    p = np.concatenate(p_list, axis=0).reshape(-1, 2)

    error = p - y

    # -------- Linear --------
    fig, ax = plt.subplots()
    ax.hist(error[:, 0], bins=50)
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Frequency")
    ax.set_title("Linear Velocity Error Histogram")
    add_right_watermark(ax, run_name)

    base_path = os.path.join(out_dir, fname_linear.split(".")[0])
    fig.savefig(base_path + ".pdf", bbox_inches="tight")
    fig.savefig(base_path + ".svg", bbox_inches="tight")
    plt.close(fig)

    print(f"[PLOT] {base_path}.pdf / .svg")

    # -------- Angular --------
    fig, ax = plt.subplots()
    ax.hist(error[:, 1], bins=50)
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Frequency")
    ax.set_title("Angular Velocity Error Histogram")
    add_right_watermark(ax, run_name)

    base_path = os.path.join(out_dir, fname_angular.split(".")[0])
    fig.savefig(base_path + ".pdf", bbox_inches="tight")
    fig.savefig(base_path + ".svg", bbox_inches="tight")
    plt.close(fig)

    print(f"[PLOT] {base_path}.pdf / .svg")


def save_predictions(model, val_ds, out_dir, input_unpack_fn,
                     fname="predictions_vs_gt.csv"):

    y_list = []
    p_list = []

    for (img_seq, lidar_seq, state_seq), y_true in val_ds:

        x_in = input_unpack_fn(img_seq, lidar_seq, state_seq)

        if hasattr(y_true, "numpy"):
            y_true = y_true.numpy()

        y_pred = model.predict_on_batch(x_in)

        y_list.append(y_true)
        p_list.append(y_pred)

    y = np.concatenate(y_list, axis=0).reshape(-1,2)
    p = np.concatenate(p_list, axis=0).reshape(-1,2)

    df = pd.DataFrame({
        "gt_linear": y[:,0],
        "gt_angular": y[:,1],
        "pred_linear": p[:,0],
        "pred_angular": p[:,1]
    })

    save_path = os.path.join(out_dir, fname)
    df.to_csv(save_path, index=False)

    print(f"[SAVE] Predictions saved -> {save_path}")

def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    setup_gpu(MIXED_PRECISION)

    train_ds, val_ds, stats = build_datasets(
        csv_path=CSV_PATH,
        image_dir=IMAGE_DIR,
        seq_len=SEQ_LEN,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        seed=SEED,
        jitter=JITTER,
        goal_mode=GOAL_MODE,
    )

    state_dim = int(stats["state_dim"])
    state_base_dim = int(stats["state_base_dim"])
    n_goals = int(stats["n_goals"])
    goal_ids = stats["goal_ids"]
    use_goal_onehot = bool(stats["use_goal_onehot"])

    print(f"[DATA] goal_mode={stats['goal_mode']} | use_goal_onehot={use_goal_onehot} | "
          f"state_dim={state_dim} | goal_ids={goal_ids}", flush=True)

    RUN_FOLDER = make_run_folder(stats["goal_mode"], use_goal_onehot)
    MODEL_DIR = os.path.join(ROOT_MODEL_DIR, RUN_FOLDER)
    LOG_DIR = os.path.join(ROOT_LOG_DIR, RUN_FOLDER)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    SAVE_PATH = os.path.join(MODEL_DIR, f"{RUN_FOLDER}.keras")
    WEIGHTS_PATH = os.path.join(MODEL_DIR, f"{RUN_FOLDER}.weights.h5")

    model, rnn_used = build_sequence_model(
        IMG_SHAPE=IMG_SHAPE,
        LIDAR_DIM=LIDAR_DIM,
        STATE_DIM=state_dim,
        STATE_BASE_DIM=state_base_dim,
        N_GOALS=n_goals,
        SEQ_LEN=SEQ_LEN,
        CFC1_UNITS=CFC1_UNITS,
        use_image=USE_IMAGE,
        use_lidar=USE_LIDAR,
        use_state=USE_STATE,
        use_goal_onehot=use_goal_onehot,
        rnn_backend=RNN_BACKEND,
    )

    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])
    model.summary()

    def input_unpack(img_seq, lidar_seq, state_seq):
        xs = ()
        if USE_IMAGE:
            xs += (img_seq,)
        if USE_LIDAR:
            xs += (lidar_seq,)
        if USE_STATE:
            xs += (state_seq,)
        else:
            if use_goal_onehot:
                goal_vec = state_seq[:, 0, state_base_dim:]   # (B,n_goals)
            else:
                goal_vec = tf.ones((tf.shape(state_seq)[0], 1), dtype=tf.float32)
            xs += (goal_vec,)
        return xs[0] if len(xs) == 1 else xs

    train_ds2 = train_ds.map(lambda x, y: (input_unpack(*x), y), num_parallel_calls=AUTOTUNE)
    val_ds2   = val_ds.map(lambda x, y: (input_unpack(*x), y), num_parallel_calls=AUTOTUNE)

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=WEIGHTS_PATH, monitor="val_mae", mode="min",
        save_best_only=True, save_weights_only=True, verbose=1
    )
    es = tf.keras.callbacks.EarlyStopping(monitor="val_mae", mode="min", patience=6, restore_best_weights=True)
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae", mode="min", factor=0.5, patience=2,
                                                min_lr=1e-6, verbose=1)
    tb = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, write_graph=False)

    print(f"[RNN] Used: {rnn_used}", flush=True)
    t0 = time.time()
    history = model.fit(train_ds2, validation_data=val_ds2, epochs=EPOCHS,
                        callbacks=[es, rlrop, tb, ckpt], verbose=1)
    train_seconds = float(time.time() - t0)

    # Save training history
    hist_csv = os.path.join(MODEL_DIR, "train_history.csv")
    pd.DataFrame(history.history).to_csv(hist_csv, index=False)
    print(f"[SAVE] train_history.csv -> {hist_csv}", flush=True)

    # Save model + weights
    try:
        model.save(SAVE_PATH)
        print(f"[SAVE] Model saved to {SAVE_PATH}", flush=True)
    except Exception as e:
        print("[SAVE] Failed model.save:", e, flush=True)

    model.save_weights(WEIGHTS_PATH)
    print(f"[SAVE] Weights saved to {WEIGHTS_PATH}", flush=True)

    # Final validation metrics
    try:
        val_loss, val_mae = model.evaluate(val_ds2, verbose=0)
        error = ""
    except Exception as e:
        val_loss, val_mae = None, None
        error = str(e)

    plot_train_val_mae(history, MODEL_DIR, RUN_FOLDER)
    plot_val_train_gap(history, MODEL_DIR, RUN_FOLDER)
    plot_per_timestep_mae(model, val_ds, SEQ_LEN, MODEL_DIR, input_unpack_fn=input_unpack, run_name=RUN_FOLDER)
    scatter_pred_vs_gt(model, val_ds, MODEL_DIR, input_unpack_fn=input_unpack, run_name=RUN_FOLDER)
    save_predictions(model, val_ds, MODEL_DIR, input_unpack_fn=input_unpack)
    plot_error_histogram(model, val_ds, MODEL_DIR, input_unpack_fn=input_unpack, run_name=RUN_FOLDER)
    



    exp_stats = {
        "run_folder": RUN_FOLDER,
        "train_seconds": train_seconds,
        "train_minutes": train_seconds / 60.0,
        "num_params": int(model.count_params()),
        "dataset": {
            "num_sequences_total": stats["num_sequences_total"],
            "num_sequences_train": stats["num_sequences_train"],
            "num_sequences_val": stats["num_sequences_val"],
        },
        "val": {
            "val_loss": val_loss,
            "val_mae": val_mae,
            "error": error,
        },
    }

    exp_path = os.path.join(MODEL_DIR, "experiment_stats.json")
    with open(exp_path, "w") as f:
        json.dump(exp_stats, f, indent=2)
    print(f"[SAVE] experiment_stats.json -> {exp_path}", flush=True)

    # FIX: remove invalid "{...}" placeholders; create real content.
    run_info = {
        "RUN_FOLDER": RUN_FOLDER,
        "paths": {
            "CSV_PATH": CSV_PATH,
            "IMAGE_DIR": IMAGE_DIR,
            "MODEL_DIR": MODEL_DIR,
            "LOG_DIR": LOG_DIR,
            "SAVE_PATH": SAVE_PATH,
            "WEIGHTS_PATH": WEIGHTS_PATH,
        },
        "env": {
            "tensorflow_version": tf.__version__,
            "mixed_precision": bool(MIXED_PRECISION),
            "has_ncps": bool(_HAS_NCPS),
        },
        "experiment": {
            "use_image": bool(USE_IMAGE),
            "use_lidar": bool(USE_LIDAR),
            "use_state": bool(USE_STATE),
            "goal_mode": stats["goal_mode"],
            "use_goal_onehot": bool(use_goal_onehot),
            "goal_injection": "pre_rnn" if (use_goal_onehot and n_goals > 1) else "none",
            "episode_limit": EPISODE_LIMIT,
            "episode_pick": str(EPISODE_PICK) if EPISODE_LIMIT is not None else None,
            "episode_seed": int(EPISODE_SEED) if (EPISODE_LIMIT is not None and str(EPISODE_PICK).lower() == "random") else None,
            "selected_episode_ids": stats.get("selected_episode_ids", None),
            "rnn_backend_requested": str(RNN_BACKEND),
            "rnn_used": rnn_used,
        },
        "hyperparams": {
            "SEQ_LEN": SEQ_LEN,
            "STRIDE": STRIDE,
            "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS": EPOCHS,
            "LEARNING_RATE": LEARNING_RATE,
            "VAL_SPLIT": VAL_SPLIT,
            "JITTER": JITTER,
            "SEED": SEED,
            "IMG_SHAPE": list(IMG_SHAPE),
            "LIDAR_DIM": LIDAR_DIM,
            "STATE_BASE_DIM": state_base_dim,
            "N_GOALS": n_goals,
            "STATE_DIM": state_dim,
            "CFC1_UNITS": CFC1_UNITS,
        },
        "scaler": {
            "lidar_mean": stats["lidar_mean"].tolist(),
            "lidar_std": stats["lidar_std"].tolist(),
            "state_mean": stats["state_mean"].tolist(),
            "state_std": stats["state_std"].tolist(),
            "lidar_max_range": float(LIDAR_MAX_RANGE),
            "max_goal_dist": float(MAX_GOAL_DIST),
        },
        "goals": {
            "ids": stats["goal_ids"],
            "goal2idx": stats["goal2idx"],
        }
    }

    run_info_path = os.path.join(MODEL_DIR, "run_info.json")
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"[SAVE] run_info.json -> {run_info_path}", flush=True)

if __name__ == "__main__":
    main()
