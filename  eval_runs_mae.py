#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, glob
from xml.parsers.expat import model
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# ====== Try CfC ======
_HAS_NCPS = False
try:
    from ncps.tf import CfC
    _HAS_NCPS = True
except Exception:
    _HAS_NCPS = False

AUTOTUNE = tf.data.AUTOTUNE

# ---------------------- CONFIG (paths only) ----------------------
ROOT_MODEL_DIR = "/home/newin/Projects/warehouse/models"
CSV_PATH   = "/home/newin/Projects/warehouse/dataset_clear/dataset_3_new20/combined_dataset_ALL__clean.csv"
IMAGE_DIR  = "/home/newin/Projects/warehouse/dataset_clear/dataset_3_new20/all_images_merged_ALL"

# Must match training defaults (unless overridden by run_info.json)
LIDAR_MAX_RANGE = 10.0
MAX_GOAL_DIST   = 50.0

IMG_SHAPE = (128, 128, 1)
LIDAR_DIM = 67
STATE_BASE_DIM = 5  # fixed in your pipeline

LIDAR_COLS  = [f"lidar_{i}" for i in range(LIDAR_DIM)]
STATE_COLS  = ["dX", "dY", "sin_dYaw", "cos_dYaw", "dist_to_goal"]
TARGET_COLS = ["cmd_linear_vel", "cmd_angular_vel"]

# These are globals used inside build_datasets in your training script.
# We will overwrite them per-run from run_info.json.
GOAL_FILTER = "C"
EPISODE_LIMIT = 15
EPISODE_PICK = "first"
EPISODE_SEED = 123


# ---------------- GPU setup (safe) ----------------
def setup_gpu():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        if gpus:
            print(f"[GPU] Visible GPUs: {len(gpus)}")
    except Exception as e:
        print("[GPU] note:", e)


# ---------------- Episode-safe index builder ----------------
def episode_index_matrix(ep_ids: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    N, out = len(ep_ids), []
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
            if not made_any or (j - (start - stride + seq_len)) > 0:
                s = max(j - seq_len, i)
                seq = np.arange(s, s + seq_len, dtype=np.int32)
                seq = np.minimum(seq, j - 1)
                if not made_any or s != (start - stride):
                    out.append(seq)
        i = j

    return np.stack(out, axis=0) if out else np.empty((0, seq_len), dtype=np.int32)

#-----------Metrics: MAE per output --------------
def eval_per_output_mae(model, val_ds2):
    # collect y_true and y_pred for the whole val set
    y_true_list, y_pred_list = [], []

    for x, y in val_ds2:
        y_pred = model(x, training=False)
        y_true_list.append(y.numpy())
        y_pred_list.append(y_pred.numpy())

    y_true = np.concatenate(y_true_list, axis=0)  # (B, T, 2)
    y_pred = np.concatenate(y_pred_list, axis=0)  # (B, T, 2)

    # flatten over batch and time
    yt = y_true.reshape(-1, 2)
    yp = y_pred.reshape(-1, 2)

    mae_lin = float(np.mean(np.abs(yt[:, 0] - yp[:, 0])))
    mae_ang = float(np.mean(np.abs(yt[:, 1] - yp[:, 1])))
    return mae_lin, mae_ang

# ---------------- Image preprocessing ----------------
def decode_to_gray_128(img_bytes: tf.Tensor) -> tf.Tensor:
    img = tf.image.decode_image(img_bytes, channels=0, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    c = tf.shape(img)[-1]
    img = tf.cond(
        tf.equal(c, 3),
        lambda: tf.image.rgb_to_grayscale(img),
        lambda: tf.cond(tf.equal(c, 4),
                        lambda: tf.image.rgb_to_grayscale(img[..., :3]),
                        lambda: img)
    )
    img = tf.image.resize(img, IMG_SHAPE[:2], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.ensure_shape(img, IMG_SHAPE)
    return img


def load_and_preprocess_image(path: tf.Tensor, jitter: bool) -> tf.Tensor:
    img = decode_to_gray_128(tf.io.read_file(path))
    # IMPORTANT: for evaluation, jitter must be False
    return img


# ---------------- Dataset builder (copied from your training) ----------------
def build_datasets(csv_path: str,
                   image_dir: str,
                   seq_len: int,
                   stride: int,
                   batch_size: int,
                   val_split: float,
                   seed: int,
                   jitter: bool,
                   goal_filter: str = "ALL"):

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

    # goal filter unchanged
    goal_filter = (goal_filter or "ALL").strip()
    if goal_filter.upper() != "ALL":
        df = df[df[goal_col].astype(str).str.strip() == goal_filter]
        if len(df) == 0:
            raise ValueError(f"GOAL_FILTER='{goal_filter}' but no rows found.")
        print(f"[FILTER] GOAL_FILTER='{goal_filter}': rows={len(df)}")

    # ---- Optional episode limiting (uses globals like your training script) ----
    selected_episode_ids = None
    if EPISODE_LIMIT is not None:
        ep_series = df[ep_col].astype(str).str.strip()
        unique_eps = sorted(ep_series.unique().tolist())

        if EPISODE_LIMIT > len(unique_eps):
            chosen_eps = unique_eps
        else:
            if EPISODE_PICK.lower() == "first":
                chosen_eps = unique_eps[:EPISODE_LIMIT]
            elif EPISODE_PICK.lower() == "random":
                rng = np.random.default_rng(EPISODE_SEED)
                chosen_eps = rng.choice(unique_eps, size=EPISODE_LIMIT, replace=False).tolist()
            else:
                raise ValueError("EPISODE_PICK must be 'first' or 'random'")

        df = df[df[ep_col].astype(str).str.strip().isin(chosen_eps)]
        selected_episode_ids = chosen_eps
        print(f"[EP_LIMIT] Episodes kept={len(chosen_eps)} | rows={len(df)}")
    # -------------------------------------------------------------------------

    def resolve_path(fn: str) -> str:
        return os.path.normpath(
            os.path.join(image_dir, os.path.basename(str(fn).strip().replace("\\", "/")))
        )

    image_paths = [resolve_path(p) for p in df[img_col].astype(str).tolist()]
    ep_ids = df[ep_col].to_numpy()

    lidar_np = df[[col(c) for c in LIDAR_COLS]].astype(np.float32).to_numpy()
    base_state_np = df[[col(c) for c in STATE_COLS]].astype(np.float32).to_numpy()

    # scaling
    base_state_np[:, 0] /= MAX_GOAL_DIST
    base_state_np[:, 1] /= MAX_GOAL_DIST
    base_state_np[:, 4] /= MAX_GOAL_DIST

    y_np = df[[col(c) for c in TARGET_COLS]].astype(np.float32).to_numpy()
    state_base_dim = base_state_np.shape[1]

    goal_series = df[goal_col].astype(str).str.strip()
    unique_goals = sorted(goal_series.unique().tolist())
    goal2idx = {g: i for i, g in enumerate(unique_goals)}
    goal_idx = goal_series.map(goal2idx).to_numpy()
    n_goals = len(unique_goals)
    goal_onehot_np = np.eye(n_goals, dtype=np.float32)[goal_idx]

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

    # LiDAR normalization
    lidar_np = np.clip(lidar_np, 0.0, float(LIDAR_MAX_RANGE))
    lidar_np = lidar_np / float(LIDAR_MAX_RANGE)
    lidar_mean = lidar_np[train_rows].mean(0)
    lidar_std = lidar_np[train_rows].std(0) + eps
    lidar_norm = (lidar_np - lidar_mean) / lidar_std

    # State base normalization
    state_mean = base_state_np[train_rows].mean(0)
    state_std = base_state_np[train_rows].std(0) + eps
    base_state_norm = (base_state_np - state_mean) / state_std

    state_full_np = np.concatenate([base_state_norm, goal_onehot_np], axis=1).astype(np.float32)
    state_dim = state_full_np.shape[1]

    paths_t = tf.constant(image_paths)
    lidar_t = tf.constant(lidar_norm, dtype=tf.float32)
    state_t = tf.constant(state_full_np, dtype=tf.float32)
    target_t = tf.constant(y_np, dtype=tf.float32)

    def make_mapper(is_training: bool):
        def _map(index_vec: tf.Tensor):
            seq_paths = tf.gather(paths_t, index_vec)
            img_seq = tf.map_fn(
                lambda p: load_and_preprocess_image(p, jitter=False),  # eval => no jitter
                seq_paths,
                fn_output_signature=tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.float32),
                parallel_iterations=16,
            )
            lidar_seq = tf.gather(lidar_t, index_vec)
            state_seq = tf.gather(state_t, index_vec)
            y_seq = tf.gather(target_t, index_vec)
            return (img_seq, lidar_seq, state_seq), y_seq
        return _map

    def as_ds(index_mat: np.ndarray, shuffle: bool):
        ds = tf.data.Dataset.from_tensor_slices(index_mat)
        if shuffle:
            ds = ds.shuffle(buffer_size=max(1, len(index_mat)), seed=seed, reshuffle_each_iteration=True)
        return ds

    train_ds = (
        as_ds(train_idx, shuffle=True)
        .map(make_mapper(True), num_parallel_calls=AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        as_ds(val_idx, shuffle=False)
        .map(make_mapper(False), num_parallel_calls=AUTOTUNE)
        .batch(batch_size, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

    stats = {
        "state_base_dim": int(state_base_dim),
        "n_goals": int(n_goals),
        "state_dim": int(state_dim),
        "goal2idx": goal2idx,
        "goal_ids": unique_goals,
        "goal_filter": goal_filter,
        "selected_episode_ids": selected_episode_ids,
        "num_sequences_total": int(len(seq_index)),
        "num_sequences_val": int(len(val_idx)),
        "num_sequences_train": int(len(train_idx)),
    }
    return train_ds, val_ds, stats


# ---------------- Model blocks (copied) ----------------
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

    if use_state:
        state_seq = layers.Input(shape=(None, STATE_DIM), name="state_seq")
        inputs.append(state_seq)

        cont_seq = layers.Lambda(lambda s: s[..., :STATE_BASE_DIM], name="state_cont")(state_seq)
        goal_seq = layers.Lambda(lambda s: s[..., STATE_BASE_DIM:], name="goal_onehot")(state_seq)

        state_feat = layers.TimeDistributed(
            build_goal_state_mlp(state_dim=STATE_BASE_DIM, out_dim=32),
            name="TD_State"
        )(cont_seq)
        feats.append(state_feat)
    else:
        goal_vec = layers.Input(shape=(N_GOALS,), name="goal_vec")
        inputs.append(goal_vec)
        goal_seq = layers.RepeatVector(SEQ_LEN, name="goal_repeat")(goal_vec)

    fused = layers.Concatenate(axis=-1, name="fuse")(feats)
    z = layers.TimeDistributed(layers.Dense(128, activation="relu"), name="pre_rnn")(fused)
    z = layers.Concatenate(axis=-1, name="pre_rnn_with_goal")([z, goal_seq])

    rnn_backend = (rnn_backend or "CFC").upper()
    if rnn_backend == "CFC" and _HAS_NCPS:
        x = CfC(units=CFC1_UNITS, return_sequences=True, name="cfc")(z)
    else:
        x = layers.GRU(units=CFC1_UNITS, return_sequences=True, name="gru")(z)

    x = layers.TimeDistributed(layers.Dense(32, activation="relu"), name="head_dense")(x)
    out = layers.TimeDistributed(layers.Dense(2, dtype="float32"), name="vel_output")(x)

    return models.Model(inputs=inputs, outputs=out, name="LNN_modal")


# ---------------- Input unpack for each run ----------------
def make_input_unpack(use_image, use_lidar, use_state, state_base_dim):
    def input_unpack(img_seq, lidar_seq, state_seq):
        xs = ()
        if use_image:
            xs += (img_seq,)
        if use_lidar:
            xs += (lidar_seq,)
        if use_state:
            xs += (state_seq,)
        else:
            goal_vec = state_seq[:, 0, state_base_dim:]  # (B, n_goals)
            xs += (goal_vec,)
        return xs[0] if len(xs) == 1 else xs
    return input_unpack


def eval_one_run(run_dir):
    run_info_path = os.path.join(run_dir, "run_info.json")
    if not os.path.exists(run_info_path):
        raise FileNotFoundError(f"Missing run_info.json in {run_dir}")

    run_info = json.load(open(run_info_path, "r"))
    exp = run_info["experiment"]
    hp  = run_info["hyperparams"]

    use_image = bool(exp["use_image"])
    use_lidar = bool(exp["use_lidar"])
    use_state = bool(exp["use_state"])

    # Important: these globals control episode filtering inside build_datasets
    global GOAL_FILTER, EPISODE_LIMIT, EPISODE_PICK, EPISODE_SEED
    GOAL_FILTER = str(exp.get("goal_filter", "ALL"))
    EPISODE_LIMIT = exp.get("episode_limit", None)
    EPISODE_PICK  = exp.get("episode_pick", "first") or "first"
    EPISODE_SEED  = exp.get("episode_seed", 123) or 123

    # Build datasets (same split logic as training, but with jitter off)
    _, val_ds, stats = build_datasets(
        csv_path=run_info["paths"].get("csv", CSV_PATH),
        image_dir=run_info["paths"].get("image_dir", IMAGE_DIR),
        seq_len=int(hp["SEQ_LEN"]),
        stride=int(hp["STRIDE"]),
        batch_size=int(hp["BATCH_SIZE"]),
        val_split=float(hp["VAL_SPLIT"]),
        seed=int(hp["SEED"]),
        jitter=False,
        goal_filter=GOAL_FILTER,
    )


    print(
    f"[SPLIT] {os.path.basename(run_dir)} "
    f"seq_total={stats['num_sequences_total']} "
    f"seq_train={stats['num_sequences_train']} "
    f"seq_val={stats['num_sequences_val']} "
    f"(SEQ_LEN={hp['SEQ_LEN']}, STRIDE={hp['STRIDE']}, VAL_SPLIT={hp['VAL_SPLIT']}, SEED={hp['SEED']})"
    )


    state_dim      = int(stats["state_dim"])
    state_base_dim = int(stats["state_base_dim"])
    n_goals        = int(stats["n_goals"])

    model = build_sequence_model(
        IMG_SHAPE=tuple(hp["IMG_SHAPE"]),
        LIDAR_DIM=int(hp["LIDAR_DIM"]),
        STATE_DIM=state_dim,
        STATE_BASE_DIM=state_base_dim,
        N_GOALS=n_goals,
        SEQ_LEN=int(hp["SEQ_LEN"]),
        CFC1_UNITS=int(hp["CFC1_UNITS"]),
        use_image=use_image,
        use_lidar=use_lidar,
        use_state=use_state,
        rnn_backend=str(exp.get("rnn_backend_requested", "CFC")),
    )

    model.compile(
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )

    weights_path = run_info["paths"].get("weights_path", None)
    if not weights_path or not os.path.exists(weights_path):
        # fallback: try to find weights file inside run dir
        candidates = glob.glob(os.path.join(run_dir, "*.weights.h5"))
        if len(candidates) == 0:
            raise FileNotFoundError(f"No weights found in {run_dir}")
        weights_path = candidates[0]

    model.load_weights(weights_path)

    input_unpack = make_input_unpack(use_image, use_lidar, use_state, state_base_dim)
    val_ds2 = val_ds.map(lambda x, y: (input_unpack(*x), y), num_parallel_calls=AUTOTUNE)

    loss, mae = model.evaluate(val_ds2, verbose=0)
    
    mae_lin, mae_ang = eval_per_output_mae(model, val_ds2)

    return float(mae), float(loss), float(mae_lin), float(mae_ang), stats, hp



def main():
    setup_gpu()
    tf.random.set_seed(42)
    np.random.seed(42)

    # Evaluate ONLY these run folders (names exactly as in /models)
    ALLOWED_RUNS = [
        "IMG0_LID0_STA1_CFC64",
        "IMG0_LID1_STA0_CFC64",
        "IMG0_LID1_STA1_CFC64",
        "IMG1_LID0_STA0_CFC64",
        "IMG1_LID0_STA1_CFC64",
        "IMG1_LID1_STA0_CFC64",
        "IMG1_LID1_STA1_CFC64",
    ]

    run_dirs = []
    missing = []
    for name in ALLOWED_RUNS:
        d = os.path.join(ROOT_MODEL_DIR, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "run_info.json")):
            run_dirs.append(d)
        else:
            missing.append(name)

    if missing:
        print("[WARN] Missing or invalid run folders:", missing)

    if not run_dirs:
        raise RuntimeError("No valid run folders found from ALLOWED_RUNS")

    rows = []
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        try:
            mae, loss, mae_lin, mae_ang, stats, hp = eval_one_run(run_dir)
            rows.append({
                "run_folder": run_name,
                "val_mae": mae,
                "val_loss": loss,
                "val_mae_linear": mae_lin,
                "val_mae_angular": mae_ang,
                "seq_val": stats["num_sequences_val"],
            })

            print(f"[OK] {run_name}: val_mae={mae:.6f}, val_loss={loss:.6f}")
        except Exception as e:
            rows.append({"run_folder": run_name, "val_mae": np.nan, "val_loss": np.nan, "error": str(e)})
            print(f"[FAIL] {run_name}: {e}")

    df = pd.DataFrame(rows)
    if "error" not in df.columns:
        df["error"] = ""

    df_sorted = df.sort_values(["val_mae"], na_position="last")
    out_csv = os.path.join(ROOT_MODEL_DIR, "val_mae_ranking.csv")
    df_sorted.to_csv(out_csv, index=False)
    print(f"[SAVE] {out_csv}")
    print(df_sorted.head(20))


if __name__ == "__main__":
    main()
