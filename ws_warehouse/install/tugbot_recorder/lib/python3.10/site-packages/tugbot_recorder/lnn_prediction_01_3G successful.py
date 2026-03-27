#!/usr/bin/env python3
"""
1 cfc 128 on hot gaosl satate and inject state after rnn 
ROS2 inference node for the LNN (CfC) controller trained with:

  - Images  : CNN encoder (3 conv layers + flatten + Dense(64))
  - LiDAR   : MLP (64->64)
  - State   : [dX, dY, sin(dYaw), cos(dYaw), one_hot(goal_id)]
  - RNN core: Single CfC (or GRU fallback) with:
        * Pre-RNN fusion of encoded [img, lidar, state]
        * Post-RNN injection of RAW state_seq before the Dense head

This node:
  - Loads run_info.json and weights from RUN_FOLDER
  - Rebuilds the **exact same architecture** as in training
  - Reconstructs the normalized state vector using the stored scaler
  - Buffers SEQ_LEN steps and publishes cmd_vel from the last timestep
 
 - smal mismatch in the lidar call back


  """

import os
import json
import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from collections import deque

from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

import tensorflow as tf
from tensorflow.keras import layers, models

# ===================== USER SETTINGS =====================
START_ID   = 'Home'   # context only
GOAL_ID    = 'B'      # <-- choose: 'Home' / 'A' / 'B' / 'C' etc.
SEQ_LEN    = 32
FPS        = 10.0     # target inference rate (Hz) once buffers are full

RUN_FOLDER     = "Goal_ALL_cfc64_st08_1hot_injPRE_tr_03"

ROOT_MODEL_DIR = "/home/newin/Projects/warehouse/models"
ROOT_LOG_DIR   = "/home/newin/Projects/warehouse/log_dir"
# =========================================================

# World/map frame coordinates from Ignition Gazebo
GOALS = {
    'Home': {'x': 0.0000,    'y': 0.0000,  'yaw_rad':  0.0000},
    'A':    {'x': -32.9899,  'y': 7.3711,  'yaw_rad': -3.0557},
    'B':    {'x': -8.6379,  'y': 27.1289, 'yaw_rad': -1.1666},
    'C':    {'x': -12.9876,  'y': 37.0778, 'yaw_rad':  1.4334},
}

# ------- Liquid / GRU core selection -------
_HAS_NCPS = False
try:
    from ncps.tf import CfC
    _HAS_NCPS = True
except Exception:
    _HAS_NCPS = False

AUTOTUNE = tf.data.AUTOTUNE

# ---------- Small helpers ----------
def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def relative_goal(robot_pose, goal_pose):
    """
    robot_pose=(x_r,y_r,yaw_r), goal_pose=(x_g,y_g,yaw_g) in WORLD.
    Returns dX,dY in ROBOT frame and sin/cos(dYaw).
    """
    x_r, y_r, th_r = robot_pose
    x_g, y_g, th_g = goal_pose
    dx = x_g - x_r
    dy = y_g - y_r
    c, s = math.cos(th_r), math.sin(th_r)
    dX =  c * dx + s * dy
    dY = -s * dx + c * dy
    dYaw = wrap_to_pi(th_g - th_r)
    return dX, dY, math.sin(dYaw), math.cos(dYaw)

def dist2d(p1, p2) -> float:
    """ Euclidean distance in WORLD frame (same as recorder.py). """
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def zscore(x, mean, std):
    return (x - mean) / std

# ---- QUANTIZE VELOCITIES ----
def quantize_linear(v):
    if v > 0.15:
        return 0.5
    elif v < -0.15:
        return -0.5
    else:
        return 0.0
    
def quantize_angular(v):
    if v > 0.08:
        return 0.5
    elif v < -0.08:
        return -0.5
    else:
        return 0.0


# ---------------- Model blocks (MUST MATCH TRAINING) ----------------
def build_cnn_encoder(input_shape=(128, 128, 1)):
    img_in = layers.Input(shape=input_shape, name="image_input")
    x = layers.Conv2D(16, 5, strides=2, padding="same", activation="relu")(img_in)  # 64x64x16
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)       # 32x32x32
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)       # 16x16x64
    x = layers.Flatten()(x)                                                         # 16*16*64=16384
    x = layers.Dense(64, activation="relu", name="cnn_feature")(x)                  # -> 64
    return models.Model(img_in, x, name="CNN_Encoder")

def build_lidar_mlp(lidar_dim=67):
    lidar_in = layers.Input(shape=(lidar_dim,), name="lidar_input")
    x = layers.Dense(64, activation="relu")(lidar_in)
    x = layers.Dense(64, activation="relu", name="lidar_feature")(x)                # -> 64
    return models.Model(lidar_in, x, name="LiDAR_MLP")

def build_goal_state_mlp(state_dim, out_dim=32):
    """
    MLP for the full state vector:
      [dX, dY, sin(dYaw), cos(dYaw), one_hot(goal_id)...]
    """
    state_in = layers.Input(shape=(state_dim,), name="state_input")
    x = layers.Dense(32, activation="relu")(state_in)
    x = layers.Dense(out_dim, activation="relu", name="state_feature")(x)
    return models.Model(state_in, x, name="GoalState_MLP")

def build_sequence_lnn_cfc(
    IMG_SHAPE=(128,128,1), LIDAR_DIM=67, STATE_DIM=8,
    STATE_BASE_DIM=5, CFC1_UNITS=64
):
    img_seq   = layers.Input(shape=(None,) + IMG_SHAPE, name="image_seq")
    lidar_seq = layers.Input(shape=(None, LIDAR_DIM),   name="lidar_seq")
    state_seq = layers.Input(shape=(None, STATE_DIM),   name="state_seq")

    # Split state
    cont_seq = layers.Lambda(lambda s: s[..., :STATE_BASE_DIM], name="state_cont")(state_seq)   # (B,T,5)
    goal_seq = layers.Lambda(lambda s: s[..., STATE_BASE_DIM:], name="goal_onehot")(state_seq)  # (B,T,3)

    cnn_enc   = build_cnn_encoder(IMG_SHAPE)   # -> 64
    lidar_enc = build_lidar_mlp(LIDAR_DIM)     # -> 64
    state_enc = build_goal_state_mlp(state_dim=STATE_BASE_DIM, out_dim=32)  # -> 32

    img_feat   = layers.TimeDistributed(cnn_enc,   name="TD_CNN")(img_seq)       # (B,T,64)
    lidar_feat = layers.TimeDistributed(lidar_enc, name="TD_LiDAR")(lidar_seq)   # (B,T,64)
    state_feat = layers.TimeDistributed(state_enc, name="TD_State")(cont_seq)    # (B,T,32)

    fused = layers.Concatenate(axis=-1, name="fuse")([img_feat, lidar_feat, state_feat])  # (B,T,160)

    # Compress features (no goal yet)
    z = layers.TimeDistributed(layers.Dense(128, activation="relu"), name="pre_rnn")(fused)  # (B,T,128)

    # Append goal one-hot right before CfC
    z = layers.Concatenate(axis=-1, name="pre_rnn_with_goal")([z, goal_seq])  # (B,T,128+n_goals)

    if _HAS_NCPS:
        x = CfC(units=CFC1_UNITS, return_sequences=True, name="cfc")(z)
    else:
        x = layers.GRU(units=CFC1_UNITS, return_sequences=True, name="gru")(z)

    x = layers.TimeDistributed(layers.Dense(32, activation="relu"), name="head_dense")(x)
    out = layers.TimeDistributed(layers.Dense(2, dtype="float32"), name="vel_output")(x)

    return models.Model([img_seq, lidar_seq, state_seq], out, name="CNN_LiDAR_State_1xCfC_noPostInject")




class InferenceNode(Node):
    def __init__(self):
        super().__init__('lnn_inference')

        # ---- Build paths from RUN_FOLDER (matches training layout) ----
        self.MODEL_DIR = os.path.join(ROOT_MODEL_DIR, RUN_FOLDER)
        self.LOG_DIR   = os.path.join(ROOT_LOG_DIR,   RUN_FOLDER)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR,   exist_ok=True)

        self.model_path    = os.path.join(self.MODEL_DIR, f"{RUN_FOLDER}.keras")
        self.run_info_path = os.path.join(self.MODEL_DIR, "run_info.json")

        self.get_logger().info(f"[MODEL] {self.model_path}")
        self.get_logger().info(f"[INFO ] {self.run_info_path}")

        # ---- Load scaler + hyperparams + goals from run_info.json ----
        with open(self.run_info_path, "r") as f:
            run_info = json.load(f)

        scaler   = run_info["scaler"]
        hparams  = run_info["hyperparams"]
        goals_md = run_info.get("goals", {})

        # LiDAR/state normalization (z-score)
        self.lidar_mean = np.array(scaler["lidar_mean"], dtype=np.float32)
        self.lidar_std  = np.array(scaler["lidar_std"],  dtype=np.float32)
        self.state_mean = np.array(scaler["state_mean"], dtype=np.float32)
        self.state_std  = np.array(scaler["state_std"],  dtype=np.float32)
        self.lidar_max_range = float(scaler.get("lidar_max_range", 10.0))

        # Hyperparams (must match training)
        self.IMG_SHAPE       = tuple(hparams["IMG_SHAPE"])
        self.LIDAR_DIM       = int(hparams["LIDAR_DIM"])
        self.STATE_BASE_DIM  = int(hparams.get("STATE_BASE_DIM"))  # base [dX,dY,sin_dYaw,cos_dYaw]
        self.STATE_DIM       = int(hparams["STATE_DIM"])              # = STATE_BASE_DIM + n_goals
        self.CFC1_UNITS      = int(hparams["CFC1_UNITS"])
        # CFC2_UNITS exists in run_info but is unused in this architecture

        self.lidar_dim = self.LIDAR_DIM
        self.state_dim = self.STATE_DIM
        self.max_goal_dist = float(scaler.get("max_goal_dist", 1.0))


        # Goal IDs ordering used for one-hot encoding during training
        self.goal_ids = goals_md.get("ids", [])
        if not self.goal_ids:
            raise ValueError("run_info.json has no 'goals.ids' list; cannot build one-hot goal state.")

        if GOAL_ID not in self.goal_ids:
            raise ValueError(f"GOAL_ID='{GOAL_ID}' not in training goal_ids {self.goal_ids}")

        # (Optional) respect mixed precision if you really want; not required for inference
        if hparams.get("MIXED_PRECISION", False):
            try:
                from tensorflow.keras import mixed_precision as mp
                mp.set_global_policy("mixed_float16")
                self.get_logger().info("[MP] Mixed precision enabled for inference.")
            except Exception as e:
                self.get_logger().warn(f"[MP] Could not enable mixed precision: {e}")

        # Rebuild the **same** architecture used during training
        self.model = build_sequence_lnn_cfc(
            IMG_SHAPE=self.IMG_SHAPE,
            LIDAR_DIM=self.LIDAR_DIM,
            STATE_DIM=self.STATE_DIM,
            STATE_BASE_DIM=self.STATE_BASE_DIM,
            CFC1_UNITS=self.CFC1_UNITS,
        )

        # Load the trained weights
        WEIGHTS_PATH = os.path.join(self.MODEL_DIR, f"{RUN_FOLDER}.weights.h5")
        self.get_logger().info(f"[MODEL] Loading weights from: {WEIGHTS_PATH}")
        self.model.load_weights(WEIGHTS_PATH)

        # ---- Buffers (rolling) ----
        self.seq_len   = SEQ_LEN
        self.buf_img   = deque(maxlen=self.seq_len)  # each (128,128) float32
        self.buf_lidar = deque(maxlen=self.seq_len)  # each (LIDAR_DIM,) float32
        self.buf_state = deque(maxlen=self.seq_len)  # each (STATE_DIM,) float32

        # ---- ROS setup ----
        self.bridge = CvBridge()

        self.create_subscription(
            Image,
            '/world/world_demo/model/tugbot/link/camera_front/sensor/color/image',
            self.image_callback, 10
        )
        self.create_subscription(
            Odometry,
            '/model/tugbot/odometry',
            self.odom_callback, 10
        )
        self.create_subscription(
            LaserScan,
            '/world/world_demo/model/tugbot/link/scan_front/sensor/scan_front/scan',
            self.lidar_callback, 10
        )

        # Publish predicted velocities to cmd_vel
        self.twist_pub = self.create_publisher(
            Twist,
            '/model/tugbot/cmd_vel',
            10
        )
        # Optional debug vector
        self.vec_pub = self.create_publisher(Float32MultiArray, '/lnn/prediction', 10)

        # Run inference periodically
        self.timer = self.create_timer(1.0 / FPS, self.inference_step)

        # ---- Goal selection (world pose for geometry) ----
        assert GOAL_ID in GOALS, "GOAL_ID must be one of: " + ", ".join(GOALS.keys())
        g = GOALS[GOAL_ID]
        self.goal_pose_world = (
            float(g['x']),
            float(g['y']),
            float(g['yaw_rad'])
        )
        self.get_logger().info(f"[GOAL] Using hardcoded world goal '{GOAL_ID}': {self.goal_pose_world}")
        self.get_logger().info(f"[GOAL] One-hot index of '{GOAL_ID}' is {self.goal_ids.index(GOAL_ID)} in {self.goal_ids}")

        # Robot pose cache (world)
        self.robot_pose_world = (0.0, 0.0, 0.0)

    # ---------- Callbacks ----------
    def image_callback(self, msg: Image):
        """Convert to grayscale 128x128 and scale to [0,1]."""
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            gray   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)

            img_f = resized.astype(np.float32) / 255.0
            img_f = img_f[..., None]   # 🔴 ADD THIS (128,128,1)

            self.buf_img.append(img_f)

        except Exception as e:
            self.get_logger().error(f"image_callback error: {e}")


    def lidar_callback(self, msg: LaserScan):
        try:
            ranges = np.array(msg.ranges, dtype=np.float32)
            n_raw  = ranges.shape[0]
            target = self.lidar_dim

            # resample
            if n_raw != target:
                idx = np.linspace(0, n_raw - 1, target, dtype=int)
                lidar = ranges[idx]
            else:
                lidar = ranges

            # clip → normalize to [0,1]
            lidar = np.nan_to_num(
                lidar,
                nan=self.lidar_max_range,
                posinf=self.lidar_max_range,
                neginf=0.0,
            )
            lidar = np.clip(lidar, 0.0, self.lidar_max_range)
            lidar = lidar / self.lidar_max_range

            # z-score (MATCH TRAINING)
            lidar_z = (lidar - self.lidar_mean) / self.lidar_std
            self.buf_lidar.append(lidar_z.astype(np.float32))

        except Exception as e:
            self.get_logger().error(f"lidar_callback error: {e}")

        
    def odom_callback(self, msg: Odometry):
        """Cache robot pose and push z-scored STATE_DIM vector."""
        try:
            x_r = float(msg.pose.pose.position.x)
            y_r = float(msg.pose.pose.position.y)
            yaw_r = quat_to_yaw(msg.pose.pose.orientation)
            self.robot_pose_world = (x_r, y_r, yaw_r)

            state_z = self._build_state_z(self.robot_pose_world, self.goal_pose_world)
            self.buf_state.append(state_z)
        except Exception as e:
            self.get_logger().error(f"odom_callback error: {e}")

    # ---------- State vector builder ----------
    def _build_state_z(self, robot_pose_world, goal_pose_world):

        # --- Relative goal in ROBOT frame ---
        dX, dY, sin_dYaw, cos_dYaw = relative_goal(
            robot_pose_world,
            goal_pose_world
        )

        # --- Distance in WORLD frame ---
        dist_to_goal = dist2d(
            (robot_pose_world[0], robot_pose_world[1]),
            (goal_pose_world[0], goal_pose_world[1])
        )

        base = np.array(
            [dX, dY, sin_dYaw, cos_dYaw, dist_to_goal],
            dtype=np.float32
        )

        if base.shape[0] != self.STATE_BASE_DIM:
            raise ValueError(
                f"STATE_BASE_DIM={self.STATE_BASE_DIM}, got {base.shape[0]}"
            )

        base[0] /= self.max_goal_dist   # dX
        base[1] /= self.max_goal_dist   # dY
        base[4] /= self.max_goal_dist   # dist_to_goal

        base_z = (base - self.state_mean) / self.state_std

        # --- raw one-hot goal ---
        one_hot = np.zeros(len(self.goal_ids), dtype=np.float32)
        one_hot[self.goal_ids.index(GOAL_ID)] = 1.0

        state = np.concatenate([base_z, one_hot], axis=0).astype(np.float32)

        if state.shape[0] != self.STATE_DIM:
            raise ValueError(
                f"STATE_DIM={self.STATE_DIM}, got {state.shape[0]}"
            )

        # Log once for sanity
        if not hasattr(self, "_state_logged"):
            self.get_logger().info(
                f"STATE base(z): {state[:5]} | onehot: {state[5:]}"
            )
            self._state_logged = True

        return state



    # ---------- Inference ----------
    def inference_step(self):
        """Run the model when all buffers have SEQ_LEN frames; publish Twist."""
        if (len(self.buf_img)   < self.seq_len or
            len(self.buf_lidar) < self.seq_len or
            len(self.buf_state) < self.seq_len):
            return

        try:
            # Stack and add batch dimension: (1, T, ...)
            img_seq   = np.stack(self.buf_img,   axis=0)[None, ...]   # (1, T, H, W)
            lidar_seq = np.stack(self.buf_lidar, axis=0)[None, ...]   # (1, T, LIDAR_DIM)
            state_seq = np.stack(self.buf_state, axis=0)[None, ...]   # (1, T, STATE_DIM)

 
            # Predict sequence of velocities: (1,T,2)
            pred = self.model.predict([img_seq, lidar_seq, state_seq], verbose=0)
            v_lin = float(pred[0, -1, 0])
            v_ang = float(pred[0, -1, 1])

            # ---- LOG PREDICTED VELOCITIES ----
            if not hasattr(self, "_vel_log_ctr"):
                self._vel_log_ctr = 0
            self._vel_log_ctr += 1

            if self._vel_log_ctr % int(FPS) == 0:
                self.get_logger().info(
                    f"PRED v_lin = {v_lin:.4f} m/s | v_ang = {v_ang:.4f} rad/s"
                 )
            if not hasattr(self, "_img_shape_logged"):
                self.get_logger().info(f"img_seq shape = {img_seq.shape}")
                self._img_shape_logged = True


            v_lin_q = quantize_linear(v_lin)
            v_ang_q = quantize_angular(v_ang)



            # Publish Twist
            tw = Twist()
            tw.linear.x  = v_lin_q
            tw.angular.z = v_ang_q
            self.twist_pub.publish(tw)

            # Optional: also publish raw vector
            msg = Float32MultiArray()
            msg.data = [v_lin_q, v_ang_q]
            self.vec_pub.publish(msg)


        except Exception as e:
            self.get_logger().error(f"inference_step error: {e}")

    # ---------- Cleanup ----------
    def destroy_node(self):
        super().destroy_node()


# ---------- Main --------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
