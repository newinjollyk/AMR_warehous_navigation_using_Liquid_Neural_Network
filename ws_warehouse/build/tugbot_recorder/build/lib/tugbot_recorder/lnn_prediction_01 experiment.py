#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import cv2
import time
import csv
from datetime import datetime
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

# ====== Try CfC ======
_HAS_NCPS = False
try:
    from ncps.tf import CfC
    _HAS_NCPS = True
except Exception:
    _HAS_NCPS = False


# ===================== USER SETTINGS =====================
RUN_FOLDER = "IMG1_LID1_STA1_CFC64"   # <-- select folder only
ROOT_MODEL_DIR = "/home/newin/Projects/warehouse/models"
FPS = 10.0

# ROS topics (keep your current ones)
TOPIC_IMAGE = "/world/world_demo/model/tugbot/link/camera_front/sensor/color/image"
TOPIC_ODOM  = "/model/tugbot/odometry"
TOPIC_LIDAR = "/world/world_demo/model/tugbot/link/scan_front/sensor/scan_front/scan"

# Goal in world frame (still needed for building state/goal features)
GOAL_ID = "C"
GOALS = {
    'Home': {'x': 0.0000,    'y': 0.0000,  'yaw_rad':  0.0000},
    'A':    {'x': -32.9899,  'y': 7.3711,  'yaw_rad': -3.0557},
    'B':    {'x': -8.6379,   'y': 27.1289, 'yaw_rad': -1.1666},
    'C':    {'x': -12.9876,  'y': 37.0778, 'yaw_rad':  1.4334},
}
# =========================================================


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def relative_goal(robot_pose, goal_pose):
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
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


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


# ---------------- Model blocks (same as training code) ----------------
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


# ---------------- ROS Node ----------------
class InferenceNode(Node):
    def __init__(self):
        super().__init__("model_inference_auto")

        self.model_dir = os.path.join(ROOT_MODEL_DIR, RUN_FOLDER)
        self.run_info_path = os.path.join(self.model_dir, "run_info.json")
        self.model_path = os.path.join(self.model_dir, f"{RUN_FOLDER}.keras")
        self.weights_path = os.path.join(self.model_dir, f"{RUN_FOLDER}.weights.h5")

        if not os.path.exists(self.run_info_path):
            raise FileNotFoundError(f"Missing {self.run_info_path}")

        run_info = json.load(open(self.run_info_path, "r"))
        exp = run_info["experiment"]
        hp = run_info["hyperparams"]
        scaler = run_info["scaler"]
        goals_md = run_info.get("goals", {})

        # flags
        self.use_image = bool(exp["use_image"])
        self.use_lidar = bool(exp["use_lidar"])
        self.use_state = bool(exp["use_state"])
        self.rnn_backend = str(exp.get("rnn_backend_requested", "CFC"))

        # hyperparams
        self.seq_len = int(hp["SEQ_LEN"])
        self.IMG_SHAPE = tuple(hp["IMG_SHAPE"])
        self.LIDAR_DIM = int(hp["LIDAR_DIM"])
        self.STATE_BASE_DIM = int(hp["STATE_BASE_DIM"])
        self.CFC1_UNITS = int(hp["CFC1_UNITS"])

        # goal ids
        self.goal_ids = goals_md.get("ids", [])
        if not self.goal_ids:
            raise ValueError("run_info.json missing goals.ids")
        if GOAL_ID not in self.goal_ids:
            raise ValueError(f"GOAL_ID='{GOAL_ID}' not in goal_ids={self.goal_ids}")

        self.N_GOALS = len(self.goal_ids)
        self.STATE_DIM = self.STATE_BASE_DIM + self.N_GOALS

        # scalers
        self.lidar_mean = np.asarray(scaler["lidar_mean"], dtype=np.float32)
        self.lidar_std  = np.asarray(scaler["lidar_std"], dtype=np.float32)
        self.state_mean = np.asarray(scaler["state_mean"], dtype=np.float32)
        self.state_std  = np.asarray(scaler["state_std"], dtype=np.float32)
        self.lidar_max_range = float(scaler.get("lidar_max_range", 10.0))
        self.max_goal_dist = float(scaler.get("max_goal_dist", 50.0))

        # goal pose + pose cache
        g = GOALS[GOAL_ID]
        self.goal_pose_world = (float(g["x"]), float(g["y"]), float(g["yaw_rad"]))
        self.robot_pose_world = (0.0, 0.0, 0.0)
        self.dist_to_goal = float("nan")

        # buffers
        self.buf_img = deque(maxlen=self.seq_len)
        self.buf_lidar = deque(maxlen=self.seq_len)
        self.buf_state = deque(maxlen=self.seq_len)

        # CSV log setup (in MODEL folder)
        self.run_id = datetime.now().strftime("%m%d_%H%M")  # MMDD_HHMM
        self.log_path = os.path.join(self.model_dir, f"{RUN_FOLDER}__{self.run_id}.csv")
        self.log_f = open(self.log_path, "w", newline="")
        self.log_w = csv.writer(self.log_f)
        self.log_w.writerow([
            "t_unix",
            "t_ros_ns",
            "run_folder",
            "run_id",
            "goal_id",
            "v_pred_lin",
            "w_pred_ang",
            "v_cmd_lin",
            "w_cmd_ang",
            "x", "y", "yaw",
            "dist_to_goal",
        ])
        self.log_f.flush()
        self.get_logger().info(f"[LOG] {self.log_path}")

        # ROS
        self.bridge = CvBridge()

        if self.use_image:
            self.create_subscription(Image, TOPIC_IMAGE, self.image_callback, 10)
            self.get_logger().info("[SUB] image enabled")
        else:
            self.get_logger().info("[SUB] image disabled")

        self.create_subscription(Odometry, TOPIC_ODOM, self.odom_callback, 10)

        if self.use_lidar:
            self.create_subscription(LaserScan, TOPIC_LIDAR, self.lidar_callback, 10)
            self.get_logger().info("[SUB] lidar enabled")
        else:
            self.get_logger().info("[SUB] lidar disabled")

        self.twist_pub = self.create_publisher(Twist, "/model/tugbot/cmd_vel", 10)
        self.vec_pub = self.create_publisher(Float32MultiArray, "/model/prediction", 10)

        # model
        self.model = self._load_model(run_info)

        # periodic inference
        self.timer = self.create_timer(1.0 / FPS, self.inference_step)

        self.get_logger().info(
            f"[RUN] {RUN_FOLDER} | use_image={self.use_image} use_lidar={self.use_lidar} use_state={self.use_state} "
            f"| seq_len={self.seq_len} | run_id={self.run_id}"
        )

        self._flush_ctr = 0

    def _load_model(self, run_info):
        if os.path.exists(self.model_path):
            try:
                self.get_logger().info(f"[MODEL] Trying load_model: {self.model_path}")
                m = tf.keras.models.load_model(self.model_path, compile=False)  # inference load [web:1014]
                self.get_logger().info("[MODEL] load_model OK")
                return m
            except Exception as e:
                self.get_logger().warn(f"[MODEL] load_model failed, fallback to weights. Reason: {e}")

        self.get_logger().info("[MODEL] Rebuilding architecture + load_weights fallback")

        built = build_sequence_model(
            IMG_SHAPE=self.IMG_SHAPE,
            LIDAR_DIM=self.LIDAR_DIM,
            STATE_DIM=self.STATE_DIM,
            STATE_BASE_DIM=self.STATE_BASE_DIM,
            N_GOALS=self.N_GOALS,
            SEQ_LEN=self.seq_len,
            CFC1_UNITS=self.CFC1_UNITS,
            use_image=self.use_image,
            use_lidar=self.use_lidar,
            use_state=self.use_state,
            rnn_backend=self.rnn_backend,
        )

        model = built[0] if isinstance(built, tuple) else built

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Missing weights: {self.weights_path}")

        self.get_logger().info(f"[MODEL] load_weights: {self.weights_path}")
        model.load_weights(self.weights_path)  # architecture must match weights [web:1021]

        return model

    # ---------- Callbacks ----------
    def image_callback(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            #resized = cv2.resize(gray, (self.IMG_SHAPE[1], self.IMG_SHAPE[0]), interpolation=cv2.INTER_AREA)
            resized = cv2.resize(gray, (128,128),interpolation=cv2.INTER_LINEAR )  # 🔧 instead of INTER_AREA
            img_f = resized.astype(np.float32) / 255.0
            img_f = img_f[..., None]
            self.buf_img.append(img_f)
        except Exception as e:
            self.get_logger().error(f"image_callback error: {e}")

    def lidar_callback(self, msg: LaserScan):
        try:
            ranges = np.array(msg.ranges, dtype=np.float32)
            lidar = ranges[::10]          # EXACTLY like recorder
            lidar = lidar[:self.LIDAR_DIM]  # safety if it becomes >67
            if lidar.shape[0] < self.LIDAR_DIM:
                pad = np.full(self.LIDAR_DIM - lidar.shape[0], self.lidar_max_range, dtype=np.float32)
                lidar = np.concatenate([lidar, pad])

            lidar = np.nan_to_num(lidar, nan=self.lidar_max_range, posinf=self.lidar_max_range, neginf=0.0)
            lidar = np.clip(lidar, 0.0, self.lidar_max_range)
            lidar = lidar / self.lidar_max_range
            lidar_z = (lidar - self.lidar_mean) / self.lidar_std
            self.buf_lidar.append(lidar_z.astype(np.float32))
        except Exception as e:
            self.get_logger().error(f"lidar_callback error: {e}")

    def odom_callback(self, msg: Odometry):
        try:
            x_r = float(msg.pose.pose.position.x)
            y_r = float(msg.pose.pose.position.y)
            yaw_r = quat_to_yaw(msg.pose.pose.orientation)
            self.robot_pose_world = (x_r, y_r, yaw_r)

            state_z = self._build_state_z(self.robot_pose_world, self.goal_pose_world)
            self.buf_state.append(state_z)
        except Exception as e:
            self.get_logger().error(f"odom_callback error: {e}")

    def _build_state_z(self, robot_pose_world, goal_pose_world):
        dX, dY, sin_dYaw, cos_dYaw = relative_goal(robot_pose_world, goal_pose_world)
        dist_to_goal = dist2d(
            (robot_pose_world[0], robot_pose_world[1]),
            (goal_pose_world[0], goal_pose_world[1])
        )
        self.dist_to_goal = float(dist_to_goal)

        base = np.array([dX, dY, sin_dYaw, cos_dYaw, dist_to_goal], dtype=np.float32)
        base[0] /= self.max_goal_dist
        base[1] /= self.max_goal_dist
        base[4] /= self.max_goal_dist

        base_z = (base - self.state_mean) / self.state_std

        one_hot = np.zeros(self.N_GOALS, dtype=np.float32)
        one_hot[self.goal_ids.index(GOAL_ID)] = 1.0

        state = np.concatenate([base_z, one_hot], axis=0).astype(np.float32)
        return state

    def _input_unpack_live(self, img_seq, lidar_seq, state_seq):
        xs = ()

        if self.use_image:
            xs += (img_seq,)
        if self.use_lidar:
            xs += (lidar_seq,)

        if self.use_state:
            xs += (state_seq,)
        else:
            if state_seq is None:
                raise RuntimeError("Goal-only mode requires state_seq to extract goal_vec.")
            goal_vec = state_seq[:, 0, self.STATE_BASE_DIM:]  # (B, n_goals)
            xs += (goal_vec,)

        return xs[0] if len(xs) == 1 else xs


    # ---------- Inference ----------
    def inference_step(self):
        if self.use_image and len(self.buf_img) < self.seq_len:
            return
        if self.use_lidar and len(self.buf_lidar) < self.seq_len:
            return
        if len(self.buf_state) < self.seq_len:
            return

        try:
            img_seq = np.stack(self.buf_img, axis=0)[None, ...] if self.use_image else None
            lidar_seq = np.stack(self.buf_lidar, axis=0)[None, ...] if self.use_lidar else None
            state_seq = np.stack(self.buf_state, axis=0)[None, ...]

            x_in = self._input_unpack_live(img_seq, lidar_seq, state_seq)

            pred = self.model(x_in, training=False).numpy()  # (1,T,2)
            v_pred = float(pred[0, -1, 0])
            w_pred = float(pred[0, -1, 1])

            v_cmd = quantize_linear(v_pred)
            w_cmd = quantize_angular(w_pred)
            # ---- LOG PREDICTED VELOCITIES (once per ~1 sec) ----
            if not hasattr(self, "_vel_log_ctr"):
                self._vel_log_ctr = 0
            self._vel_log_ctr += 1

            # roughly once per second (FPS messages per sec)
            if self._vel_log_ctr % max(1, int(FPS)) == 0:
                self.get_logger().info(
                    f"PRED v_lin={v_pred:.4f} | Q {v_cmd:.2f} || "
                    f"PRED w_ang={w_pred:.4f} | Q {w_cmd:.2f} || "
                    f"dist={self.dist_to_goal:.2f} m"
                )


            # publish
            tw = Twist()
            #tw.linear.x = v_cmd
            #tw.angular.z = w_cmd
            # TEMPORARY TEST
            tw.linear.x  = float(np.clip(v_pred, -0.5, 0.5))
            tw.angular.z = float(np.clip(w_pred, -0.5, 0.5)) 

            self.twist_pub.publish(tw)

            msg = Float32MultiArray()
            msg.data = [v_pred, w_pred, v_cmd, w_cmd]
            self.vec_pub.publish(msg)

            # log row (time-based)
            t_unix = time.time()
            t_ros_ns = int(self.get_clock().now().nanoseconds)  # ROS time [web:1171]
            x, y, yaw = self.robot_pose_world

            self.log_w.writerow([
                t_unix,
                t_ros_ns,
                RUN_FOLDER,
                self.run_id,
                GOAL_ID,
                v_pred,
                w_pred,
                v_cmd,
                w_cmd,
                x, y, yaw,
                self.dist_to_goal,
            ])

            # flush occasionally (reduce overhead)
            self._flush_ctr += 1
            if self._flush_ctr % 10 == 0:
                self.log_f.flush()

        except Exception as e:
            self.get_logger().error(f"inference_step error: {e}")

    def destroy_node(self):
        try:
            if hasattr(self, "log_f") and self.log_f is not None:
                self.log_f.flush()
                self.log_f.close()
        except Exception:
            pass
        super().destroy_node()


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


if __name__ == "__main__":
    main()
