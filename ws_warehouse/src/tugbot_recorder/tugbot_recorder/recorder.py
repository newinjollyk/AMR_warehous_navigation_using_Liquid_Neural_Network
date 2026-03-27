#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import os
import csv
from datetime import datetime
import numpy as np
import math

# ===================== USER SETTINGS =====================
START_ID   = 'Home'
GOAL_ID    = 'C'
EPISODE_ID = 21

GOALS = {
    'Home': {'x': 0.0000,    'y': 0.0000,  'yaw_rad':  0.0000},
    'A':    {'x': -32.9899,  'y': 7.3711,  'yaw_rad': -3.0557},
    'B':    {'x': -8.6379,   'y': 27.1289, 'yaw_rad': -1.1666},
    'C':    {'x': -12.9876,  'y': 37.0778, 'yaw_rad':  1.4334},
}

# ========================================================

def wrap_to_pi(a):
    return (a + math.pi) % (2*math.pi) - math.pi

def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def relative_goal(robot_pose, goal_pose):
    x_r, y_r, th_r = robot_pose
    x_g, y_g, th_g = goal_pose
    dx = x_g - x_r
    dy = y_g - y_r
    c, s = math.cos(th_r), math.sin(th_r)
    dX =  c*dx + s*dy
    dY = -s*dx + c*dy
    dYaw = wrap_to_pi(th_g - th_r)
    return dX, dY, math.sin(dYaw), math.cos(dYaw)

def dist2d(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

class TugbotRecorder(Node):
    def __init__(self):
        super().__init__('tugbot_recorder')

        assert START_ID in GOALS and GOAL_ID in GOALS

        self.start_id   = START_ID
        self.goal_id    = GOAL_ID
        self.episode_id = int(EPISODE_ID)

        # Runtime caches
        self.current_image = None
        self.odom_linear_vel = 0.0
        self.odom_angular_vel = 0.0
        self.cmd_linear_vel = 0.0
        self.cmd_angular_vel = 0.0
        self.current_lidar = []

        self.x_r = 0.0
        self.y_r = 0.0
        self.yaw_r = 0.0

        # File setup
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = f"dataset/{run_stamp}_{self.start_id}2{self.goal_id}_ep{self.episode_id}"
        img_folder = os.path.join(folder, "images")
        os.makedirs(img_folder, exist_ok=True)
        self.image_dir = img_folder

        csv_name = f"data_{self.start_id}2{self.goal_id}_ep{self.episode_id}.csv"
        self.csv_path = os.path.join(folder, csv_name)
        os.makedirs(folder, exist_ok=True)
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # CSV header (cleaned)
        self.csv_writer.writerow([
            'episode_id',
            'start_id','goal_id',
            'image_file',
            'x_r','y_r','yaw_r',
            'x_g','y_g','yaw_g',
            'dX','dY','sin_dYaw','cos_dYaw',
            'dist_to_goal','dist_home_to_goal',
            'odom_linear_vel','odom_angular_vel',
            'cmd_linear_vel','cmd_angular_vel',
            'lidar_points'
        ])

        # ROS setup
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
        self.create_subscription(
            Twist,
            '/model/tugbot/cmd_vel',
            self.cmd_callback, 10
        )

        self.timer = self.create_timer(0.1, self.save_data)

        self.home_xy = (GOALS['Home']['x'], GOALS['Home']['y'])
        gx, gy = GOALS[self.goal_id]['x'], GOALS[self.goal_id]['y']
        self.dist_home_to_goal_static = dist2d(self.home_xy, (gx, gy))

        self.get_logger().info(f"Logging to: {self.csv_path}")

    # --- Callbacks ---
    def image_callback(self, msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")

    def odom_callback(self, msg):
        self.odom_linear_vel  = float(msg.twist.twist.linear.x)
        self.odom_angular_vel = float(msg.twist.twist.angular.z)

        self.x_r = float(msg.pose.pose.position.x)
        self.y_r = float(msg.pose.pose.position.y)
        self.yaw_r = quat_to_yaw(msg.pose.pose.orientation)

    def cmd_callback(self, msg):
        self.cmd_linear_vel  = float(msg.linear.x)
        self.cmd_angular_vel = float(msg.angular.z)

    def lidar_callback(self, msg):
        self.current_lidar = [round(r, 3) for r in msg.ranges[::10]]

    # --- Save ---
    def save_data(self):
        if self.current_image is None:
            return

        g = GOALS[self.goal_id]
        x_g, y_g = float(g['x']), float(g['y'])
        yaw_g = float(g['yaw_rad'])

        dX, dY, sin_dYaw, cos_dYaw = relative_goal(
            (self.x_r, self.y_r, self.yaw_r),
            (x_g, y_g, yaw_g)
        )

        dist_to_goal = dist2d((self.x_r, self.y_r), (x_g, y_g))
        dist_home_to_goal = self.dist_home_to_goal_static

        img_name = f"{datetime.now().strftime('%H%M%S_%f')}.jpg"
        cv2.imwrite(os.path.join(self.image_dir, img_name), self.current_image)

        self.csv_writer.writerow([
            self.episode_id,
            self.start_id, self.goal_id,
            img_name,
            f"{self.x_r:.6f}", f"{self.y_r:.6f}", f"{self.yaw_r:.6f}",
            f"{x_g:.6f}", f"{y_g:.6f}", f"{yaw_g:.6f}",
            f"{dX:.6f}", f"{dY:.6f}", f"{sin_dYaw:.6f}", f"{cos_dYaw:.6f}",
            f"{dist_to_goal:.6f}", f"{dist_home_to_goal:.6f}",
            f"{self.odom_linear_vel:.6f}", f"{self.odom_angular_vel:.6f}",
            f"{self.cmd_linear_vel:.6f}", f"{self.cmd_angular_vel:.6f}",
            self.current_lidar
        ])

        self.get_logger().info(
            f"Saved {img_name} | ep={self.episode_id} {self.start_id}->{self.goal_id}"
        )

    def destroy_node(self):
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = TugbotRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()