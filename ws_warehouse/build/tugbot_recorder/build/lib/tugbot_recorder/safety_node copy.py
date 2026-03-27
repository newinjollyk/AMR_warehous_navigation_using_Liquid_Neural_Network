#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import csv
from enum import Enum
from datetime import datetime

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from visualization_msgs.msg import Marker
from rviz_2d_overlay_msgs.msg import OverlayText
from std_msgs.msg import ColorRGBA


# ========================= SAFETY CONSTANTS (edit here) =========================
TOPIC_CMD_IN  = "/pred_vel"
TOPIC_CMD_OUT = "/model/tugbot/cmd_vel"
TOPIC_SCAN    = "/world/world_demo/model/tugbot/link/scan_front/sensor/scan_front/scan"

CONTROL_HZ = 30.0
GOAL_RADIUS = 1.9

T_STARTUP_ALLOW = 10.0
T_WAIT_AFTER_STOP = 1.0

# Policy limit (what prediction code outputs)
V_MAX_POLICY = 0.5
W_MAX_POLICY = 0.5

# Safety node allowed max (can exceed policy if you want)
V_MAX_SAFETY = 0.7
W_MAX_SAFETY = 0.5

V_SLOW_SCALE = 0.35

W_CORRIDOR = 0.0
V_CORRIDOR_SCALE = 0.6

D_STOP = 0.30

# ---- NEW: separate warn thresholds for shoulders + (optional) front ----
D_WARN_F  = 1.0
D_WARN_FL = 1.0
D_WARN_FR = 1.0

# Sides: keep a distance warn if you like (optional), but TTC is primary
D_WARN_SIDE = 0.85

D_WALL = 0.35
HYST = 0.05

# Sector angles (same as your code)
DEG_FRONT = 30.0
DEG_FR    = 60.0
DEG_FL    = 60.0

# ---- NEW TTC + EVADE ----

TTC_SIDE_THRESH = 1.5      # seconds (start here with 10 Hz scan)
TTC_EPS_DDOT    = 0.20     # m/s (ignore tiny approaching rates = noise)
D_SIDE_TTC_MAX  = 3.0      # only compute TTC when side object is within this range

T_EVADE = 0.3              # seconds
D_CLEAR_FRONT = 1.5        # if front is very clear -> speed up to clear intersection
V_EVADE_BACK = 0.30        # m/s (reverse)
V_EVADE_FWD  = 0.60        # m/s (burst forward, limited by V_MAX_SAFETY)
W_EVADE = 0.0
# ==============================================================================


def clamp(x, lo, hi):
    return max(lo, min(x, hi))


class State(Enum):
    STARTUP_ALLOW_POLICY = 0
    NAVIGATE = 1
    SLOW = 2
    CORRIDOR_DAMPING = 3
    EVADE_SIDE = 4          # NEW
    STOP_EMERGENCY = 5
    WAIT = 6


class SafetyFSMNode(Node):
    def __init__(self):
        super().__init__("safety_fsm_node")

        # FSM
        self.state = State.STARTUP_ALLOW_POLICY
        self.state_enter_time_s = self._now_s()
        self.prev_state = self.state
        self.last_reason = ""

        # Command / scan
        self.last_cmd_in = None
        self.last_scan = None

        # Cached values for 1 Hz status prints (filled by control_loop)
        self.last_dR = 999.0
        self.last_dFR = 999.0
        self.last_dF = 999.0
        self.last_dFL = 999.0
        self.last_dL = 999.0
        self.last_ttcL = float("inf")
        self.last_ttcR = float("inf")
        self.last_ttc_side = float("inf")

        # NEW: TTC memory
        self.prev_dL = None
        self.prev_dR = None
        self.prev_side_t = None

        # Startup sync on first cmd
        self.seen_first_cmd = False
        self.startup_deadline_s = None

        # Goal / odom
        self.goal_pose = None
        self.odom_pose = None
        self.goal_reached = False
        self._warned_frame_mismatch = False

        # ROS timer for status prints in terminal
        self.timer_status = self.create_timer(1.0, self.status_loop)

        # Mission status tracking (for CSV + overlay)
        self.mission_status = "RUNNING"
        self.prev_mission_status = "RUNNING"

        # Visualization params
        self.viz_frame = "tugbot/scan_front/scan_front"
        self.radar_max_d = 5.0

        # ROS pubs/subs
        self.pub_cmd = self.create_publisher(Twist, TOPIC_CMD_OUT, 10)
        self.pub_radar = self.create_publisher(Marker, "/safety/radar", 10)
        self.pub_overlay_safety  = self.create_publisher(OverlayText, "/ui/safety_overlay", 10)
        self.pub_overlay_mission = self.create_publisher(OverlayText, "/ui/mission_overlay", 10)
        self.pub_radar_text = self.create_publisher(Marker, "/safety/radar_text", 10)

        self.create_subscription(Twist, TOPIC_CMD_IN, self.cmd_cb, 10)
        self.create_subscription(LaserScan, TOPIC_SCAN, self.scan_cb, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self.goal_cb, 10)
        self.create_subscription(Odometry, "/model/tugbot/odometry", self.odom_cb, 10)

        self.timer = self.create_timer(1.0 / CONTROL_HZ, self.control_loop)

        # CSV log
        run_id = datetime.now().strftime("%m%d_%H%M%S")
        self.csv_path = f"/home/newin/Projects/warehouse/log_safety_node/safety_log_{run_id}.csv"
        self.csv_f = open(self.csv_path, "w", newline="", buffering=1)
        self.csv_w = csv.writer(self.csv_f)
        self.csv_w.writerow([
            "t_ros_ns",
            "mission_status",
            "state_prev", "state_now", "event", "reason",
            "dR","dFR","dF","dFL","dL",
            "ttcR","ttcL","ttc_side",
            "v_in","w_in","v_out","w_out"
        ])

        self.get_logger().info(f"[SAFETY_LOG] {self.csv_path}")
        self.get_logger().info(
            f"SafetyFSM started. cmd_in={TOPIC_CMD_IN} cmd_out={TOPIC_CMD_OUT} scan={TOPIC_SCAN}"
        )

    # ---------- helpers ----------
    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _time_in_state(self) -> float:
        return self._now_s() - self.state_enter_time_s

    def _set_state(self, s: State, reason: str = ""):
        if s != self.state:
            old = self.state
            self.state = s
            self.state_enter_time_s = self._now_s()
            self.last_reason = reason
            self.get_logger().info(f"FSM {old.name} -> {s.name} | reason: {reason}")

    # ---------- ROS callbacks ----------
    def scan_cb(self, scan: LaserScan):
        self.last_scan = scan

    def cmd_cb(self, cmd: Twist):
        self.last_cmd_in = cmd
        if not self.seen_first_cmd:
            self.seen_first_cmd = True
            self.state = State.STARTUP_ALLOW_POLICY
            self.state_enter_time_s = self._now_s()
            self.startup_deadline_s = self.state_enter_time_s + T_STARTUP_ALLOW
            self.get_logger().info(
                f"Startup bypass armed for {T_STARTUP_ALLOW:.1f}s from first {TOPIC_CMD_IN}"
            )

    def goal_cb(self, msg: PoseStamped):
        self.goal_pose = msg
        self.goal_reached = False
        self.mission_status = "RUNNING"
        self.prev_mission_status = "RUNNING"

    def odom_cb(self, msg: Odometry):
        self.odom_pose = msg

    def status_loop(self):
        cmd = self.last_cmd_in if self.last_cmd_in is not None else Twist()

        if self.last_scan is None:
            self.get_logger().info(
                f"[SAFETY] mission={self.mission_status} safety={self.state.name} | no scan yet | "
                f"cmd_in v={cmd.linear.x:.2f} w={cmd.angular.z:.2f}"
            )
            return

        ttc_disp = self.last_ttc_side if self.last_ttc_side != float("inf") else 999.0
        self.get_logger().info(
            f"[SAFETY] mission={self.mission_status} safety={self.state.name} | "
            f"dF={self.last_dF:.2f} dFL={self.last_dFL:.2f} dFR={self.last_dFR:.2f} "
            f"dL={self.last_dL:.2f} dR={self.last_dR:.2f} | "
            f"ttc_side={ttc_disp:.2f}s | "
            f"cmd_in v={cmd.linear.x:.2f} w={cmd.angular.z:.2f}"
        )

    # ---------- Laser processing ----------
    def _min_dist_in_angle_range(self, scan: LaserScan, a_lo: float, a_hi: float) -> float:
        lo = max(a_lo, scan.angle_min)
        hi = min(a_hi, scan.angle_max)
        if hi <= lo:
            return float("inf")

        rmin = float("inf")
        a_min = scan.angle_min
        inc = scan.angle_increment

        for i, r in enumerate(scan.ranges):
            if not (r > 0.0) or math.isinf(r) or math.isnan(r):
                continue
            if r < scan.range_min or r > scan.range_max:
                continue
            ang = a_min + i * inc
            if lo <= ang <= hi:
                rmin = min(rmin, r)

        return rmin

    def _compute_sector_mins(self, scan: LaserScan):
        F_lo = math.radians(-DEG_FRONT)
        F_hi = math.radians(+DEG_FRONT)

        FR_lo = math.radians(-DEG_FR)
        FR_hi = math.radians(-DEG_FRONT)

        FL_lo = math.radians(+DEG_FRONT)
        FL_hi = math.radians(+DEG_FL)

        R_lo = scan.angle_min
        R_hi = math.radians(-DEG_FR)

        L_lo = math.radians(+DEG_FL)
        L_hi = scan.angle_max

        dF  = self._min_dist_in_angle_range(scan, F_lo,  F_hi)
        dFR = self._min_dist_in_angle_range(scan, FR_lo, FR_hi)
        dFL = self._min_dist_in_angle_range(scan, FL_lo, FL_hi)
        dR  = self._min_dist_in_angle_range(scan, R_lo,  R_hi)
        dL  = self._min_dist_in_angle_range(scan, L_lo,  L_hi)

        def fin(x):
            return x if x != float("inf") else float(scan.range_max)

        return fin(dR), fin(dFR), fin(dF), fin(dFL), fin(dL)

    # ---------- TTC from side range-rate ----------
    def _compute_side_ttc(self, dL: float, dR: float):
        now = self._now_s()
        dt = (now - self.prev_side_t) if (self.prev_side_t is not None) else None

        def ttc_one(d_prev, d_now):
            if dt is None or d_prev is None or dt <= 1e-3:
                return float("inf")
            if d_now > D_SIDE_TTC_MAX:
                return float("inf")

            ddot = (d_now - d_prev) / dt  # m/s (negative if approaching)
            if ddot < -TTC_EPS_DDOT:
                return d_now / (-ddot)
            return float("inf")

        ttcL = ttc_one(self.prev_dL, dL)
        ttcR = ttc_one(self.prev_dR, dR)
        ttc_side = min(ttcL, ttcR)

        self.prev_dL = dL
        self.prev_dR = dR
        self.prev_side_t = now

        return ttcL, ttcR, ttc_side

    # ---------- RViz radar + overlays ----------
    def _c(self, r, g, b, a=1.0):
        c = ColorRGBA()
        c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
        return c

    def _overlay(self, text, fg, bg, x, y, w=380, h=55):
        o = OverlayText()
        o.action = OverlayText.ADD
        o.width = int(w)
        o.height = int(h)
        o.horizontal_alignment = OverlayText.LEFT
        o.vertical_alignment = OverlayText.TOP
        o.horizontal_distance = int(x)
        o.vertical_distance = int(y)
        o.bg_color = bg
        o.fg_color = fg
        o.text_size = 18.0
        o.line_width = 2
        o.font = "DejaVu Sans"
        o.text = text
        return o

    def _pt(self, x, y, z=0.05):
        p = Point()
        p.x, p.y, p.z = float(x), float(y), float(z)
        return p

    def _append_sector_triangles(self, marker: Marker, a0: float, a1: float, r: float, n: int = 10):
        r = max(0.0, min(float(r), self.radar_max_d))
        for i in range(n):
            t0 = a0 + (a1 - a0) * (i / n)
            t1 = a0 + (a1 - a0) * ((i + 1) / n)
            marker.points.extend([
                self._pt(0.0, 0.0),
                self._pt(r * math.cos(t0), r * math.sin(t0)),
                self._pt(r * math.cos(t1), r * math.sin(t1)),
            ])

    def _publish_text_marker(self, mid: int, theta: float, r: float, text: str):
        t = Marker()
        t.lifetime.sec = 0
        t.lifetime.nanosec = int(0.2 * 1e9)
        t.header.frame_id = self.viz_frame
        t.header.stamp = self.get_clock().now().to_msg()
        t.ns = "safety_radar_text"
        t.id = int(mid)
        t.type = Marker.TEXT_VIEW_FACING
        t.action = Marker.ADD
        t.pose.orientation.w = 1.0
        t.pose.position.x = float(r * math.cos(theta))
        t.pose.position.y = float(r * math.sin(theta))
        t.pose.position.z = 0.25
        t.scale.z = 0.20
        t.color.r, t.color.g, t.color.b, t.color.a = 1.0, 1.0, 1.0, 1.0
        t.text = text
        t.frame_locked = True
        self.pub_radar_text.publish(t)

    def _publish_viz(self, dR, dFR, dF, dFL, dL, ttc_side):
        # overlays
        if self.state == State.NAVIGATE:
            safety_col = self._c(0.0, 1.0, 0.0, 1.0)
        elif self.state == State.SLOW:
            safety_col = self._c(1.0, 1.0, 0.0, 1.0)
        elif self.state == State.EVADE_SIDE:
            safety_col = self._c(0.7, 0.2, 1.0, 1.0)  # purple
        elif self.state == State.STOP_EMERGENCY:
            safety_col = self._c(1.0, 0.0, 0.0, 1.0)
        elif self.state == State.STARTUP_ALLOW_POLICY:
            safety_col = self._c(0.2, 0.4, 1.0, 1.0)
        elif self.state == State.CORRIDOR_DAMPING:
            safety_col = self._c(1.0, 0.55, 0.0, 1.0)
        else:
            safety_col = self._c(1.0, 1.0, 1.0, 1.0)

        if self.mission_status == "GOAL_REACHED":
            mission_col = self._c(0.0, 1.0, 0.0, 1.0)
        else:
            mission_col = self._c(1.0, 0.55, 0.0, 1.0)

        bg = self._c(0.0, 0.0, 0.0, 0.35)

        safety_line = f"● SAFETY: {self.state.name}"
        mission_line = f"● MISSION: {self.mission_status}"
        self.pub_overlay_safety.publish(self._overlay(safety_line, safety_col, bg, x=10, y=10))
        self.pub_overlay_mission.publish(self._overlay(mission_line, mission_col, bg, x=10, y=75))

        # radar wedges
        m = Marker()
        m.header.frame_id = self.viz_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "safety_radar"
        m.id = 0
        m.type = Marker.TRIANGLE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 1.0
        m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.6, 1.0, 0.35
        m.frame_locked = True
        m.points = []

        deg = math.radians
        sectors = [
            (deg(-30), deg(+30), dF),
            (deg(-60), deg(-30), dFR),
            (deg(+30), deg(+60), dFL),
            (deg(-82), deg(-60), dR),
            (deg(+60), deg(+82), dL),
        ]
        for a0, a1, dist in sectors:
            self._append_sector_triangles(m, a0, a1, dist, n=10)

        self.pub_radar.publish(m)

        r_lbl = 1.2
        self._publish_text_marker(0, 0.0,               r_lbl, f"F:{dF:.2f}")
        self._publish_text_marker(1, math.radians(-45), r_lbl, f"FR:{dFR:.2f}")
        self._publish_text_marker(2, math.radians(+45), r_lbl, f"FL:{dFL:.2f}")
        self._publish_text_marker(3, math.radians(-71), r_lbl, f"R:{dR:.2f}")
        self._publish_text_marker(4, math.radians(+71), r_lbl, f"L:{dL:.2f}")
        self._publish_text_marker(5, math.radians(+90), r_lbl, f"TTC:{(ttc_side if ttc_side!=float('inf') else 999.0):.2f}s")

    # ---------- FSM ----------
    def control_loop(self):
        if not self.seen_first_cmd:
            return

        cmd_in = self.last_cmd_in if self.last_cmd_in is not None else Twist()

        # Sector distances (always compute for viz/log even in startup)
        dR = dFR = dF = dFL = dL = 999.0
        if self.last_scan is not None:
            dR, dFR, dF, dFL, dL = self._compute_sector_mins(self.last_scan)

        # keep cached for status
        self.last_dR, self.last_dFR, self.last_dF, self.last_dFL, self.last_dL = dR, dFR, dF, dFL, dL

        # TTC (based on side mins)
        ttcL, ttcR, ttc_side = self._compute_side_ttc(dL, dR)
        self.last_ttcL, self.last_ttcR, self.last_ttc_side = ttcL, ttcR, ttc_side

        # --- STARTUP FULL BYPASS ---
        if self.state == State.STARTUP_ALLOW_POLICY:
            self.pub_cmd.publish(cmd_in)
            if (self.startup_deadline_s is not None) and (self._now_s() >= self.startup_deadline_s):
                self._set_state(State.NAVIGATE, "startup allow expired -> enable safety")

            self._publish_viz(dR, dFR, dF, dFL, dL, ttc_side)
            return

        # --- Goal reached check (mission label only) ---
        goal_valid = False
        dist = None
        if (self.goal_pose is not None) and (self.odom_pose is not None):
            if self.goal_pose.header.frame_id != self.odom_pose.header.frame_id:
                if not self._warned_frame_mismatch:
                    self.get_logger().warn(
                        f"Frame mismatch: goal_pose in '{self.goal_pose.header.frame_id}' "
                        f"but odom in '{self.odom_pose.header.frame_id}'. Goal distance invalid."
                    )
                    self._warned_frame_mismatch = True
            else:
                self._warned_frame_mismatch = False
                rx = self.odom_pose.pose.pose.position.x
                ry = self.odom_pose.pose.pose.position.y
                gx = self.goal_pose.pose.position.x
                gy = self.goal_pose.pose.position.y
                dist = math.hypot(gx - rx, gy - ry)
                goal_valid = True

        self.goal_reached = (goal_valid and (dist <= GOAL_RADIUS))
        mission_status = "GOAL_REACHED" if self.goal_reached else "RUNNING"
        mission_changed = (mission_status != self.prev_mission_status)
        self.mission_status = mission_status

        # reset reason unless changed this loop
        self.last_reason = ""

        # --- Transitions ---
        stop_hit = (min(dF, dFL, dFR, dL, dR) <= D_STOP)
        side_ttc_hit = (ttc_side < TTC_SIDE_THRESH) and (mission_status != "GOAL_REACHED")

        # Priority: hard stop first, then TTC evade (you can swap if you prefer)
        if stop_hit:
            self._set_state(State.STOP_EMERGENCY, "min(d*) <= D_STOP")
        else:
            # NEW: TTC-based side evade (only when actively approaching)
            if side_ttc_hit and self.state not in (State.STOP_EMERGENCY, State.WAIT):
                self._set_state(State.EVADE_SIDE, f"side TTC={ttc_side:.2f}s < {TTC_SIDE_THRESH:.2f}s")

            if self.state in (State.NAVIGATE, State.SLOW, State.CORRIDOR_DAMPING):
                if (dL < D_WALL) and (dR < D_WALL):
                    self._set_state(State.CORRIDOR_DAMPING, "corridor: dL<D_WALL and dR<D_WALL")
                else:
                    if self.state == State.CORRIDOR_DAMPING and (
                        (dL > D_WALL + HYST) or (dR > D_WALL + HYST)
                    ):
                        self._set_state(State.NAVIGATE, "corridor cleared")

                    # NEW: per-sector warn
                    front_near = (dF < D_WARN_F) or (dFL < D_WARN_FL) or (dFR < D_WARN_FR)
                    side_near  = (min(dL, dR) < D_WARN_SIDE)

                    if front_near or side_near:
                        reason = (
                            f"warn: dF={dF:.2f}(<{D_WARN_F}) "
                            f"dFL={dFL:.2f}(<{D_WARN_FL}) "
                            f"dFR={dFR:.2f}(<{D_WARN_FR}) "
                            f"dL={dL:.2f} dR={dR:.2f}"
                        )
                        self._set_state(State.SLOW, reason)
                    else:
                        if self.state == State.SLOW:
                            if (dF >= D_WARN_F + HYST) and \
                               (dFL >= D_WARN_FL + HYST) and \
                               (dFR >= D_WARN_FR + HYST) and \
                               (min(dL, dR) >= D_WARN_SIDE + HYST):
                                self._set_state(State.NAVIGATE, "warn cleared")

            elif self.state == State.EVADE_SIDE:
                if self._time_in_state() >= T_EVADE:
                    self._set_state(State.NAVIGATE, "evade done")

            elif self.state == State.STOP_EMERGENCY:
                self._set_state(State.WAIT, "entered stop -> wait")

            elif self.state == State.WAIT:
                if self._time_in_state() >= T_WAIT_AFTER_STOP:
                    self._set_state(State.NAVIGATE, "wait timer done")

        # --- Output action ---
        out = Twist()

        if self.state == State.NAVIGATE:
            out.linear.x  = clamp(cmd_in.linear.x,  -V_MAX_POLICY, V_MAX_POLICY)
            out.angular.z = clamp(cmd_in.angular.z, -W_MAX_POLICY, W_MAX_POLICY)

        elif self.state == State.SLOW:
            s = V_SLOW_SCALE
            out.linear.x  = clamp(s * cmd_in.linear.x,  -V_MAX_POLICY, V_MAX_POLICY)
            out.angular.z = clamp(s * cmd_in.angular.z, -W_MAX_POLICY, W_MAX_POLICY)

        elif self.state == State.CORRIDOR_DAMPING:
            out.linear.x  = clamp(cmd_in.linear.x * V_CORRIDOR_SCALE, -V_MAX_POLICY, V_MAX_POLICY)
            out.angular.z = clamp(cmd_in.angular.z, -abs(W_CORRIDOR), abs(W_CORRIDOR))

        elif self.state == State.EVADE_SIDE:
            # Reverse by default, or burst forward if the whole front wedge is clear
            out.angular.z = clamp(W_EVADE, -W_MAX_SAFETY, W_MAX_SAFETY)

            front_clear_for_burst = min(dF, dFL, dFR) > D_CLEAR_FRONT

            if front_clear_for_burst:
                out.linear.x = clamp(+V_EVADE_FWD, -V_MAX_SAFETY, V_MAX_SAFETY)
            else:
                out.linear.x = clamp(-V_EVADE_BACK, -V_MAX_SAFETY, V_MAX_SAFETY)

        else:
            out.linear.x = 0.0
            out.angular.z = 0.0

        # --- Logging ---
        t_ros_ns = int(self.get_clock().now().nanoseconds)
        event = ""
        if self.state != self.prev_state:
            event = "STATE_CHANGE"
        elif mission_changed:
            event = "MISSION_CHANGE"

        if event:
            self.csv_w.writerow([
                t_ros_ns, mission_status,
                self.prev_state.name, self.state.name, event, self.last_reason,
                dR, dFR, dF, dFL, dL,
                (ttcR if ttcR != float("inf") else 999.0),
                (ttcL if ttcL != float("inf") else 999.0),
                (ttc_side if ttc_side != float("inf") else 999.0),
                cmd_in.linear.x, cmd_in.angular.z,
                out.linear.x, out.angular.z
            ])
            self.csv_f.flush()

        self.prev_state = self.state
        self.prev_mission_status = mission_status

        # Publish final command
        self.pub_cmd.publish(out)

        # Publish viz
        self._publish_viz(dR, dFR, dF, dFL, dL, ttc_side)

    def destroy_node(self):
        try:
            if hasattr(self, "csv_f") and self.csv_f is not None:
                self.csv_f.flush()
                self.csv_f.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SafetyFSMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
