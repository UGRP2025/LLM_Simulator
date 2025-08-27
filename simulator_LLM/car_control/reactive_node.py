import rclpy
from rclpy.node import Node
import math, time
import numpy as np
from queue import PriorityQueue
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry

def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x

class GapFollower(Node):
    def __init__(self):
        super().__init__('reactive_node')

        # === Topics ===
        ns = '/autodrive/f1tenth_1'
        self.lidar_topic    = f'{ns}/lidar'
        self.steer_topic    = f'{ns}/steering_command'   # expects steering angle (rad)
        self.throttle_topic = f'{ns}/throttle_command'   # expects throttle [-1,1]
        self.odom_topic     = f'{ns}/odom'               # optional

        # === Publishers & Subscribers ===
        self.steer_pub    = self.create_publisher(Float32, self.steer_topic,    10)
        self.throttle_pub = self.create_publisher(Float32, self.throttle_topic, 10)
        self.scan_sub     = self.create_subscription(LaserScan, self.lidar_topic, self.lidar_cb, 10)
        self.odom_sub     = self.create_subscription(Odometry, self.odom_topic,  self.odom_cb, 10)

        # === Vehicle/Actuator (from simulator doc) ===
        self.max_steer_rad   = 0.5236         # ±30°
        self.steer_rate_rad  = 3.2            # rad/s
        self.top_speed       = 22.88          # m/s
        self.throttle_minmax = (-1.0, 1.0)    # [-1,1]
        self.motor_torque    = 428.0          # Nm (정보용)
        # Geometry (adjust to your car; affects curvature speed):
        self.wheelbase       = 0.33           # m (F1TENTH typical)
        self.track_width     = 0.19           # m (정보용)

        # === Control params ===
        self.lat_accel_max   = 6.0            # m/s^2  (tunable)
        self.safe_distance   = 2.0            # m      (gap threshold)
        self.stop_distance   = 0.7            # m      (hard brake threshold)
        self.obs_window_deg  = 15.0           # deg for v_obs
        self.k_obs           = 3.0            # m/s per meter beyond safe_distance
        self.v_openloop_gamma= 1.0            # throttle FF exponent (1=linear)
        self.idle_comp       = 0.06           # to overcome idle brake torque (tunable)

        # Speed PI (only used if odometry available)
        self.Kp_v            = 0.25
        self.Ki_v            = 0.05
        self.v_err_int       = 0.0
        self.v_meas          = None           # m/s (from odom)

        # LiDAR smoothing
        self.smooth_window   = 7              # samples (box filter length)
        self.jump_threshold  = 1.5            # m between neighbors allowed
        self.min_gap_len     = 30             # samples
        self.max_gap_len     = 500            # samples

        # Steering smoothing / rate limit
        self.delta_cmd_prev  = 0.0
        self.delta_ema_alpha = 0.2
        self.t_prev          = self.get_clock().now().nanoseconds * 1e-9

        self.get_logger().info("GapFollower+ (rate-limited steer, curvature speed, torque-aware throttle)")

    # ---- Utils ----
    def odom_cb(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.v_meas = float(np.hypot(vx, vy))

    def box_filter(self, arr, w):
        if w <= 1: return np.asarray(arr, dtype=float)
        kernel = np.ones(w, dtype=float)/w
        return np.convolve(arr, kernel, mode='same')

    def sanitize_ranges(self, ranges, r_max):
        arr = np.asarray(ranges, dtype=float)
        arr[~np.isfinite(arr)] = r_max
        arr = np.clip(arr, 0.0, r_max)
        return arr

    def extract_gaps(self, rng, safe_thr, jump_thr, min_len, max_len):
        # contiguous indices where rng >= safe_thr and without big jumps
        good = rng >= safe_thr
        gaps = []
        N = len(rng)
        i = 0
        while i < N:
            if not good[i]:
                i += 1
                continue
            start = i
            i += 1
            while i < N and good[i] and abs(rng[i]-rng[i-1]) <= jump_thr and (i-start) <= max_len:
                i += 1
            end = i-1
            if (end - start + 1) >= min_len:
                gaps.append((start, end))
        return gaps

    def pick_gap(self, gaps, rng, front_lo, front_hi):
        if not gaps:
            return int(np.argmax(rng))
        # score by angular width * min depth, with bonus for being near straight ahead
        best_score, best_mid = -1.0, None
        for (s, e) in gaps:
            mid = (s + e) // 2
            width = (e - s + 1)
            depth = float(np.min(rng[s:e+1]))
            # front bias: penalize away from [front_lo, front_hi]
            front_pen = 1.0
            if mid < front_lo or mid > front_hi:
                off = min(abs(mid-front_lo), abs(mid-front_hi))
                front_pen = 1.0 / (1.0 + 0.01*off)
            score = width * depth * front_pen
            if score > best_score:
                best_score, best_mid = score, mid
        return best_mid

    # ---- Main callback ----
    def lidar_cb(self, scan: LaserScan):
        # Time and dt
        t_now = self.get_clock().now().nanoseconds * 1e-9
        dt = max(1e-3, t_now - self.t_prev)
        self.t_prev = t_now

        angle_min = scan.angle_min
        angle_max = scan.angle_max
        inc       = scan.angle_increment
        N         = len(scan.ranges)
        r_max     = scan.range_max

        # Sanitize & smooth
        ranges = self.sanitize_ranges(scan.ranges, r_max)
        ranges = self.box_filter(ranges, self.smooth_window)

        # Define a forward FOV for target preference (e.g., ±60°)
        fov_deg  = 60.0
        mid_idx  = N // 2
        fov_half = int(round(math.radians(fov_deg) / inc))
        front_lo = clamp(mid_idx - fov_half, 0, N-1)
        front_hi = clamp(mid_idx + fov_half, 0, N-1)

        # Build gaps using safety threshold
        gaps = self.extract_gaps(ranges, self.safe_distance, self.jump_threshold,
                                 self.min_gap_len, self.max_gap_len)

        # Target index = best gap center (with front bias). Fallback: global max range.
        target_idx = self.pick_gap(gaps, ranges, front_lo, front_hi)
        target_idx = clamp(target_idx, 0, N-1)

        # Convert to target steering angle (rad)
        delta_ref = angle_min + target_idx * inc

        # Saturate by steering angle limits
        delta_ref = clamp(delta_ref, -self.max_steer_rad, self.max_steer_rad)

        # Steering rate limiting
        d_raw = delta_ref
        d_prev = self.delta_cmd_prev
        max_step = self.steer_rate_rad * dt
        d_limited = d_prev + clamp(d_raw - d_prev, -max_step, +max_step)

        # Small EMA to reduce jitter
        d_cmd = (1.0 - self.delta_ema_alpha) * d_limited + self.delta_ema_alpha * d_prev
        d_cmd = clamp(d_cmd, -self.max_steer_rad, self.max_steer_rad)

        # === Speed scheduling ===
        # Curvature-limited speed
        kappa = abs(math.tan(d_cmd) / max(1e-6, self.wheelbase))
        v_curv = self.top_speed if kappa < 1e-6 else math.sqrt(self.lat_accel_max / kappa)
        v_curv = min(v_curv, self.top_speed)

        # Obstacle-limited speed (use ±obs_window around target)
        obs_half = int(round(math.radians(self.obs_window_deg) / inc))
        lo = clamp(target_idx - obs_half, 0, N-1)
        hi = clamp(target_idx + obs_half, 0, N-1)
        d_min = float(np.min(ranges[lo:hi+1])) if hi >= lo else float(np.min(ranges))
        v_obs = self.k_obs * max(0.0, d_min - self.safe_distance)

        # Final desired speed
        v_des = max(0.0, min(v_curv, v_obs, self.top_speed))

        # === Throttle command ===
        # Feed-forward (open loop) normalized by top speed
        u_ff = (v_des / self.top_speed) ** self.v_openloop_gamma if self.top_speed > 0 else 0.0

        # Feedback PI if odom available
        u_fb = 0.0
        if self.v_meas is not None:
            v_err = v_des - self.v_meas
            self.v_err_int += v_err * dt
            u_fb = self.Kp_v * v_err + self.Ki_v * self.v_err_int

        # Close obstacle → strong braking
        u_brake = 0.0
        if d_min < self.stop_distance:
            # linear braking to -1.0 as it gets closer
            u_brake = -clamp((self.stop_distance - d_min) / max(1e-3, self.stop_distance), 0.0, 1.0)

        throttle = u_ff + u_fb + self.idle_comp + u_brake
        throttle = clamp(throttle, *self.throttle_minmax)

        # === Publish ===
        steer_msg = Float32();   steer_msg.data = d_cmd
        thr_msg   = Float32();   thr_msg.data   = throttle
        self.steer_pub.publish(steer_msg)
        self.throttle_pub.publish(thr_msg)

        # keep for next step
        self.delta_cmd_prev = d_cmd

        # Debug (optional)
        # self.get_logger().info(f"d_cmd={d_cmd:+.3f} rad, v_des={v_des:.2f}, d_min={d_min:.2f}, throttle={throttle:+.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = GapFollower()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
