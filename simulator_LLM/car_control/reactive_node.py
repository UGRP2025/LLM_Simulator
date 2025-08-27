import rclpy
from rclpy.node import Node
import math, numpy as np
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from queue import PriorityQueue

def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x

class GapFollower(Node):
    def __init__(self):
        super().__init__('reactive_node')

        # === Topics ===
        ns = '/autodrive/f1tenth_1'
        self.lidar_topic    = f'{ns}/lidar'
        self.steer_topic    = f'{ns}/steering_command'   # Float32 [rad]
        self.throttle_topic = f'{ns}/throttle_command'   # Float32 [-1,1]
        self.odom_topic     = f'{ns}/odom'               # optional

        self.steer_pub    = self.create_publisher(Float32, self.steer_topic,    10)
        self.throttle_pub = self.create_publisher(Float32, self.throttle_topic, 10)
        self.scan_sub     = self.create_subscription(LaserScan, self.lidar_topic, self.lidar_cb, 10)
        self.odom_sub     = self.create_subscription(Odometry, self.odom_topic,  self.odom_cb, 10)

        # === Vehicle / Actuator (Simulator specs) ===
        self.max_steer_rad   = 0.5236        # ±30°
        self.steer_rate_rad  = 3.2           # rad/s
        self.top_speed       = 22.88         # m/s
        self.throttle_minmax = (-1.0, 1.0)

        # Geometry (from autodrive params)
        self.wheelbase       = 0.3240        # m
        self.track_width     = 0.2360        # m
        self.car_width       = 0.2700        # m

        # === Speed scheduling params ===
        # Lateral tire peak ~1.0*g -> conservative ay limit
        self.lat_accel_max   = 4.0           # m/s^2 (tight & twisty-safe)
        self.safe_distance   = 1.20          # m : gap threshold & obs headroom
        self.stop_distance   = 0.50          # m : hard brake threshold
        self.obs_window_deg  = 10.0          # deg around target for d_min
        self.k_obs           = 2.0           # m/s per meter beyond safe_distance
        self.ttc_min         = 0.70          # s : minimal time-to-collision
        self.a_accel_max     = 3.0           # m/s^2 desired accel limit
        self.a_brake_max     = 6.0           # m/s^2 desired brake limit

        # Followability limiter (steering rate → speed cap)
        self.rate_scale_exp  = 1.5           # stronger penalty when rate is over

        # Speed control (PI, uses odom if available)
        self.Kp_v            = 0.25
        self.Ki_v            = 0.05
        self.v_err_int       = 0.0
        self.v_meas          = None
        self.v_des_prev      = 0.0

        # LiDAR smoothing & gap extraction
        self.smooth_window   = 5             # samples (0.25°/step → 1.25° equiv)
        self.jump_threshold  = 1.0           # m (neighbor diff allowed)
        self.base_min_gap    = 40            # floor on min gap steps (dynamic add-on)

        # Front preference window
        self.front_fov_deg   = 70.0

        # Steering smoothing
        self.delta_cmd_prev  = 0.0
        self.delta_ema_alpha = 0.2
        self.t_prev          = self.get_clock().now().nanoseconds * 1e-9

        self.get_logger().info("GapFollower++ (twisty-safe speed caps, rate-limited steer, dynamic gap width)")

    # ---- Callbacks & helpers ----
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
        return np.clip(arr, 0.0, r_max)

    def extract_gaps(self, rng, safe_thr, jump_thr, min_len, max_len):
        good = rng >= safe_thr
        gaps = []
        N = len(rng); i = 0
        while i < N:
            if not good[i]: i += 1; continue
            s = i; i += 1
            while i < N and good[i] and abs(rng[i]-rng[i-1]) <= jump_thr and (i-s) <= max_len:
                i += 1
            e = i-1
            if (e - s + 1) >= min_len:
                gaps.append((s, e))
        return gaps

    def pick_gap(self, gaps, rng, front_lo, front_hi):
        if not gaps: return int(np.argmax(rng))
        best_score, best_mid = -1.0, None
        for (s, e) in gaps:
            mid = (s + e)//2
            width = (e - s + 1)
            depth = float(np.min(rng[s:e+1]))
            # front bias
            front_pen = 1.0
            if mid < front_lo or mid > front_hi:
                off = min(abs(mid-front_lo), abs(mid-front_hi))
                front_pen = 1.0 / (1.0 + 0.01*off)
            score = width * depth * front_pen
            if score > best_score: best_score, best_mid = score, mid
        return best_mid

    def lidar_cb(self, scan: LaserScan):
        # timing
        t_now = self.get_clock().now().nanoseconds * 1e-9
        dt = max(1e-3, t_now - self.t_prev); self.t_prev = t_now

        angle_min, inc = scan.angle_min, scan.angle_increment
        N, r_max = len(scan.ranges), scan.range_max

        # sanitize & smooth
        ranges = self.sanitize_ranges(scan.ranges, r_max)
        ranges = self.box_filter(ranges, self.smooth_window)

        # forward window
        mid_idx = N//2
        front_half = int(round(math.radians(self.front_fov_deg) / inc))
        front_lo = clamp(mid_idx - front_half, 0, N-1)
        front_hi = clamp(mid_idx + front_half, 0, N-1)

        # ----- Dynamic minimum gap steps based on car width -----
        # effective width = car width + safety margin
        w_eff = self.car_width + 0.20  # m
        # use forward distance proxy to estimate angular width needed
        forward_slice = ranges[front_lo:front_hi+1] if front_hi >= front_lo else ranges
        d_proxy = float(np.median(forward_slice)) if forward_slice.size > 0 else 2.0
        d_proxy = max(d_proxy, 0.6)  # avoid asin domain issues
        theta_req = 2.0 * math.asin(min(1.0, w_eff / (2.0 * d_proxy)))
        steps_req = int(theta_req / inc)
        min_gap_steps = max(self.base_min_gap, steps_req)     # dynamic minimum
        max_gap_steps = 600                                   # cap

        # build gaps
        gaps = self.extract_gaps(ranges, self.safe_distance, self.jump_threshold,
                                 min_gap_steps, max_gap_steps)

        # select target index
        target_idx = self.pick_gap(gaps, ranges, front_lo, front_hi)
        target_idx = clamp(target_idx, 0, N-1)

        # steering reference
        delta_ref = angle_min + target_idx * inc
        delta_ref = clamp(delta_ref, -self.max_steer_rad, self.max_steer_rad)

        # steering rate limiting + EMA
        d_prev = self.delta_cmd_prev
        max_step = self.steer_rate_rad * dt
        d_limited = d_prev + clamp(delta_ref - d_prev, -max_step, +max_step)
        d_cmd = (1.0 - self.delta_ema_alpha) * d_limited + self.delta_ema_alpha * d_prev
        d_cmd = clamp(d_cmd, -self.max_steer_rad, self.max_steer_rad)

        # ===== Speed scheduling =====
        # (1) curvature limit
        kappa = abs(math.tan(d_cmd) / max(1e-6, self.wheelbase))
        v_curv = self.top_speed if kappa < 1e-6 else math.sqrt(self.lat_accel_max / kappa)
        v_curv = min(v_curv, self.top_speed)

        # (2) obstacle limit (±obs_window around target)
        obs_half = int(round(math.radians(self.obs_window_deg) / inc))
        lo = clamp(target_idx - obs_half, 0, N-1)
        hi = clamp(target_idx + obs_half, 0, N-1)
        d_min = float(np.min(ranges[lo:hi+1])) if hi >= lo else float(np.min(ranges))
        v_obs = self.k_obs * max(0.0, d_min - self.safe_distance)

        # (3) TTC limit
        # If we are too close, ensure TTC >= ttc_min (rough, forward-looking)
        d_for_ttc = max(0.0, d_min - self.stop_distance)
        v_ttc = d_for_ttc / max(1e-3, self.ttc_min)

        # (4) Steering followability (required steering rate → speed cap)
        delta_rate_req = abs(delta_ref - d_prev) / dt
        if delta_rate_req < 1e-6:
            v_rate = self.top_speed
        else:
            scale = clamp(self.steer_rate_rad / delta_rate_req, 0.2, 1.0)
            v_rate = self.top_speed * (scale ** self.rate_scale_exp)

        # combine
        v_raw = max(0.0, min(v_curv, v_obs, v_ttc, v_rate, self.top_speed))

        # (5) accel/brake rate limits on v_des
        v_prev = self.v_des_prev
        if v_raw >= v_prev:
            v_des = min(v_raw, v_prev + self.a_accel_max * dt)
        else:
            v_des = max(v_raw, v_prev - self.a_brake_max * dt)
        self.v_des_prev = v_des

        # ===== Throttle command =====
        # Open-loop FF
        u_ff = (v_des / self.top_speed) if self.top_speed > 0 else 0.0
        # Feedback (if odom)
        u_fb = 0.0
        if self.v_meas is not None:
            v_err = v_des - self.v_meas
            self.v_err_int += v_err * dt
            u_fb = self.Kp_v * v_err + self.Ki_v * self.v_err_int

        # Close obstacle: strong braking
        u_brake = 0.0
        if d_min < self.stop_distance:
            u_brake = -clamp((self.stop_distance - d_min) / max(1e-3, self.stop_distance), 0.0, 1.0)

        # Idle torque compensation (small positive bias)
        idle_comp = 0.06

        throttle = clamp(u_ff + u_fb + idle_comp + u_brake, *self.throttle_minmax)

        # publish
        steer_msg = Float32(); steer_msg.data = d_cmd
        thr_msg   = Float32(); thr_msg.data   = throttle
        self.steer_pub.publish(steer_msg)
        self.throttle_pub.publish(thr_msg)

        self.delta_cmd_prev = d_cmd

        # Debug (optional)
        # self.get_logger().info(f"δref={delta_ref:+.3f}, δcmd={d_cmd:+.3f}, v_des={v_des:.2f}, "
        #                        f"v_curv={v_curv:.2f}, v_obs={v_obs:.2f}, v_ttc={v_ttc:.2f}, v_rate={v_rate:.2f}, "
        #                        f"d_min={d_min:.2f}, throttle={throttle:+.2f}")

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
