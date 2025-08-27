import csv, os, math, time
import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Point
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from ament_index_python.packages import get_package_share_directory

# -------------------- 유틸 --------------------
def clamp(x, lo, hi): 
    return lo if x < lo else hi if x > hi else x

def normalize_angle(a):
    while a > math.pi:  a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def euler_from_quaternion(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    # roll
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch
    sinp = 2 * (w * y - z * x)
    pitch = math.asin(clamp(sinp, -1.0, 1.0))
    # yaw
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

def csv_reading(file_path, col_name):
    idx = 0 if col_name == 'positions_X' else 1 if col_name == 'positions_y' else None
    if idx is None: return []
    out = []
    with open(file_path, 'r') as f:
        for row in csv.reader(f):
            if len(row) <= idx: continue
            try: out.append(float(row[idx]))
            except: pass
    return out

# -------------------- PP 최적 제어 노드 --------------------
class PurePursuitOptimal(Node):
    def __init__(self):
        super().__init__('pure_pursuit_controller')

        # ---- Topics ----
        ns = '/autodrive/f1tenth_1'
        self.steer_topic    = f'{ns}/steering_command'    # Float32: steering angle [rad]
        self.throttle_topic = f'{ns}/throttle_command'    # Float32: throttle [-1,1]
        self.ips_topic      = f'{ns}/ips'                 # geometry_msgs/Point (x,y)
        self.imu_topic      = f'{ns}/imu'                 # sensor_msgs/Imu
        self.odom_topic     = f'{ns}/odom'                # nav_msgs/Odometry (optional)

        # ---- Publishers/Subscribers ----
        self.steer_pub    = self.create_publisher(Float32, self.steer_topic, 10)
        self.throttle_pub = self.create_publisher(Float32, self.throttle_topic, 10)
        self.create_subscription(Point, self.ips_topic, self.ips_cb, 10)
        self.create_subscription(Imu,   self.imu_topic, self.imu_cb, 10)
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 10)

        # ---- Vehicle / Actuator (시뮬레이터 스펙) ----
        self.max_steer     = 0.5236   # [rad] ±30°
        self.steer_rate    = 3.2      # [rad/s]
        self.v_top         = 22.88    # [m/s]
        self.L             = 0.3240   # wheelbase [m]
        self.track         = 0.2360   # [m]
        self.car_width     = 0.2700   # [m]

        # ---- Speed scheduling & safety ----
        # 타이어 횡마찰 피크(~1.0g) 대비 보수적 횡가속 한계
        self.ay_max        = 4.0      # [m/s^2] (twisty-safe)
        # 조향 추종성 기반 속도 캡(조향 속도 한계 초과시 속도 하향)
        self.rate_penalty_exp = 1.5
        # 가감속 레이트 제한
        self.a_accel_max   = 3.0      # [m/s^2]
        self.a_brake_max   = 6.0      # [m/s^2]

        # ---- Lookahead (동적) ----
        self.Ld_min        = 0.6      # [m]
        self.Ld_max        = 2.5      # [m]
        self.k_Ld_v        = 0.25     # [s]  Ld_base = Ld_min + k*v
        self.k_Ld_kappa    = 0.8      # 곡률 클수록 Ld 축소 (Ld /= (1 + k*kappa))

        # ---- Speed control (Throttle PI) ----
        self.Kp_v          = 0.25
        self.Ki_v          = 0.05
        self.v_err_int     = 0.0
        self.idle_comp     = 0.06     # 아이들 브레이크 보정
        self.v_meas        = None     # 측정 속도 [m/s]

        # ---- State ----
        self.yaw           = 0.0
        self.pos           = np.array([0.0, 0.0])
        self.pos_prev      = None
        self.v_est         = 0.0
        self.v_alpha       = 0.3      # 속도 저역통과
        self.delta_prev    = 0.0
        self.v_des_prev    = 0.0
        self.t_prev        = self.get_clock().now().nanoseconds * 1e-9

        # ---- Load path ----
        try:
            pkg_dir   = get_package_share_directory('car_control')
            file_path = os.path.join(pkg_dir, 'CSVs', 'Centerline_points.csv')
            self.get_logger().info(f"Loading waypoints from: {file_path}")
        except Exception as e:
            self.get_logger().error(f"Package path error: {e}")
            rclpy.shutdown(); return

        xs = csv_reading(file_path, 'positions_X')
        ys = csv_reading(file_path, 'positions_y')
        self.path = np.array(list(zip(xs, ys)), dtype=float)
        if self.path.shape[0] == 0:
            self.get_logger().error("Failed to load waypoints.")
            rclpy.shutdown(); return

        # 미리 호 길이 누적(arc-length) 계산
        diffs = np.diff(self.path, axis=0, prepend=self.path[:1])
        seg   = np.linalg.norm(diffs, axis=1)
        seg[0]=0.0
        self.s = np.cumsum(seg)

        # CSV 로깅(선택)
        self.csv_path = "/home/autodrive_devkit/src/simulator_LLM/car_control/CSVs/actual.csv"
        self._csv_header_written = False

        self.get_logger().info(f"Waypoints loaded: {self.path.shape[0]}")

    # ---------- Subscribers ----------
    def imu_cb(self, msg: Imu):
        _, _, yaw = euler_from_quaternion(msg.orientation)
        self.yaw = yaw

    def odom_cb(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.v_meas = float((vx**2 + vy**2)**0.5)

    def ips_cb(self, pt: Point):
        # 시간/DT
        t_now = self.get_clock().now().nanoseconds * 1e-9
        dt = max(1e-3, t_now - self.t_prev)
        self.t_prev = t_now

        # 위치 업데이트 & 속도 추정(IPS)
        p = np.array([pt.x, pt.y], dtype=float)
        if self.pos_prev is not None:
            v_raw = np.linalg.norm(p - self.pos_prev) / dt
            self.v_est = (1-self.v_alpha)*self.v_est + self.v_alpha*v_raw
            if self.v_meas is None:
                self.v_meas = self.v_est
        self.pos_prev = p.copy()
        self.pos = p

        # Pure Pursuit 목표점 선택 (동적 룩어헤드)
        nearest_idx = int(np.argmin(np.linalg.norm(self.path - self.pos, axis=1)))

        # 국소 경로 곡률 근사(세 점 사용)
        i1 = nearest_idx
        i2 = min(i1 + 5,  len(self.path)-1)
        i3 = min(i1 + 10, len(self.path)-1)
        kappa_path = self._curvature_three_points(self.path[i1], self.path[i2], self.path[i3])

        v_now = self.v_meas if self.v_meas is not None else 0.0
        Ld_base = self.Ld_min + self.k_Ld_v * v_now
        Ld_eff  = Ld_base / (1.0 + self.k_Ld_kappa * abs(kappa_path))
        Ld      = clamp(Ld_eff, self.Ld_min, self.Ld_max)

        goal_idx = self._goal_index_by_distance(nearest_idx, Ld)
        goal_pt  = self.path[goal_idx]

        # 차량 좌표계로 변환
        dx = goal_pt[0] - self.pos[0]
        dy = goal_pt[1] - self.pos[1]
        cy = math.cos(self.yaw); sy = math.sin(self.yaw)
        xL =  cy*dx + sy*dy
        yL = -sy*dx + cy*dy

        # Pure Pursuit 조향 (표준식): delta = atan(2L*y / Ld^2)
        delta_ref = math.atan2(2.0 * self.L * yL, max(1e-6, Ld**2))
        delta_ref = clamp(delta_ref, -self.max_steer, self.max_steer)

        # 조향 속도 제한 + EMA 스무딩
        d_prev   = self.delta_prev
        max_step = self.steer_rate * dt
        d_limited = d_prev + clamp(delta_ref - d_prev, -max_step, +max_step)
        d_cmd     = 0.8*d_limited + 0.2*d_prev        # 소량 EMA
        d_cmd     = clamp(d_cmd, -self.max_steer, self.max_steer)

        # ===== 속도 스케줄링 =====
        # (1) 곡률 한계(횡가속)
        kappa = abs(math.tan(d_cmd) / max(1e-6, self.L))
        v_curv = self.v_top if kappa < 1e-6 else math.sqrt(self.ay_max / kappa)
        v_curv = min(v_curv, self.v_top)

        # (2) 조향 추종성 한계 (필요 조향속도 → 속도 캡)
        delta_rate_req = abs(delta_ref - d_prev) / dt
        if delta_rate_req < 1e-6:
            v_rate = self.v_top
        else:
            scale = clamp(self.steer_rate / delta_rate_req, 0.2, 1.0)
            v_rate = self.v_top * (scale ** self.rate_penalty_exp)

        # (3) 통합 및 가/감속 레이트 제한
        v_raw = max(0.0, min(v_curv, v_rate, self.v_top))
        v_prev = self.v_des_prev
        if v_raw >= v_prev:   v_des = min(v_raw, v_prev + self.a_accel_max * dt)
        else:                 v_des = max(v_raw, v_prev - self.a_brake_max * dt)
        self.v_des_prev = v_des

        # ===== Throttle ([-1,1]) =====
        u_ff = (v_des / self.v_top) if self.v_top > 0 else 0.0
        u_fb = 0.0
        if self.v_meas is not None:
            v_err = v_des - self.v_meas
            self.v_err_int += v_err * dt
            u_fb = self.Kp_v * v_err + self.Ki_v * self.v_err_int

        throttle = clamp(u_ff + u_fb + self.idle_comp, -1.0, 1.0)

        # ===== Publish =====
        steer_msg = Float32(); steer_msg.data = d_cmd            # [rad]
        thr_msg   = Float32(); thr_msg.data   = throttle         # [-1,1]
        self.steer_pub.publish(steer_msg)
        self.throttle_pub.publish(thr_msg)

        self.delta_prev = d_cmd

        # ---- CSV 로깅(선택) ----
        try:
            if not self._csv_header_written and (not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0):
                with open(self.csv_path, "a") as f:
                    f.write("positions_X,positions_y\n")
                self._csv_header_written = True
            with open(self.csv_path, "a") as f:
                f.write(f"{self.pos[0]},{self.pos[1]}\n")
        except Exception as e:
            # 로깅 실패는 무시(경로 없을 수 있음)
            pass

    # ---------- Path helpers ----------
    def _goal_index_by_distance(self, start_idx, Ld):
        # s: 누적 호길이. start_idx 이후로 s 차이가 Ld 이상인 첫 인덱스
        s0 = self.s[start_idx]
        i  = start_idx
        n  = len(self.s)
        while i < n-1 and self.s[i] - s0 < Ld:
            i += 1
        return i

    def _curvature_three_points(self, p1, p2, p3):
        # 원의 곡률 k = 4A / (a*b*c) (A=삼각형 면적, a,b,c=변 길이)
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)
        if a < 1e-6 or b < 1e-6 or c < 1e-6: return 0.0
        s = 0.5 * (a + b + c)
        A2 = max(0.0, s*(s-a)*(s-b)*(s-c))   # 헤론
        A  = math.sqrt(A2)
        denom = a*b*c
        if denom < 1e-9: return 0.0
        return 4.0 * A / denom

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitOptimal()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
