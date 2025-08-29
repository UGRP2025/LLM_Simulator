#!/usr/bin/env python3
import rclpy, json, base64, cv2, time
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from openai import OpenAI
from threading import Lock

# 호환 퍼블리시(있으면 좋음)
try:
    from ackermann_msgs.msg import AckermannDriveStamped
    HAS_ACK = True
except Exception:
    HAS_ACK = False

# =========================
#  VLM SYSTEM PROMPT (STRICT JSON, speed_steer)
# =========================
SYSTEM_PROMPT = (
    "You are an on-board VLM controller for an F1TENTH car in a simulator.\n"
    "From the single front camera image, output a STRICT JSON with direct driving commands for Ackermann.\n"
    "NO extra text, NO code fences.\n"
    "\n"
    "Keys (required):\n"
    "- control_mode: must be \"speed_steer\".\n"
    "- steering_angle: float, radians, range [-0.6, 0.6].\n"
    "- speed_mps: float, range [0.0, 6.0].\n"
    "\n"
    "Optional keys:\n"
    "- steering_rate_limit: float rad/s (e.g., 2.0).\n"
    "- speed_limit_mps: float, cap on speed (<=6.0).\n"
    "- emergency_stop: boolean. If true, set speed to 0 and steering_angle to 0.\n"
    "- confidence: float in [0,1].\n"
    "- reason: short string.\n"
    "- ts_ms: int (ms timestamp).\n"
    "\n"
    "Safety defaults if unsure:\n"
    "- control_mode=\"speed_steer\", speed_mps<=1.0, abs(steering_angle)<=0.1, confidence<=0.4.\n"
    "- ALWAYS return valid JSON only.\n"
)

class VlmDrive(Node):
    def __init__(self):
        super().__init__("vlm_drive")

        # ---------- Parameters ----------
        # 네임스페이스 하나만 바꿔도 모든 토픽이 같이 바뀌도록
        self.declare_parameter("base_ns", "/autodrive/f1tenth_1")
        self.declare_parameter("camera_topic_rel", "front_camera")
        self.declare_parameter("speed_topic_rel", "speed")
        self.declare_parameter("drive_topic_rel", "drive")
        self.declare_parameter("steer_cmd_topic_rel", "steering_command")
        self.declare_parameter("throttle_cmd_topic_rel", "throttle_command")

        # VLM/주기/인코딩
        self.declare_parameter("api_base", "http://127.0.0.1:8000/v1")
        self.declare_parameter("api_key",  "EMPTY")
        self.declare_parameter("model",    "qwen2_5_vl")
        self.declare_parameter("hz", 5.0)
        self.declare_parameter("jpeg_w", 640)
        self.declare_parameter("jpeg_h", 384)
        self.declare_parameter("jpeg_q", 80)
        self.declare_parameter("max_tokens", 120)
        self.declare_parameter("temperature", 0.1)

        # 제한/맵핑/제어
        self.declare_parameter("max_steer_rad", 0.6)      # VLM 입력/출력 기준(rad)
        self.declare_parameter("max_speed_mps", 6.0)
        self.declare_parameter("steer_rate_limit", 2.0)   # rad/s
        self.declare_parameter("steering_is_rad", True)   # steering_command 단위: True=rad, False=normalized[-1,1]
        self.declare_parameter("throttle_min", 0.0)
        self.declare_parameter("throttle_max", 1.0)
        self.declare_parameter("throttle_kp", 0.5)        # 간단 P제어 게인
        self.declare_parameter("speed_deadband", 0.05)    # m/s

        p = lambda k: self.get_parameter(k).get_parameter_value()
        base_ns = p("base_ns").string_value.rstrip("/")
        topic = lambda rel: f"{base_ns}/{rel}"

        self.camera_topic   = topic(p("camera_topic_rel").string_value)
        self.speed_topic    = topic(p("speed_topic_rel").string_value)
        self.drive_topic    = topic(p("drive_topic_rel").string_value)
        self.steer_cmd_topic= topic(p("steer_cmd_topic_rel").string_value)
        self.throt_cmd_topic= topic(p("throttle_cmd_topic_rel").string_value)

        self.jpeg_w = int(p("jpeg_w").integer_value)
        self.jpeg_h = int(p("jpeg_h").integer_value)
        self.jpeg_q = int(p("jpeg_q").integer_value)
        self.max_tokens = int(p("max_tokens").integer_value)
        self.temperature = float(p("temperature").double_value)
        self.max_steer = float(p("max_steer_rad").double_value)
        self.max_speed = float(p("max_speed_mps").double_value)
        self.rate_lim  = float(p("steer_rate_limit").double_value)
        self.steering_is_rad = bool(p("steering_is_rad").bool_value)
        self.kp = float(p("throttle_kp").double_value)
        self.th_min = float(p("throttle_min").double_value)
        self.th_max = float(p("throttle_max").double_value)
        self.db = float(p("speed_deadband").double_value)

        self.client = OpenAI(base_url=p("api_base").string_value,
                             api_key=p("api_key").string_value)
        self.model = p("model").string_value

        # ---------- State ----------
        self.bridge = CvBridge()
        self._img_lock = Lock()
        self._latest_bgr = None
        self._steer_prev = 0.0
        self._speed_meas = 0.0

        # ---------- QoS ----------
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        qos_sensor = QoSProfile(depth=1)
        qos_sensor.reliability = ReliabilityPolicy.BEST_EFFORT
        qos_sensor.history = HistoryPolicy.KEEP_LAST

        # ---------- Subs / Pubs ----------
        self.sub_cam = self.create_subscription(Image, self.camera_topic, self.cb_image, qos_sensor)
        self.sub_speed = self.create_subscription(Float32, self.speed_topic, self.cb_speed, 10)

        self.pub_steer_cmd = self.create_publisher(Float32, self.steer_cmd_topic, 10)
        self.pub_throttle_cmd = self.create_publisher(Float32, self.throt_cmd_topic, 10)

        if HAS_ACK:
            self.pub_ack = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        else:
            self.pub_ack = None

        period = 1.0 / max(float(p("hz").double_value), 0.1)
        self.timer = self.create_timer(period, self.tick)

        self.get_logger().info(f"[VLM] Camera: {self.camera_topic}")
        self.get_logger().info(f"[CTRL] Speed(meas): {self.speed_topic}")
        if self.pub_ack:
            self.get_logger().info(f"[PUB ] Ackermann: {self.drive_topic}")
        self.get_logger().info(f"[PUB ] SteeringCmd(Float32): {self.steer_cmd_topic}")
        self.get_logger().info(f"[PUB ] ThrottleCmd(Float32): {self.throt_cmd_topic}")
        self.get_logger().info(f"[VLM] {p('api_base').string_value}, model={self.model}")

    # ---------- Callbacks ----------
    def cb_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self._img_lock:
                self._latest_bgr = bgr
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}")

    def cb_speed(self, msg: Float32):
        try:
            self._speed_meas = float(msg.data)
        except Exception:
            pass

    # ---------- Main Tick ----------
    def tick(self):
        frame = None
        with self._img_lock:
            if self._latest_bgr is not None:
                frame = self._latest_bgr.copy()
        if frame is None:
            return

        try:
            # Encode to JPEG (smaller for VLM)
            resized = cv2.resize(frame, (self.jpeg_w, self.jpeg_h))
            ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_q])
            if not ok:
                return
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")

            content = [
                {"type": "text", "text": SYSTEM_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            txt = (resp.choices[0].message.content or "").strip()

            # Strip code fences if any
            if txt.startswith("```"):
                txt = txt.strip("` \n")
                lines = [ln for ln in txt.splitlines() if not ln.lower().startswith("json")]
                txt = "\n".join(lines)

            data = json.loads(txt)

            # ---- Parse JSON (speed_steer) ----
            mode = str(data.get("control_mode", "speed_steer")).lower()
            emg  = bool(data.get("emergency_stop", False))
            steer_des = float(data.get("steering_angle", 0.0))
            sp_des = float(data.get("speed_mps", 0.0))
            sp_cap = float(data.get("speed_limit_mps", self.max_speed))

            # clamp
            steer_des = max(-self.max_steer, min(self.max_steer, steer_des))
            sp_des = max(0.0, min(self.max_speed, sp_des))
            sp_cmd = 0.0 if emg else min(sp_des, sp_cap)

            # Steering rate limit
            dt = max(self.timer.timer_period_ns / 1e9, 1e-3)
            max_delta = max(0.0, float(self.get_parameter("steer_rate_limit").get_parameter_value().double_value)) * dt \
                        if self.has_parameter("steer_rate_limit") else self.rate_lim * dt
            steer = max(self._steer_prev - max_delta, min(self._steer_prev + max_delta, steer_des))
            self._steer_prev = steer

            # Throttle via simple P control on measured speed
            throttle = 0.0
            if not emg:
                err = sp_cmd - self._speed_meas
                if abs(err) < self.db:
                    err = 0.0
                throttle = self.kp * err
                throttle = max(self.th_min, min(self.th_max, throttle))
            # emergency → 0
            if emg:
                steer = 0.0
                throttle = 0.0

            # ---- Publish steering/throttle commands (Float32) ----
            # steering_command unit selection
            if self.steering_is_rad:
                steer_cmd_val = float(steer)
            else:
                # normalize to [-1, 1] by max_steer
                steer_cmd_val = float(max(-1.0, min(1.0, steer / max(self.max_steer, 1e-6))))

            self.pub_steer_cmd.publish(Float32(data=steer_cmd_val))
            self.pub_throttle_cmd.publish(Float32(data=float(throttle)))

            # ---- Also publish Ackermann for compatibility ----
            if self.pub_ack:
                ack = AckermannDriveStamped()
                ack.header.stamp = self.get_clock().now().to_msg()
                ack.drive.steering_angle = steer
                ack.drive.speed = sp_cmd if not emg else 0.0
                self.pub_ack.publish(ack)

            # Log
            conf = float(data.get("confidence", 0.0))
            reason = str(data.get("reason", ""))[:120]
            self.get_logger().info(
                f"drive: steer(rad)={steer:.3f}  sp_des={sp_des:.2f} "
                f"sp_cap={sp_cap:.2f} sp_meas={self._speed_meas:.2f} "
                f"throttle={throttle:.2f} emg={emg} conf={conf:.2f} reason='{reason}'"
            )

        except Exception as e:
            self.get_logger().warn(f"VLM call/parse failed: {e}")
            self.safe_stop()

    # ---------- Safe stop ----------
    def safe_stop(self):
        try:
            self.pub_steer_cmd.publish(Float32(data=0.0))
            self.pub_throttle_cmd.publish(Float32(data=0.0))
            if self.pub_ack:
                ack = AckermannDriveStamped()
                ack.header.stamp = self.get_clock().now().to_msg()
                ack.drive.steering_angle = 0.0
                ack.drive.speed = 0.0
                self.pub_ack.publish(ack)
        except Exception:
            pass

def main():
    rclpy.init()
    node = VlmDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

