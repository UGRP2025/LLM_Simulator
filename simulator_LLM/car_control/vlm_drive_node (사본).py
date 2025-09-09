#!/usr/bin/env python3
import rclpy, json, base64, cv2, math
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from openai import OpenAI
from threading import Lock

# 선택적 Ackermann (있으면 퍼블리시)
try:
    from ackermann_msgs.msg import AckermannDriveStamped
    HAS_ACK = True
except Exception:
    HAS_ACK = False

SYSTEM_PROMPT = (
    "You are an on-board VLM controller for an F1TENTH car in a simulator.\n"
    "From the single front camera image, output a STRICT JSON with direct driving commands.\n"
    "NO extra text, NO code fences.\n"
    "\n"
    "REQUIRED keys (always include all of them):\n"
    "- control_mode: must be \"throttle_steer\".\n"
    "- steering_angle: float, radians, range [-0.6, 0.6].\n"
    "- throttle: float in [0,1].\n"
    "- confidence: float in [0,1].\n"
    "- reason: short string (<80 chars).\n"
    "\n"
    "Optional keys:\n"
    "- steering_rate_limit: float rad/s (e.g., 2.0).\n"
    "- emergency_stop: boolean.\n"
    "- ts_ms: int (ms timestamp).\n"
    "\n"
    "Safety defaults if unsure:\n"
    "- throttle<=0.15, abs(steering_angle)<=0.1, confidence<=0.4.\n"
    "- ALWAYS return valid JSON only.\n"
)

class VlmDrive(Node):
    def __init__(self):
        super().__init__("vlm_drive")

        # ---------- Parameters ----------
        # 토픽/네임스페이스
        self.declare_parameter("base_ns", "/autodrive/f1tenth_1")
        self.declare_parameter("camera_topic_rel", "front_camera")
        self.declare_parameter("drive_topic_rel", "drive")  # Ackermann(옵션)
        self.declare_parameter("steer_cmd_topic_rel", "steering_command")   # Float32 [-1,1] 권장
        self.declare_parameter("throttle_cmd_topic_rel", "throttle_command")# Float32 [0,1]

        # 주기 분리
        self.declare_parameter("vlm_hz", 0.3)   # 카메라→VLM 호출(느림)
        self.declare_parameter("cmd_hz", 20.0)  # 펄스/퍼블리시(빠름)

        # VLM/인코딩
        self.declare_parameter("api_base", "http://127.0.0.1:8000/v1")
        self.declare_parameter("api_key",  "EMPTY")
        self.declare_parameter("model",    "qwen2_5_vl")
        self.declare_parameter("jpeg_w", 640)
        self.declare_parameter("jpeg_h", 384)
        self.declare_parameter("jpeg_q", 80)
        self.declare_parameter("max_tokens", 120)
        self.declare_parameter("temperature", 0.1)

        # 조향/안전
        self.declare_parameter("max_steer_rad", 0.6)       # 내부 라디안 상한(안전캡)
        self.declare_parameter("steer_rate_limit", 2.0)    # rad/s (cmd 주기 기준)
        self.declare_parameter("steering_is_rad", False)   # True=rad 그대로, False=[-1,1] 정규화
        self.declare_parameter("steer_norm_deg_max", 30.0) # ±30° -> ±1.0
        self.declare_parameter("invert_steering", True)    # 현재 환경 기본 ON
        self.declare_parameter("flip_camera_horizontal", False)

        # 스로틀(펄스 제어)
        self.declare_parameter("throttle_min", 0.0)
        self.declare_parameter("throttle_max", 0.3)
        self.declare_parameter("throttle_scale", 0.15)     # VLM throttle * scale → 펄스 크기
        self.declare_parameter("pulse_duration_s", 0.05)   # 시간 기반 펄스 길이(엔코더 Off일 때)

        # 엔코더 기반 마이크로 이동(옵션)
        self.declare_parameter("use_encoder_micro_move", False)
        self.declare_parameter("left_encoder_topic_rel", "left_encoder")
        self.declare_parameter("right_encoder_topic_rel", "right_encoder")
        self.declare_parameter("encoder_delta_counts", 20.0)
        self.declare_parameter("pulse_timeout_s", 0.12)    # 엔코더 모드 타임아웃 (짧게!)

        # 디버그
        self.declare_parameter("log_vlm_json", True)

        # ---------- Resolve params ----------
        p = lambda k: self.get_parameter(k).get_parameter_value()
        base_ns = p("base_ns").string_value.rstrip("/")
        topic = lambda rel: f"{base_ns}/{rel}"

        self.camera_topic    = topic(p("camera_topic_rel").string_value)
        self.drive_topic     = topic(p("drive_topic_rel").string_value)
        self.steer_topic     = topic(p("steer_cmd_topic_rel").string_value)
        self.throttle_topic  = topic(p("throttle_cmd_topic_rel").string_value)
        self.left_enc_topic  = topic(p("left_encoder_topic_rel").string_value)
        self.right_enc_topic = topic(p("right_encoder_topic_rel").string_value)

        self.vlm_hz = float(p("vlm_hz").double_value)
        self.cmd_hz = float(p("cmd_hz").double_value)

        self.jpeg_w = int(p("jpeg_w").integer_value)
        self.jpeg_h = int(p("jpeg_h").integer_value)
        self.jpeg_q = int(p("jpeg_q").integer_value)
        self.max_tokens = int(p("max_tokens").integer_value)
        self.temperature = float(p("temperature").double_value)

        self.max_steer = float(p("max_steer_rad").double_value)
        self.rate_lim  = float(p("steer_rate_limit").double_value)
        self.steering_is_rad = bool(p("steering_is_rad").bool_value)
        self.steer_norm_deg_max = float(p("steer_norm_deg_max").double_value)
        self.invert_steering = bool(p("invert_steering").bool_value)
        self.flip_cam = bool(p("flip_camera_horizontal").bool_value)

        self.th_min = float(p("throttle_min").double_value)
        self.th_max = float(p("throttle_max").double_value)
        self.th_scale = float(p("throttle_scale").double_value)
        self.pulse_dur = float(p("pulse_duration_s").double_value)

        self.use_enc = bool(p("use_encoder_micro_move").bool_value)
        self.enc_delta = float(p("encoder_delta_counts").double_value)
        self.pulse_timeout = float(p("pulse_timeout_s").double_value)

        self.log_vlm_json = bool(p("log_vlm_json").bool_value)

        self.client = OpenAI(base_url=p("api_base").string_value,
                             api_key=p("api_key").string_value)
        self.model = p("model").string_value

        # ---------- State ----------
        self.bridge = CvBridge()

        # 최신 카메라 프레임
        self._img_lock = Lock()
        self._latest_bgr = None

        # 최신 VLM 명령 (공유상태)
        self._cmd_lock = Lock()
        self._vlm_new_cmd = False
        self._vlm_steer_des = 0.0   # rad
        self._vlm_thr_in = 0.0      # [0,1]
        self._vlm_emg = False
        self._vlm_conf = 0.0
        self._vlm_reason = ""

        # 퍼블리시 쪽 상태
        self._steer_prev = 0.0      # rad (rate-limit)
        self._pulse_active = False
        self._pulse_value = 0.0
        self._pulse_end_t = 0.0
        self._enc_left = 0.0
        self._enc_right = 0.0
        self._enc_target = None
        self._pulse_deadline = 0.0

        # ---------- QoS ----------
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        qos_sensor = QoSProfile(depth=1)
        qos_sensor.reliability = ReliabilityPolicy.BEST_EFFORT
        qos_sensor.history = HistoryPolicy.KEEP_LAST

        # ---------- Subs / Pubs ----------
        self.create_subscription(Image, self.camera_topic, self.cb_image, qos_sensor)

        if self.use_enc:
            self.create_subscription(Float32, self.left_enc_topic, self.cb_left_enc, 10)
            self.create_subscription(Float32, self.right_enc_topic, self.cb_right_enc, 10)

        self.pub_steer = self.create_publisher(Float32, self.steer_topic, 10)
        self.pub_throt = self.create_publisher(Float32, self.throttle_topic, 10)
        self.pub_ack   = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10) if HAS_ACK else None

        # ---------- Timers (분리) ----------
        cmd_period = 1.0 / max(self.cmd_hz, 0.1)
        vlm_period = 1.0 / max(self.vlm_hz, 0.01)
        self.cmd_timer = self.create_timer(cmd_period, self.tick_cmd)  # 빠른 퍼블리시/펄스
        self.vlm_timer = self.create_timer(vlm_period, self.tick_vlm)  # 느린 VLM 호출

        # ---- Banner ----
        self.get_logger().info(f"[VLM] Camera: {self.camera_topic}")
        self.get_logger().info(f"[CMD ] Steering(Float32): {self.steer_topic} (mode={'rad' if self.steering_is_rad else 'norm[-1,1]'}, norm_deg_max={self.steer_norm_deg_max})")
        self.get_logger().info(f"[CMD ] Throttle(Float32): {self.throttle_topic}")
        if self.pub_ack:
            self.get_logger().info(f"[CMD ] Ackermann: {self.drive_topic}")
        self.get_logger().info(f"[CFG ] invert_steering={self.invert_steering}  flip_camera_horizontal={self.flip_cam}  log_vlm_json={self.log_vlm_json}")
        if self.use_enc:
            self.get_logger().info(f"[ENC ] using encoder micro-move: delta={self.enc_delta} timeout={self.pulse_timeout}s  topics=({self.left_enc_topic},{self.right_enc_topic})")
        self.get_logger().info(f"[HZ  ] vlm_hz={self.vlm_hz}  cmd_hz={self.cmd_hz}")
        self.get_logger().info(f"[VLM] model={self.model}")

    # ---------- Helpers ----------
    def now_sec(self) -> float:
        t = self.get_clock().now()
        return t.nanoseconds / 1e9

    # ---------- Callbacks ----------
    def cb_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self._img_lock:
                self._latest_bgr = bgr
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}")

    def cb_left_enc(self, msg: Float32):
        try:
            self._enc_left = float(msg.data)
        except Exception:
            pass

    def cb_right_enc(self, msg: Float32):
        try:
            self._enc_right = float(msg.data)
        except Exception:
            pass

    # ---------- VLM Tick (느림) ----------
    def tick_vlm(self):
        frame = None
        with self._img_lock:
            if self._latest_bgr is not None:
                frame = self._latest_bgr.copy()
        if frame is None:
            return

        try:
            # 이미지 인코딩
            resized = cv2.resize(frame, (self.jpeg_w, self.jpeg_h))
            if self.flip_cam:
                resized = cv2.flip(resized, 1)
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
            if txt.startswith("```"):
                txt = txt.strip("` \n")
                lines = [ln for ln in txt.splitlines() if not ln.lower().startswith("json")]
                txt = "\n".join(lines)

            if self.log_vlm_json:
                self.get_logger().info(f"[VLM RAW] {txt}")

            data = json.loads(txt)

            # ---- Parse JSON (throttle_steer) ----
            emg       = bool(data.get("emergency_stop", False))
            steer_des = float(data.get("steering_angle", 0.0))  # rad
            thr_in    = float(data.get("throttle", 0.0))        # [0,1]

            try:
                conf = float(data.get("confidence", 0.0))
                if not (0.0 <= conf <= 1.0):
                    conf = 0.0
            except Exception:
                conf = 0.0
            reason = str(data.get("reason", "")).strip() or f"fallback: steer={steer_des:.2f}rad thr={thr_in:.2f}"

            # clamp steering
            steer_des = max(-self.max_steer, min(self.max_steer, steer_des))

            with self._cmd_lock:
                self._vlm_steer_des = steer_des
                self._vlm_thr_in = thr_in
                self._vlm_emg = emg
                self._vlm_conf = conf
                self._vlm_reason = reason
                self._vlm_new_cmd = True  # 새 펄스 시작 신호

        except Exception as e:
            self.get_logger().warn(f"VLM call/parse failed: {e}")

    # ---------- CMD Tick (빠름) ----------
    def tick_cmd(self):
        now = self.now_sec()

        # 최신 VLM 명령 스냅샷
        with self._cmd_lock:
            steer_des = float(self._vlm_steer_des)
            thr_in    = float(self._vlm_thr_in)
            emg       = bool(self._vlm_emg)
            conf      = float(self._vlm_conf)
            reason    = str(self._vlm_reason)
            new_cmd   = bool(self._vlm_new_cmd)
            # 바로 소모 (한 번만 펄스 시작)
            self._vlm_new_cmd = False

        # 스티어 rate limit (cmd 주기 기준)
        dt = max(1.0 / max(self.cmd_hz, 0.1), 1e-3)
        max_delta = self.rate_lim * dt
        steer_rad = max(self._steer_prev - max_delta, min(self._steer_prev + max_delta, steer_des))
        self._steer_prev = steer_rad

        # 스티어 publish 값 구성
        if self.steering_is_rad:
            steer_pub = float(steer_rad)
        else:
            deg = math.degrees(steer_rad)
            denom = max(self.steer_norm_deg_max, 1e-6)
            steer_pub = max(-1.0, min(1.0, deg / denom))
        if self.invert_steering:
            steer_pub = -steer_pub

        # Emergency stop → 즉시 스로틀 0/펄스 종료
        if emg:
            self._pulse_active = False
            self._pulse_value = 0.0
            self._pulse_end_t = now
            self._enc_target = None
        else:
            # 새 VLM 명령이 들어온 순간에만 펄스 시작/갱신
            if new_cmd:
                target = max(self.th_min, min(self.th_max, thr_in * self.th_scale))
                self._pulse_value = float(target)
                self._pulse_active = True
                if self.use_enc:
                    enc_now = 0.5 * (self._enc_left + self._enc_right)
                    self._enc_target = enc_now + self.enc_delta
                    self._pulse_deadline = now + self.pulse_timeout
                else:
                    self._pulse_end_t = now + self.pulse_dur

        # 현재 스로틀 결정
        throttle_pub = 0.0
        if self._pulse_active:
            if self.use_enc:
                enc_now = 0.5 * (self._enc_left + self._enc_right)
                if (self._enc_target is not None and enc_now >= self._enc_target) or (now >= self._pulse_deadline):
                    self._pulse_active = False
                    self._enc_target = None
                else:
                    throttle_pub = self._pulse_value
            else:
                if now >= self._pulse_end_t:
                    self._pulse_active = False
                else:
                    throttle_pub = self._pulse_value

        # 퍼블리시
        self.pub_steer.publish(Float32(data=steer_pub))
        self.pub_throt.publish(Float32(data=float(throttle_pub)))

        if self.pub_ack:
            ack = AckermannDriveStamped()
            ack.header.stamp = self.get_clock().now().to_msg()
            ack.drive.steering_angle = steer_rad  # 라디안 그대로
            ack.drive.speed = 0.0
            self.pub_ack.publish(ack)

        # 로그
        self.get_logger().info(
            f"[CMD] steer_rad={steer_rad:.3f} steer_pub({'rad' if self.steering_is_rad else 'norm'})={steer_pub:.3f} "
            f"thr_in={thr_in:.2f} pulse_thr={throttle_pub:.2f} pulse_active={self._pulse_active} "
            f"conf={conf:.2f} reason='{reason}'"
        )

    # ---------- Safe stop ----------
    def safe_stop(self):
        try:
            self.pub_steer.publish(Float32(data=0.0))
            self.pub_throt.publish(Float32(data=0.0))
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

