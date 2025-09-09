#!/usr/bin/env python3
import rclpy, json, base64, cv2, math
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from openai import OpenAI
from threading import Lock

# 옵션: Ackermann 호환 퍼블리시
try:
    from ackermann_msgs.msg import AckermannDriveStamped
    HAS_ACK = True
except Exception:
    HAS_ACK = False

# === 파인튜닝 프롬프트 반영 ===
SYSTEM_PROMPT = (
    "You are a driving-assistant VLM for an F1TENTH car in a simulator. "
    "From the front camera image, estimate the whole situation and give a short-horizon (<=3 seconds) hint. "
    "You can view the front part of the car, floor(grey) and the wall(black). "
    "The road is a loop enclosed by black walls, so there exist at least one way to go. "
    "Avoid wall and find the way to drive safely. "
    "Respond STRICT JSON with keys: {steering_angle: float in [-0.6,0.6], throttle: float in [0,1], "
    "confidence: float in [0,1], reason: string with the recognized situation and driving hint with following format: "
    "\\\"Situation: ... Hint: ...\\\"}. "
    "If unsure, steer a lot to get different view point and set throttle < 0.1 to be safe, with confidence<=0.4. "
    "NO extra text, NO code fences."
)

class VlmDrive(Node):
    def __init__(self):
        super().__init__("vlm_drive")

        # ---------- Topics / Namespace ----------
        self.declare_parameter("base_ns", "/autodrive/f1tenth_1")
        self.declare_parameter("camera_topic_rel", "front_camera")
        self.declare_parameter("drive_topic_rel", "drive")
        self.declare_parameter("steer_cmd_topic_rel", "steering_command")    # std_msgs/Float32
        self.declare_parameter("throttle_cmd_topic_rel", "throttle_command") # std_msgs/Float32
        self.declare_parameter("speed_topic_rel", "speed")                    # std_msgs/Float32 (m/s)

        # ---------- Frequencies ----------
        self.declare_parameter("vlm_hz", 0.3)   # 이미지→VLM 추론
        self.declare_parameter("cmd_hz", 20.0)  # 퍼블리시/펄스

        # ---------- VLM / encoding ----------
        self.declare_parameter("api_base", "http://127.0.0.1:8000/v1")
        self.declare_parameter("api_key",  "EMPTY")
        self.declare_parameter("model",    "qwen2_5_vl")
        self.declare_parameter("jpeg_w", 640)
        self.declare_parameter("jpeg_h", 384)
        self.declare_parameter("jpeg_q", 80)
        self.declare_parameter("max_tokens", 120)
        self.declare_parameter("temperature", 0.1)

        # ---------- Steering ----------
        self.declare_parameter("max_steer_rad", 0.6)
        self.declare_parameter("steer_rate_limit", 2.0)  # rad/s
        self.declare_parameter("steering_is_rad", False) # False => [-1,1]
        self.declare_parameter("steer_norm_deg_max", 30.0)
        self.declare_parameter("invert_steering", True)
        self.declare_parameter("flip_camera_horizontal", False)

        # ---------- Throttle modes ----------
        self.declare_parameter("throttle_mode", "velocity")  # "velocity" | "pulse"

        # velocity 모드
        self.declare_parameter("v_min_mps", 0.15)
        self.declare_parameter("v_max_mps", 0.45)
        self.declare_parameter("hold_decay_rate", 0.9)

        # pulse 모드 (엔코더로 종료)
        self.declare_parameter("throttle_min", 0.0)
        self.declare_parameter("throttle_max", 0.35)
        self.declare_parameter("throttle_scale", 0.25)

        # ---------- 엔코더 기반 펄스 끊기 ----------
        self.declare_parameter("use_encoder_micro_move", False)
        self.declare_parameter("left_encoder_topic_rel", "left_encoder")
        self.declare_parameter("right_encoder_topic_rel", "right_encoder")

        # 이동 목표 단위: "counts" | "revs" | "meters"
        self.declare_parameter("encoder_move_by", "revs")
        self.declare_parameter("encoder_delta_counts", 20.0)   # counts 모드
        self.declare_parameter("counts_per_rev", 2048.0)       # revs/meters 변환
        self.declare_parameter("wheel_rotations_per_cmd", 1.0) # revs 모드: N바퀴
        self.declare_parameter("wheel_radius_m", 0.03)         # meters 모드: 바퀴 반경
        self.declare_parameter("distance_m_per_cmd", 0.0)      # meters 모드: 이동거리
        self.declare_parameter("encoder_sign", 1.0)            # 카운트 증가 방향 맞추기

        # ---------- NEW: 펄스 중 조향 고정 + 오버슈트 보정 ----------
        self.declare_parameter("hold_steer_while_pulse", True)
        self.declare_parameter("taper_enable", True)
        self.declare_parameter("taper_start_ratio", 0.25)  # 전체 목표 카운트의 25% 남았을 때부터 감쇠
        self.declare_parameter("taper_min_scale", 0.25)    # 최소 스로틀 스케일

        # ---------- Latency compensation ----------
        self.declare_parameter("actuation_latency_s", 0.15)
        self.declare_parameter("wheel_base_m", 0.324)
        self.declare_parameter("lead_gain", 1.0)

        # ---------- Debug ----------
        self.declare_parameter("log_vlm_json", True)

        # ---------- Resolve params ----------
        p = lambda k: self.get_parameter(k).get_parameter_value()
        base_ns = p("base_ns").string_value.rstrip("/")
        topic = lambda rel: f"{base_ns}/{rel}"

        self.camera_topic    = topic(p("camera_topic_rel").string_value)
        self.drive_topic     = topic(p("drive_topic_rel").string_value)
        self.steer_topic     = topic(p("steer_cmd_topic_rel").string_value)
        self.throttle_topic  = topic(p("throttle_cmd_topic_rel").string_value)
        self.speed_topic     = topic(p("speed_topic_rel").string_value)
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

        self.throttle_mode = p("throttle_mode").string_value
        self.v_min = float(p("v_min_mps").double_value)
        self.v_max = float(p("v_max_mps").double_value)
        self.hold_decay = float(p("hold_decay_rate").double_value)

        self.th_min = float(p("throttle_min").double_value)
        self.th_max = float(p("throttle_max").double_value)
        self.th_scale = float(p("throttle_scale").double_value)

        self.use_enc = bool(p("use_encoder_micro_move").bool_value)
        self.enc_move_by = p("encoder_move_by").string_value
        self.enc_delta_counts = float(p("encoder_delta_counts").double_value)
        self.counts_per_rev = float(p("counts_per_rev").double_value)
        self.wheel_rots = float(p("wheel_rotations_per_cmd").double_value)
        self.wheel_radius = float(p("wheel_radius_m").double_value)
        self.distance_m = float(p("distance_m_per_cmd").double_value)
        self.encoder_sign = float(p("encoder_sign").double_value)

        self.tau = float(p("actuation_latency_s").double_value)
        self.L = float(p("wheel_base_m").double_value)
        self.lead_gain = float(p("lead_gain").double_value)

        self.hold_steer = bool(p("hold_steer_while_pulse").bool_value)
        self.taper_enable = bool(p("taper_enable").bool_value)
        self.taper_ratio = float(p("taper_start_ratio").double_value)
        self.taper_min = float(p("taper_min_scale").double_value)

        self.log_vlm_json = bool(p("log_vlm_json").bool_value)

        self.client = OpenAI(base_url=p("api_base").string_value,
                             api_key=p("api_key").string_value)
        self.model = p("model").string_value

        # ---------- State ----------
        self.bridge = CvBridge()
        self._img_lock = Lock()
        self._latest_bgr = None

        self._cmd_lock = Lock()
        self._vlm_new_cmd = False
        self._vlm_steer_des = 0.0   # rad
        self._vlm_thr_in = 0.0      # [0,1]
        self._vlm_emg = False
        self._vlm_conf = 0.0
        self._vlm_reason = ""

        self._steer_prev = 0.0
        self._pulse_active = False
        self._pulse_value = 0.0
        self._enc_left = 0.0
        self._enc_right = 0.0
        self._enc_target = None
        self._target_delta_counts = 0.0
        self._pulse_steer_hold = None     # 펄스 동안 고정할 steer_pub
        self._pending_pulse_steer = None  # 이번 틱에 계산된 steer_pub 후보

        self._speed_meas = 0.0
        self._last_v_cmd = 0.0

        # ---------- QoS & I/O ----------
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        qos_sensor = QoSProfile(depth=1)
        qos_sensor.reliability = ReliabilityPolicy.BEST_EFFORT
        qos_sensor.history = HistoryPolicy.KEEP_LAST

        self.create_subscription(Image, self.camera_topic, self.cb_image, qos_sensor)
        self.create_subscription(Float32, self.speed_topic, self.cb_speed, 10)
        if self.use_enc:
            self.create_subscription(Float32, self.left_enc_topic, self.cb_left_enc, 10)
            self.create_subscription(Float32, self.right_enc_topic, self.cb_right_enc, 10)

        self.pub_steer = self.create_publisher(Float32, self.steer_topic, 10)
        self.pub_throt = self.create_publisher(Float32, self.throttle_topic, 10)
        self.pub_ack   = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10) if HAS_ACK else None

        # ---------- Timers ----------
        cmd_period = 1.0 / max(self.cmd_hz, 0.1)
        vlm_period = 1.0 / max(self.vlm_hz, 0.01)
        self.cmd_timer = self.create_timer(cmd_period, self.tick_cmd)
        self.vlm_timer = self.create_timer(vlm_period, self.tick_vlm)

        # ---------- Banner ----------
        self.get_logger().info(f"[VLM] Camera: {self.camera_topic}")
        self.get_logger().info(f"[CMD ] Steering: {self.steer_topic} (mode={'rad' if self.steering_is_rad else 'norm[-1,1]'}, norm_deg_max={self.steer_norm_deg_max})")
        self.get_logger().info(f"[CMD ] Throttle: {self.throttle_topic} (throttle_mode={self.throttle_mode})")
        self.get_logger().info(f"[MEAS] Speed: {self.speed_topic}")
        if self.pub_ack:
            self.get_logger().info(f"[CMD ] Ackermann: {self.drive_topic}")
        self.get_logger().info(f"[CFG ] invert_steering={self.invert_steering} flip_camera_horizontal={self.flip_cam} log_vlm_json={self.log_vlm_json}")
        if self.use_enc:
            self.get_logger().info(
                f"[ENC ] move_by={self.enc_move_by} counts_per_rev={self.counts_per_rev} "
                f"wheel_rots={self.wheel_rots} radius_m={self.wheel_radius} dist_m={self.distance_m} sign={self.encoder_sign}"
            )
        self.get_logger().info(f"[TAPER] hold_steer={self.hold_steer} taper_enable={self.taper_enable} "
                               f"start_ratio={self.taper_ratio} min_scale={self.taper_min}")
        self.get_logger().info(f"[HZ  ] vlm_hz={self.vlm_hz} cmd_hz={self.cmd_hz}")
        self.get_logger().info(f"[VLM] model={self.model}")

    # ---------- Helpers ----------
    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _enc_now(self) -> float:
        # 전진 시 증가하도록 부호 정규화
        return self.encoder_sign * 0.5 * (self._enc_left + self._enc_right)

    def _delta_counts_from_setting(self) -> float:
        if self.enc_move_by == "counts":
            return max(0.0, self.enc_delta_counts)
        elif self.enc_move_by == "revs":
            return max(0.0, self.wheel_rots) * max(1e-6, self.counts_per_rev)
        elif self.enc_move_by == "meters":
            circ = 2.0 * math.pi * max(1e-6, self.wheel_radius)
            revs = max(0.0, self.distance_m) / circ
            return revs * max(1e-6, self.counts_per_rev)
        # fallback
        return max(0.0, self.enc_delta_counts)

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

    # ---------- VLM Tick (slow) ----------
    def tick_vlm(self):
        frame = None
        with self._img_lock:
            if self._latest_bgr is not None:
                frame = self._latest_bgr.copy()
        if frame is None:
            return
        try:
            # JPEG 인코딩
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

            # ---- Parse JSON (simplified) ----
            emg       = bool(data.get("emergency_stop", False))
            steer_des = float(data.get("steering_angle", 0.0))  # rad
            thr_in    = float(data.get("throttle", 0.0))        # [0,1]
            try:
                conf = float(data.get("confidence", 0.0))
                if not (0.0 <= conf <= 1.0): conf = 0.0
            except Exception:
                conf = 0.0
            reason = str(data.get("reason", "")).strip() or f"fallback: steer={steer_des:.2f} thr={thr_in:.2f}"

            steer_des = max(-self.max_steer, min(self.max_steer, steer_des))

            with self._cmd_lock:
                self._vlm_steer_des = steer_des
                self._vlm_thr_in = thr_in
                self._vlm_emg = emg
                self._vlm_conf = conf
                self._vlm_reason = reason
                self._vlm_new_cmd = True
        except Exception as e:
            self.get_logger().warn(f"VLM call/parse failed: {e}")

    # ---------- CMD Tick (fast) ----------
    def tick_cmd(self):
        # 최신 VLM 명령 스냅샷
        with self._cmd_lock:
            steer_des = float(self._vlm_steer_des)
            thr_in    = float(self._vlm_thr_in)
            emg       = bool(self._vlm_emg)
            conf      = float(self._vlm_conf)
            reason    = str(self._vlm_reason)
            new_cmd   = bool(self._vlm_new_cmd)
            self._vlm_new_cmd = False

        # --- Latency lead compensation (bicycle approx) ---
        v = max(self._speed_meas, 0.0)
        lead = self.lead_gain * (v / max(self.L, 1e-6)) * math.tan(steer_des) * max(self.tau, 0.0)
        steer_des_lead = max(-self.max_steer, min(self.max_steer, steer_des + lead))

        # Steering rate limit (cmd_hz)
        dt = max(1.0 / max(self.cmd_hz, 0.1), 1e-3)
        max_delta = self.rate_lim * dt
        steer_rad = max(self._steer_prev - max_delta, min(self._steer_prev + max_delta, steer_des_lead))
        self._steer_prev = steer_rad

        # Steering output unit (여기서 이번 틱 후보값 저장)
        if self.steering_is_rad:
            steer_pub = float(steer_rad)
        else:
            deg = math.degrees(steer_rad)
            denom = max(self.steer_norm_deg_max, 1e-6)
            steer_pub = max(-1.0, min(1.0, deg / denom))
        if self.invert_steering:
            steer_pub = -steer_pub
        self._pending_pulse_steer = steer_pub  # 이번 틱 계산된 출력값

        # ---------- Throttle output ----------
        throttle_pub = 0.0
        if emg:
            # 즉시 정지
            self._pulse_active = False
            self._pulse_value = 0.0
            self._enc_target = None
            self._pulse_steer_hold = None
            self._last_v_cmd = 0.0
        else:
            if self.throttle_mode == "velocity":
                if new_cmd:
                    v_cmd = self.v_min + max(0.0, min(1.0, thr_in)) * max(0.0, self.v_max - self.v_min)
                else:
                    v_cmd = self._last_v_cmd * self.hold_decay
                throttle_pub = float(v_cmd)
                self._last_v_cmd = v_cmd
            else:  # ----- pulse (엔코더 전용 종료) -----
                if not self.use_enc:
                    self.get_logger().warn("[ENC] use_encoder_micro_move is False; pulse ignored (throttle=0)")
                else:
                    if new_cmd:
                        target = max(self.th_min, min(self.th_max, thr_in * self.th_scale))
                        self._pulse_value = float(target)
                        self._pulse_active = True

                        # 엔코더 타깃 생성 (단위 변환)
                        self._target_delta_counts = self._delta_counts_from_setting()
                        enc_now = self._enc_now()
                        self._enc_target = enc_now + self._target_delta_counts

                        # 펄스 동안 조향 고정
                        if self.hold_steer:
                            self._pulse_steer_hold = self._pending_pulse_steer

                        self.get_logger().info(
                            f"[ENC] pulse start: val={self._pulse_value:.2f} "
                            f"delta_counts={self._target_delta_counts:.1f} enc_now={enc_now:.1f} target={self._enc_target:.1f} "
                            f"hold_steer={'on' if self._pulse_steer_hold is not None else 'off'}"
                        )

                    if self._pulse_active:
                        enc_now = self._enc_now()
                        rem = self._enc_target - enc_now if (self._enc_target is not None) else 0.0
                        if self._enc_target is not None and rem <= 0.0:
                            self.get_logger().info(
                                f"[ENC] pulse end: enc_now={enc_now:.1f} target={self._enc_target:.1f} rem=0.0"
                            )
                            self._pulse_active = False
                            self._enc_target = None
                            self._pulse_steer_hold = None
                        else:
                            # 오버슈트 보정: 남은 구간이 짧아지면 스로틀 감쇠
                            out_val = self._pulse_value
                            if self.taper_enable and self._target_delta_counts > 0.0:
                                start = max(1.0, self.taper_ratio * self._target_delta_counts)  # 시작 구간(카운트)
                                if rem <= start:
                                    # 선형 스케일: rem/start ∈ [0,1] → [min_scale,1]로 보정
                                    lin = max(0.0, min(1.0, rem / start))
                                    scale = max(self.taper_min, lin)
                                    out_val *= scale
                                    self.get_logger().info(
                                        f"[TAPER] rem={rem:.1f}/{start:.1f} scale={scale:.2f} out={out_val:.2f}"
                                    )
                            throttle_pub = out_val

        # ---------- 조향 publish (펄스 중 고정 적용) ----------
        if self._pulse_active and self.hold_steer and (self._pulse_steer_hold is not None):
            steer_to_publish = float(self._pulse_steer_hold)
        else:
            steer_to_publish = float(self._pending_pulse_steer)

        self.pub_steer.publish(Float32(data=steer_to_publish))
        self.pub_throt.publish(Float32(data=float(throttle_pub)))

        if self.pub_ack:
            ack = AckermannDriveStamped()
            ack.header.stamp = self.get_clock().now().to_msg()
            # ack에는 실제 라디안 값을 싣되, hold 중에도 steer_rad(내부 추정)로 유지됨
            ack.drive.steering_angle = self._steer_prev
            ack.drive.speed = 0.0
            self.pub_ack.publish(ack)

        # ---------- Log ----------
        enc_state = ""
        if self.use_enc:
            enc_now = self._enc_now()
            enc_state = f" enc_now={enc_now:.1f}"
            if self._enc_target is not None:
                enc_state += f" tgt={self._enc_target:.1f} rem={max(0.0, self._enc_target - enc_now):.1f}"

        self.get_logger().info(
            f"[CMD] mode={self.throttle_mode} v_meas={self._speed_meas:.2f} "
            f"steer_hold={'on' if (self._pulse_active and self._pulse_steer_hold is not None) else 'off'} "
            f"thr_in={thr_in:.2f} out={throttle_pub:.2f} active={self._pulse_active} "
            f"conf={conf:.2f}{enc_state} reason='{reason}'"
        )

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

