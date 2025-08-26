#!/usr/bin/env python3
import rclpy, time, json, base64, cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from openai import OpenAI
from threading import Lock

SYSTEM_PROMPT = (
    "You are a driving-assistant VLM for an F1TENTH car in a simulator. "
    "From the front camera image, estimate a short-horizon (<=5 m) hint. "
    "You can view the front part of the car, floor(grey) and the wall(black)."
    "Avoid wall to drive safely."
    "Respond STRICT JSON with keys: "
    "{lane_preference: one of [left,center,right,none], "
    "speed_factor: float in [0,1], confidence: float in [0,1], reason: string}. "
    "If unsure, set lane_preference='none', speed_factor<=0.6, confidence<=0.4."
)

class VlmFromCam(Node):
    def __init__(self):
        super().__init__("vlm_from_cam")
        # ---- 파라미터 ----
        self.declare_parameter("camera_topic", "/autodrive/f1tenth_1/front_camera")
        self.declare_parameter("api_base", "http://127.0.0.1:8000/v1")
        self.declare_parameter("api_key", "EMPTY")
        self.declare_parameter("model", "qwen2_5_vl")
        self.declare_parameter("hz", 3.0)
        self.declare_parameter("jpeg_w", 640)
        self.declare_parameter("jpeg_h", 384)
        self.declare_parameter("jpeg_q", 80)
        self.declare_parameter("max_tokens", 120)
        self.declare_parameter("temperature", 0.1)

        p = lambda k: self.get_parameter(k).get_parameter_value()
        self.camera_topic = p("camera_topic").string_value
        self.jpeg_w = int(p("jpeg_w").integer_value)
        self.jpeg_h = int(p("jpeg_h").integer_value)
        self.jpeg_q = int(p("jpeg_q").integer_value)
        self.max_tokens = int(p("max_tokens").integer_value)
        self.temperature = float(p("temperature").double_value)

        self.client = OpenAI(base_url=p("api_base").string_value,
                             api_key=p("api_key").string_value)
        self.model = p("model").string_value

        self.bridge = CvBridge()
        self._latest_bgr = None
        self._img_lock = Lock()

        # 센서 QoS는 Best Effort가 흔해서 SensorData QoS 사용
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST

        self.sub = self.create_subscription(Image, self.camera_topic, self.cb_img, qos)
        period = 1.0 / max(float(p("hz").double_value), 0.1)
        self.timer = self.create_timer(period, self.tick)

        self.get_logger().info(f"Subscribing: {self.camera_topic}")
        self.get_logger().info(f"Connecting VLM @ {p('api_base').string_value}, model={self.model}")

    def cb_img(self, msg: Image):
        try:
            # 시뮬레이터는 보통 'bgr8' 또는 'rgb8' 사용
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self._img_lock:
                self._latest_bgr = bgr
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}")

    def tick(self):
        frame = None
        with self._img_lock:
            if self._latest_bgr is not None:
                frame = self._latest_bgr.copy()
        if frame is None:
            return

        try:
            # 리사이즈 + JPEG 인코딩
            resized = cv2.resize(frame, (self.jpeg_w, self.jpeg_h))
            ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_q])
            if not ok:
                return
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            self.get_logger().debug(f"Encoded JPEG size: {len(buf)} bytes")
            self.get_logger().debug(f"Base64 length: {len(b64)} chars")
            self.get_logger().debug(f"Base64 preview: {b64[:80]}...")
            
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

            # 코드블록/잡텍스트 정리
            if txt.startswith("```"):
                txt = txt.strip("` \n")
                lines = [ln for ln in txt.splitlines() if not ln.lower().startswith("json")]
                txt = "\n".join(lines)

            data = json.loads(txt)
            lane = str(data.get("none")).lower()
            if lane not in ("left", "center", "right", "none"):
                lane = "none"
            speed = float(data.get("speed_factor", 0.6))
            conf  = float(data.get("confidence", 0.0))
            reason= str(data.get("reason", ""))

            hint = {
                "lane": lane,
                "speed": max(0.0, min(1.0, speed)),
                "confidence": max(0.0, min(1.0, conf)),
                "reason": reason
            }
            self.get_logger().info(f"VLM hint: {hint}")

        except Exception as e:
            self.get_logger().warn(f"VLM call failed: {e}")

def main():
    rclpy.init()
    node = VlmFromCam()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
