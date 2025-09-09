#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from typing import Optional

from sensor_msgs.msg import JointState
from std_msgs.msg import Float32


class JS2Counts(Node):
    """
    JointState → Float32(counts) 브릿지
    - 입력: JointState (position: rad), 토픽당 1개 조인트만 오든 여러 개가 오든 처리
    - 출력: Float32 (연속 언랩 후 → rev → counts, 전진 방향 부호 encoder_sign 적용)
    파라미터:
      in_left_topic  (str) : 입력 Left  토픽 (기본 /autodrive/f1tenth_1/left_encoder)
      in_right_topic (str) : 입력 Right 토픽 (기본 /autodrive/f1tenth_1/right_encoder)
      out_left_topic (str) : 출력 Left  토픽 (기본 /autodrive/f1tenth_1/left_encoder_clean)
      out_right_topic(str) : 출력 Right 토픽 (기본 /autodrive/f1tenth_1/right_encoder_clean)
      counts_per_rev (float): CPR, 1회전당 카운트 수 (기본 2048.0)
      encoder_sign   (float): 전진 시 counts 증가 방향(기본 +1.0, 반대면 -1.0)
      left_joint_name  (str): JointState.name에서 사용할 조인트 이름 (빈 문자열이면 position[0] 사용)
      right_joint_name (str): JointState.name에서 사용할 조인트 이름
    """

    def __init__(self):
        super().__init__("encoder_bridge")

        # ---- Params ----
        self.declare_parameter("in_left_topic",  "/autodrive/f1tenth_1/left_encoder")
        self.declare_parameter("in_right_topic", "/autodrive/f1tenth_1/right_encoder")
        self.declare_parameter("out_left_topic",  "/autodrive/f1tenth_1/left_encoder_clean")
        self.declare_parameter("out_right_topic", "/autodrive/f1tenth_1/right_encoder_clean")

        self.declare_parameter("counts_per_rev", 2048.0)
        self.declare_parameter("encoder_sign", 1.0)

        self.declare_parameter("left_joint_name",  "")
        self.declare_parameter("right_joint_name", "")

        p = lambda k: self.get_parameter(k).get_parameter_value()

        self.inL  = p("in_left_topic").string_value
        self.inR  = p("in_right_topic").string_value
        self.outL = p("out_left_topic").string_value
        self.outR = p("out_right_topic").string_value

        self.cpr  = float(p("counts_per_rev").double_value)
        self.sign = float(p("encoder_sign").double_value)

        self.left_name  = p("left_joint_name").string_value.strip()
        self.right_name = p("right_joint_name").string_value.strip()

        # ---- QoS: 센서용(BEST_EFFORT) ----
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST

        # 입력은 JointState로만 구독 → 혼합 타입 토픽에서도 JS 퍼블리셔에만 연결됨
        self.subL = self.create_subscription(JointState, self.inL,  self.cb_left,  qos)
        self.subR = self.create_subscription(JointState, self.inR,  self.cb_right, qos)

        # 출력은 Float32 단일 타입
        self.pubL = self.create_publisher(Float32, self.outL, 10)
        self.pubR = self.create_publisher(Float32, self.outR, 10)

        # 연속 언랩 상태 저장 (조인트별)
        self._prev = {}  # key -> last_unwrapped_angle (rad)

        self.get_logger().info(
            f"[ENC-BRIDGE] inL={self.inL} inR={self.inR}  outL={self.outL} outR={self.outR} "
            f"cpr={self.cpr} sign={self.sign} names=('{self.left_name}','{self.right_name}')"
        )

    # -------- Helpers --------
    def _unwrap(self, key: str, ang: float) -> float:
        """2π 언랩: 이전 값과 연속되도록 각도(rad) 보정"""
        if key not in self._prev:
            self._prev[key] = ang
            return ang
        a0 = self._prev[key]
        da = ang - a0
        # -pi~pi 범위로 차이를 접어 연속성 유지
        while da > math.pi:
            da -= 2.0 * math.pi
        while da < -math.pi:
            da += 2.0 * math.pi
        a1 = a0 + da
        self._prev[key] = a1
        return a1

    def _pick_angle(self, msg: JointState, prefer_name: str, key_hint: str) -> Optional[float]:
        """name이 주어지면 해당 index, 아니면 position[0] 사용"""
        try:
            if prefer_name:
                if msg.name and prefer_name in msg.name:
                    i = msg.name.index(prefer_name)
                    if i < len(msg.position):
                        return float(msg.position[i])
                    return None
            # fallback: 첫 번째 포지션 사용
            if msg.position:
                return float(msg.position[0])
        except Exception:
            pass
        self.get_logger().warn(f"[ENC-BRIDGE] {key_hint}: cannot pick angle from JointState")
        return None

    def _process_side(self, msg: JointState, prefer_name: str, key_hint: str, pub):
        ang = self._pick_angle(msg, prefer_name, key_hint)
        if ang is None:
            return
        # 언랩 → rev → counts → sign 적용
        key = f"{key_hint}_unwrap"
        ang_u = self._unwrap(key, ang)
        revs = ang_u / (2.0 * math.pi)
        counts = self.sign * revs * self.cpr
        pub.publish(Float32(data=float(counts)))

    # -------- Callbacks --------
    def cb_left(self, msg: JointState):
        self._process_side(msg, self.left_name, "left", self.pubL)

    def cb_right(self, msg: JointState):
        self._process_side(msg, self.right_name, "right", self.pubR)


def main():
    rclpy.init()
    node = JS2Counts()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
