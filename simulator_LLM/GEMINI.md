목표

로우레벨 제어: Pure Pursuit(PP)로 inner/center/outer 3개 레인 중 선택된 웨이포인트 트랙을 추종. 곡률·장애물 거리 기반 속도 프로파일러 적용.

퍼셉션: YOLO(또는 동등 모델)로 전방 좌/정면/우 위험도를 수치화.

VLM 어드바이저(5–10 Hz): 이미지+상황 요약 → {"lane","speed","reason","confidence"} JSON 반환. 이 토큰은 비용함수 가중치 힌트로만 사용(최종 결론은 비용최소화).

세이프티 가드: JSON 스키마 불일치/낮은 신뢰도/타임아웃 시 무시, 충돌/이탈 하드제약 고정, 디폴트는 센터라인 유지.

주기: 제어루프 20–50 Hz(20–50 ms), VLM 힌트는 별 스레드 ≤10 Hz 비동기.

Pure Pursuit는 F1TENTH에서 검증된 고속 경로추종기로, lookahead 조정이 핵심입니다(Adaptive Lookahead 개념). Nav2의 Regulated Pure Pursuit는 코너 감속을 내재화한 참고 사례입니다. 
GitHub
automaticaddison.com

1. 리포지토리 연계 포인트

레인 데이터: simulator_LLM/car_control/CSVs/Centerline_points.csv, inner_bound_points.csv, outer_bound_points.csv (샘플 레인 포인트) 
GitHub
+1

PP 참조 코드: simulator_LLM/car_control/Pure_pursuit.py (구현/인터페이스 일치 유지) 
GitHub

ROS I/O (시뮬레이터 제공)

Pub: /autodrive/f1tenth_1/throttle_command (std_msgs/Float32)

Pub: /autodrive/f1tenth_1/steering_command (std_msgs/Float32)

Sub: /autodrive/f1tenth_1/ips (geometry_msgs/Point) – 차량 평면 위치

Sub: /autodrive/f1tenth_1/imu (sensor_msgs/Imu) – yaw(헤딩) 추출

2. 폴더 구조(제안)
simulator_LLM/car_control/
├─ CSVs/
│  ├─ Centerline_points.csv
│  ├─ inner_bound_points.csv
│  └─ outer_bound_points.csv
├─ pure_pursuit_controller.py      # PP 래퍼(lookahead/게인/좌표 변환 포함)
├─ speed_profiler.py               # 곡률·장애물 거리 기반 목표 속도
├─ lane_loader.py                  # CSV->웨이포인트 로딩/보정/곡률 산출
├─ perception_yolo_bridge.py       # YOLO 결과 -> 좌/정/우 위험도 수치화
├─ vlm_advisor.py                  # VLM 질의 스레드(JSON 스키마/타임아웃 가드)
├─ cost_fusion.py                  # 비용함수 + VLM 힌트 융합 + 후보 레인 선택
├─ behavior_planner_node.py        # 메인 ROS2 노드(아래 스켈레톤 참조)
└─ params.yaml                     # 하이퍼파라미터/토픽/스케일링

3. 레인 데이터 로딩·전처리
3.1 CSV 형식 가정

각 파일은 x,y(필요 시 s 또는 인덱스) 열을 포함. 좌표 단위/프레임은 시뮬레이터 맵 좌표와 일치해야 함.

3.2 보정 파이프라인

CSV 로드 → np.array([(x_i, y_i)])

등간격 재샘플링(직선·곡선 구간 균형)

접선 t(s)·법선 n(s) 계산 → 곡률 κ(s) 수치미분

속도 상한 초기화: v_max_curve(s) = sqrt( max(ε, μ g / |κ(s)|) )

(옵션) Savitzky–Golay 또는 스플라인으로 노이즈 완화

Pure Pursuit는 lookahead 거리에 민감. 레이싱에서는 waypoint별 속성(곡률/장애물 여유)을 반영한 adaptive lookahead가 corner-cutting/오버슈트 완화에 유용합니다. 
GitHub

4. 퍼셉션: 좌/정면/우 위험도
4.1 입력

YOLO 등 객체검출 결과(토픽 자유) → [(bbox, cls, score)]

(가능 시) 거리 추정: 레이저/깊이/사이즈-거리 휴리스틱

4.2 산출

섹터 분할: 이미지 좌/중/우 또는 yaw 기준 전방 각도 섹터

각 섹터에 대해

최근접 장애물 거리 d_left, d_front, d_right

장애물 폭/점유율, 상대 측면 여부(차선 내/밖)

위험도 risk = w_d / (d+ε) + w_w*width_ratio + w_cls*class_weight

perception_yolo_bridge.py는 위 피처를 dict(left, front, right, timestamp)로 출력.

5. VLM 어드바이저(비동기, ≤10 Hz)
5.1 입력 컨텍스트(예시)
TASK: Pick lane & speed for next 10 m.
EGO: v=3.5 m/s, yaw=..., curvature_ahead=0.12 1/m
OBSTACLES: left=2.4m, front=6.7m, right=4.0m
TRACK_FREE: inner=..., center=..., outer=...
RULES: prefer center; avoid obstacles; keep progress; no collisions.
OUTPUT_JSON:
{"lane":"inner|center|outer","speed":"slow|normal|fast",
 "reason":"...", "confidence":0.0~1.0}

5.2 스키마·가드

JSON 파싱 실패/confidence<τ/응답지연(예: 80 ms) → 무시

(옵션) few-shot 예시 3–5개 포함 → 일관된 토큰 유도

최종 조향/가감속은 PP+속도 프로파일러가 수행하고, VLM은 가중치 힌트만 제공합니다(안전·지연 대응). 이는 F1TENTH 권장 운용과 잘 맞습니다. 
GitHub

6. 비용함수 융합(레인 선택)

후보 레인 
𝑟
∈
{
inner
,
center
,
outer
}
r∈{inner,center,outer}:

𝐽
(
𝑟
)
=
𝛼
⋅
1
free_dist
𝑟
+
𝛽
⋅
∣
𝜅
𝑟
∣
−
𝛾
⋅
progress
𝑟
+
𝛿
⋅
lane_change_penalty
+
LLM_hint
(
𝑟
)
J(r)=α⋅
free_dist
r
	​

1
	​

+β⋅∣κ
r
	​

∣−γ⋅progress
r
	​

+δ⋅lane_change_penalty+LLM_hint(r)

(\text{LLM_hint}(r) = w_\text{lane}\cdot \mathbf{1}[r=\text{lane_token}]

w_\text{speed}\cdot \text{speed_factor})

하드제약: 충돌·트랙이탈 위험 후보는 평가 전 제거

파라미터(α,β,γ,δ, w_*)는 실험으로 튜닝

7. 속도 프로파일러
7.1 기본값

곡률 기반: 
𝑣
curve
(
𝑠
)
=
max
⁡
(
𝜖
,
𝜇
𝑔
/
∣
𝜅
(
𝑠
)
∣
)
v
curve
	​

(s)=
max(ϵ,μg/∣κ(s)∣)
	​


장애물 기반 감속: 
𝑣
obs
=
𝑘
𝑑
⋅
min
⁡
(
𝑑
front
,
𝑑
lane
)
v
obs
	​

=k
d
	​

⋅min(d
front
	​

,d
lane
	​

)

최종 목표속도: 
𝑣
∗
=
min
⁡
(
𝑣
curve
,
𝑣
obs
,
𝑣
limit
)
v
∗
=min(v
curve
	​

,v
obs
	​

,v
limit
	​

)

7.2 Regulated PP 영감

코너·블라인드 구간에서 자동 감속을 적용해 조향 안정 및 이탈 감소(Nav2 Regulated PP 참고). 
automaticaddison.com

8. ROS2 메인 노드 스켈레톤 (behavior_planner_node.py)
import rclpy, json, threading, time
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

from lane_loader import load_three_lanes, compute_lane_metrics
from pure_pursuit_controller import PurePursuit
from speed_profiler import target_speed
from perception_yolo_bridge import get_sector_risks
from vlm_advisor import VLMAdvisor
from cost_fusion import select_lane

class BehaviorPlanner(Node):
    def __init__(self):
        super().__init__('behavior_planner')
        # pubs
        self.throttle_pub = self.create_publisher(Float32, "/autodrive/f1tenth_1/throttle_command", 10)
        self.steer_pub    = self.create_publisher(Float32, "/autodrive/f1tenth_1/steering_command", 10)
        # subs
        self.create_subscription(Point, "/autodrive/f1tenth_1/ips", self.cb_pose, 10)
        self.create_subscription(Imu,   "/autodrive/f1tenth_1/imu", self.cb_imu, 10)

        # load lanes & controllers
        lanes = load_three_lanes("CSVs/Centerline_points.csv",
                                 "CSVs/inner_bound_points.csv",
                                 "CSVs/outer_bound_points.csv")
        self.wp = {"center": lanes.center, "inner": lanes.inner, "outer": lanes.outer}
        self.pp = {k: PurePursuit(self.wp[k]) for k in self.wp}  # lookahead per-lane 가능

        # perception + vlm
        self.vlm = VLMAdvisor(hz=8, timeout_s=0.08)  # 비동기 스레드 내부 구동
        self.perception_state = None

        # control loop
        self.timer = self.create_timer(0.04, self.control_loop)  # 25 Hz

        # state
        self.pose = None; self.yaw = None

    def cb_pose(self, msg: Point): self.pose = (msg.x, msg.y)
    def cb_imu(self, msg: Imu): self.yaw = extract_yaw(msg)

    def control_loop(self):
        if self.pose is None or self.yaw is None:
            return

        # 1) Perception
        risks = get_sector_risks()  # dict(left, front, right, timestamp)

        # 2) Lane metrics (free distance, curvature, progress, etc.)
        metrics = compute_lane_metrics(self.pose, self.yaw, self.wp, risks)

        # 3) VLM hint (thread-safe fetch). 실패/타임아웃/낮은 신뢰도면 내부적으로 None 반환
        hint = self.vlm.latest_hint()  # {"lane","speed","confidence"} or None

        # 4) Lane selection with hard constraints
        lane = select_lane(metrics, hint)  # default="center" on failure

        # 5) Low-level control
        steer = self.pp[lane].compute_steer(self.pose, self.yaw)  # Pure Pursuit
        v_cmd = target_speed(metrics[lane], risks, hint)          # speed profile

        # 6) Publish
        self.steer_pub.publish(Float32(data=float(steer)))   # [-1,1] 정규화 가정
        self.throttle_pub.publish(Float32(data=float(v_cmd)))# [0,1] or m/s -> sim 스케일에 맞춤

def main():
    rclpy.init()
    rclpy.spin(BehaviorPlanner())
    rclpy.shutdown()

if __name__ == "__main__":
    main()


주의: 스티어/스로틀의 수치 범위는 시뮬레이터에 맞춰 정규화 필요. 기존 Pure_pursuit.py의 출력 스케일·좌표계와 일치시키세요. 
GitHub

9. 모듈별 구현 포인트
9.1 pure_pursuit_controller.py

입력: 웨이포인트 배열, lookahead 
𝐿
𝑑
L
d
	​

, 차폭/최소곡률 보호 파라미터

출력: 정규화 스티어 
[
−
1
,
1
]
[−1,1]

팁: lookahead는 속도/곡률 따라 적응(테이블 또는 함수). 코너에서 과도한 코너컷 방지. 
GitHub

9.2 speed_profiler.py

target_speed(metrics_r, risks, hint)

곡률 기반 상한, 전방·선택 레인 내 장애물 거리 기반 감속

VLM speed가 slow/normal/fast면 곱셈 계수 
𝑘
𝑠
∈
{
0.7
,
1.0
,
1.2
}
k
s
	​

∈{0.7,1.0,1.2} 등

9.3 lane_loader.py

CSV 로드 → 등간격 샘플 → 곡률/법선 계산 → 누적거리 s 부여

각 레인에 대해 진행도(progress), **자유거리(free_distance)**를 계산하는 유틸 포함

9.4 perception_yolo_bridge.py

YOLO 토픽 구독 또는 폴링 → 좌/정/우 섹터로 투영

각 섹터 최근접 거리·폭 점유율 계산 → risk dict 반환

9.5 vlm_advisor.py

별도 스레드로 5–10 Hz 수행

타임아웃/JSON 스키마/신뢰도 임계 가드

실패 시 None 저장(메인 루프는 힌트 없이 동작)

9.6 cost_fusion.py

하드제약: 레인 후보 중 충돌/이탈 위반 제거

비용 
𝐽
(
𝑟
)
J(r) 계산 + 
arg
⁡
min
⁡
argmin 선택

직전 레인과 다르면 lane_change_penalty 부과(플리커 방지)

10. 파라미터 (params.yaml 예시)
pp:
  lookahead_min: 1.0         # m
  lookahead_max: 4.5
  lookahead_gain_vs_speed: 0.35
speed:
  vmax: 6.0                   # m/s
  mu: 0.9                     # 마찰 가정
  obs_kd: 0.8                 # 거리→감속 계수
risk:
  sector_angles: [-45, 45]    # 왼/정/오 구획 기준
  class_weights: {car: 1.0, cone: 0.6, person: 1.2}
fusion:
  alpha: 2.0
  beta: 1.0
  gamma: 1.0
  delta: 0.5
  w_lane: 0.5
  w_speed: 0.3
vlm:
  hz: 8
  timeout_ms: 80
  conf_thresh: 0.6
topics:
  ips: "/autodrive/f1tenth_1/ips"
  imu: "/autodrive/f1tenth_1/imu"
  steer_cmd: "/autodrive/f1tenth_1/steering_command"
  throttle_cmd: "/autodrive/f1tenth_1/throttle_command"

11. 실행 순서(안)

CSVs 확인 및 좌표계 일치 점검(맵 프레임/단위) 
GitHub

lane_loader.py로 3개 레인 불러와 보정/곡률 계산

behavior_planner_node.py 실행 → 스티어/스로틀 토픽 발행

YOLO 토픽 연결(없으면 위험도=무/전방만)

VLM 연결(없으면 힌트 무시, 센터라인+전통 제어만으로 정상 주행)

파라미터 스윕(lookahead/αβγδ/w_*)

12. 평가·로그

지표: 랩타임, 충돌 수, 오프트랙 %, 최소 장애물거리, 명령지연(95p), 힌트 무시율(파싱/지연)

로그: lane, J(r) 항목별 기여, VLM reason, confidence, v_target, steer

시나리오: 단일 전방 장애물, 좌/우측 장애물, S커브, 가짜 표지판(적대 케이스)

13. 구현 체크리스트

 CSV 불러와 레인 3종 생성(센터/인/아웃)

 Pure Pursuit 래퍼: 속도·곡률에 따른 lookahead 적응

 YOLO 브리지: 좌/정/우 위험도 산출

 속도 프로파일러: 곡률·장애물 거리 기반

 비용함수 융합: 하드제약 + VLM 힌트

 VLM 스레드: JSON 스키마/타임아웃/신뢰도 가드

 ROS2 노드: 25–50 Hz 제어루프, 실패 시 센터라인 유지

 파라미터화(params.yaml) 및 로깅

부록 A. Pure Pursuit 핵심 수식·튜닝 요령

목표점까지의 lookahead 
𝐿
𝑑
L
d
	​

 지정 → 원호 기하로 곡률 
𝜅
κ 산출 → 스티어 비례 제어

고속 주행/코너링에서는 lookahead 증감과 속도 연동이 안정성에 중요

코너·블라인드 즈음 자동 감속은 흔한 개선(Ref: Regulated PP) 
automaticaddison.com

F1TENTH 추천 구현/튜닝 배경은 공식 Pure Pursuit 자료 참조 
GitHub

참고 링크

리포지토리: UGRP2025/LLM_Simulator (폴더·파일 경로) 
GitHub
+2
GitHub
+2

F1TENTH Pure Pursuit(ROS) 설명/튜닝 포인트 
GitHub

Nav2 Regulated Pure Pursuit(코너 감속 개념) 
automaticaddison.com
