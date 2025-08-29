#!/usr/bin/env python3
import os, csv, math, signal, time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2

def yaw_from_imu(msg: Imu) -> float:
    w, x, y, z = msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    return math.atan2(siny_cosp, cosy_cosp)

def rot_z(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[ c,-s,0],
                     [ s, c,0],
                     [ 0, 0,1]], dtype=np.float32)

def euler_to_R(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]], dtype=np.float32)
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]], dtype=np.float32)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]], dtype=np.float32)
    return Rz @ Ry @ Rx

def cam_base_rot():
    # 차량(x:앞, y:좌, z:위) -> 카메라(OpenCV, x:우, y:하, z:앞)
    return np.array([[0, -1,  0],
                     [0,  0, -1],
                     [1,  0,  0]], dtype=np.float32)

class WaypointOverlay(Node):
    def __init__(self):
        super().__init__("waypoint_overlay")

        # ---------- Parameters ----------
        self.declare_parameter("camera_topic", "/autodrive/f1tenth_1/front_camera")
        self.declare_parameter("waypoint_csv", "waypoints.csv")
        self.declare_parameter("waypoint_frame", "map")   # "map" or "vehicle"
        self.declare_parameter("waypoint_z", 0.0)         # m

        # 지도 보정(좌표계 오프셋) - NEW
        self.declare_parameter("map_dx", 0.0)             # m
        self.declare_parameter("map_dy", 0.0)             # m
        self.declare_parameter("map_dyaw", 0.0)           # rad

        # intrinsics (cx,cy=0.0이면 이미지 중심 사용)
        self.declare_parameter("fx", 600.0)
        self.declare_parameter("fy", 600.0)
        self.declare_parameter("cx", 0.0)
        self.declare_parameter("cy", 0.0)

        # camera extrinsics (vehicle→camera), 사용자 보정 (카메라 축 기준)
        self.declare_parameter("cam_tx", 0.20)            # m
        self.declare_parameter("cam_ty", 0.00)            # m
        self.declare_parameter("cam_tz", 0.25)            # m
        self.declare_parameter("cam_roll", 0.0)           # rad
        self.declare_parameter("cam_pitch", -0.10)        # rad
        self.declare_parameter("cam_yaw", 0.0)            # rad

        # 좌표계/회전 부호 옵션
        self.declare_parameter("use_cv_cam_frame", True)  # 차량->카메라 고정 회전 사용
        self.declare_parameter("invert_yaw", False)       # map->vehicle 회전 부호 반전

        # drawing & UI
        self.declare_parameter("draw_radius", 5)
        self.declare_parameter("draw_thickness", 3)
        self.declare_parameter("show_window", False)
        self.declare_parameter("draw_guides", True)       # NEW: 중앙선/수평선 그리기

        gp = lambda k: self.get_parameter(k).value
        self.camera_topic   = gp("camera_topic")
        self.waypoint_csv   = gp("waypoint_csv")
        self.waypoint_frame = gp("waypoint_frame")
        self.waypoint_z     = float(gp("waypoint_z"))

        # 지도 보정 파라미터
        self.map_dx   = float(gp("map_dx"))
        self.map_dy   = float(gp("map_dy"))
        self.map_dyaw = float(gp("map_dyaw"))

        self.fx,self.fy = float(gp("fx")), float(gp("fy"))
        self.cx,self.cy = float(gp("cx")), float(gp("cy"))

        self.t_cam = np.array([[float(gp("cam_tx"))],
                               [float(gp("cam_ty"))],
                               [float(gp("cam_tz"))]], dtype=np.float32)

        # 카메라 회전 행렬 구성
        use_cv_cam_frame = bool(gp("use_cv_cam_frame"))
        R_user = euler_to_R(float(gp("cam_roll")),
                            float(gp("cam_pitch")),
                            float(gp("cam_yaw")))
        R_fix = cam_base_rot() if use_cv_cam_frame else np.eye(3, dtype=np.float32)
        self.R_cam = R_user @ R_fix

        self.draw_radius    = int(gp("draw_radius"))
        self.draw_thickness = int(gp("draw_thickness"))
        self.show_window    = bool(gp("show_window"))
        self.draw_guides    = bool(gp("draw_guides"))
        self.invert_yaw     = bool(gp("invert_yaw"))

        # ---------- Waypoints load ----------
        here = os.path.dirname(os.path.abspath(__file__))
        csv_path = self.waypoint_csv if os.path.isabs(self.waypoint_csv) else os.path.join(here, self.waypoint_csv)
        self.waypoints_xy = self._load_csv(csv_path)

        # 지도 보정(Δyaw, Δx, Δy) 적용 - NEW
        if self.waypoints_xy.shape[0] > 0:
            c, s = math.cos(self.map_dyaw), math.sin(self.map_dyaw)
            R2 = np.array([[c, -s], [s, c]], dtype=np.float32)  # 2x2
            self.waypoints_xy = (R2 @ self.waypoints_xy.T).T
            self.waypoints_xy[:,0] += self.map_dx
            self.waypoints_xy[:,1] += self.map_dy

        self.get_logger().info(
            f"Loaded {len(self.waypoints_xy)} waypoints from {csv_path} "
            f"(after map offset: dx={self.map_dx}, dy={self.map_dy}, dyaw={self.map_dyaw})"
        )

        # ---------- ROS IO ----------
        self.bridge = CvBridge()

        img_qos = QoSProfile(depth=10)
        img_qos.reliability = ReliabilityPolicy.RELIABLE
        img_qos.history = HistoryPolicy.KEEP_LAST

        self.sub_img = self.create_subscription(Image, self.camera_topic, self.cb_img, img_qos)
        self.pub_overlay = self.create_publisher(Image, "/waypoints/overlay", 10)

        # 차량 pose/imu (map 프레임일 때 사용)
        self.pose_xy = None
        self.yaw = 0.0
        if str(self.waypoint_frame).lower() == "map":
            self.sub_ips = self.create_subscription(Point, "/autodrive/f1tenth_1/ips", self.cb_pose, 10)
            self.sub_imu = self.create_subscription(Imu,   "/autodrive/f1tenth_1/imu", self.cb_imu, 10)

        # 디버그 카운터
        self._frame_cnt = 0
        self._last_log_t = time.time()

        signal.signal(signal.SIGINT, self._sigint)

        self.get_logger().info(
            f"Params: frame={self.waypoint_frame}, wz={self.waypoint_z}, "
            f"fx,fy={self.fx},{self.fy} cx,cy={self.cx},{self.cy}, "
            f"cam_t={self.t_cam.ravel().tolist()}, "
            f"cam_rpy={(self.get_parameter('cam_roll').value, self.get_parameter('cam_pitch').value, self.get_parameter('cam_yaw').value)}, "
            f"use_cv_cam_frame={use_cv_cam_frame}, invert_yaw={self.invert_yaw}"
        )
        self.get_logger().info(f"Subscribing camera: {self.camera_topic}")

    # --- CSV: x,y ---
    def _load_csv(self, path):
        pts = []
        with open(path, "r", newline="") as f:
            rdr = csv.reader(f)
            for row in rdr:
                if not row or str(row[0]).strip().startswith("#"):
                    continue
                try:
                    x = float(row[0]); y = float(row[1])
                    pts.append((x,y))
                except Exception:
                    continue
        return np.array(pts, dtype=np.float32) if pts else np.zeros((0,2), np.float32)

    def cb_pose(self, msg: Point):
        self.pose_xy = np.array([msg.x, msg.y], dtype=np.float32)

    def cb_imu(self, msg: Imu):
        self.yaw = yaw_from_imu(msg)

    def cb_img(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}")
            return

        self._frame_cnt += 1
        h, w = bgr.shape[:2]
        cx = self.cx if self.cx > 1.0 else w * 0.5
        cy = self.cy if self.cy > 1.0 else h * 0.5

        # 1) world/map → vehicle (2D)
        if str(self.waypoint_frame).lower() == "map":
            if self.pose_xy is None:
                self._publish_and_maybe_show(bgr, msg.header, draw_guides=True)
                return
            yaw_eff = (self.yaw if self.invert_yaw else -self.yaw)
            Rwv = rot_z(yaw_eff)  # world->vehicle 평면 회전 (상단 3x3 중 2x2만 사용)
            t_w = self.pose_xy.reshape(2,1)
            xy_world = self.waypoints_xy.T  # (2,N)
            xy_vehicle = Rwv[:2,:2] @ (xy_world - t_w)  # (2,N)
        else:
            xy_vehicle = self.waypoints_xy.T  # (2,N)

        # 2) vehicle → camera (3D)
        N = xy_vehicle.shape[1] if xy_vehicle.size > 0 else 0
        if N == 0:
            self._publish_and_maybe_show(bgr, msg.header, draw_guides=True)
            return

        P_vehicle = np.vstack([xy_vehicle,
                               np.ones((1,N), np.float32)*float(self.waypoint_z)])  # (3,N)
        Pc = (self.R_cam @ P_vehicle) + self.t_cam  # (3,N)

        # 3) pinhole projection
        Z_all = Pc[2,:]
        front_mask = Z_all > 0.05
        n_front = int(front_mask.sum())
        if n_front == 0:
            self.get_logger().info(f"All points behind/too close. Z=({Z_all.min():.3f},{Z_all.max():.3f})")
            self._publish_and_maybe_show(bgr, msg.header, draw_guides=True)
            return

        X = Pc[0,front_mask]; Y = Pc[1,front_mask]; Z = Z_all[front_mask]
        u = (self.fx * (X / Z) + cx)
        v = (self.fy * (Y / Z) + cy)

        umin, umax = float(np.min(u)), float(np.max(u))
        vmin, vmax = float(np.min(v)), float(np.max(v))
        self.get_logger().debug(f"projected={n_front} u:[{umin:.1f},{umax:.1f}] v:[{vmin:.1f},{vmax:.1f}]")

        u = u.astype(np.int32); v = v.astype(np.int32)

        # 4) draw
        img = bgr.copy()
        if self.draw_guides:
            # 중앙 십자선/수평선
            cv2.line(img, (int(cx), 0), (int(cx), h-1), (100,100,100), 1)
            cv2.line(img, (0, int(cy)), (w-1, int(cy)), (100,100,100), 1)

        H, W = img.shape[:2]
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        n_inb = int(inb.sum())
        self.get_logger().info(f"projected={n_front} drawn(in-bounds)={n_inb}")

        if n_inb > 0:
            ui = u[inb]; vi = v[inb]
            for (uu,vv) in zip(ui,vi):
                cv2.circle(img, (uu,vv), self.draw_radius, (0,255,0), self.draw_thickness)

        self._publish_and_maybe_show(img, msg.header, draw_guides=False)

    def _publish_and_maybe_show(self, img, header, draw_guides=False):
        out = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        out.header = header
        self.pub_overlay.publish(out)
        if self.show_window:
            cv2.imshow("waypoints overlay", img)
            cv2.waitKey(1)

    def _sigint(self, *args):
        if self.show_window:
            cv2.destroyAllWindows()
        rclpy.shutdown()

def main():
    rclpy.init()
    node = WaypointOverlay()
    try:
        rclpy.spin(node)
    finally:
        if node.show_window:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
