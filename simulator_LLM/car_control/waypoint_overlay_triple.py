#!/usr/bin/env python3
import rclpy, math, csv, cv2, json
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

def rotx(a): c,s=math.cos(a),math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]],dtype=np.float32)
def roty(a): c,s=math.cos(a),math.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]],dtype=np.float32)
def rotz(a): c,s=math.cos(a),math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=np.float32)

class WpTriple(Node):
    def __init__(self):
        super().__init__("waypoint_overlay_triple")
        # ---- params ----
        self.declare_parameter("camera_topic", "/autodrive/f1tenth_1/front_camera")
        self.declare_parameter("center_csv", "Centerline_points.csv")
        self.declare_parameter("inner_csv",  "inner_bound_points.csv")
        self.declare_parameter("outer_csv",  "outer_bound_points.csv")

        self.declare_parameter("z_inner", 0.00)   # m
        self.declare_parameter("z_center",0.02)
        self.declare_parameter("z_outer", 0.04)

        self.declare_parameter("fx", 1409.85); self.declare_parameter("fy", 1409.85)
        self.declare_parameter("cx", 640.0);   self.declare_parameter("cy", 360.0)

        # camera pose in vehicle frame (Forward+X, Left+Y, Up+Z in CV mode)
        self.declare_parameter("cam_tx", -0.015)  # m
        self.declare_parameter("cam_ty",  0.0)
        self.declare_parameter("cam_tz",  0.25)
        self.declare_parameter("cam_roll", 0.0)       # rad
        self.declare_parameter("cam_pitch",-0.25)     # rad (약 -14°; 바닥 보게)
        self.declare_parameter("cam_yaw",  0.0)       # rad

        # world↔map 보정(지금은 world=map 가정)
        self.declare_parameter("map_dx",   0.0)
        self.declare_parameter("map_dy",   0.0)
        self.declare_parameter("map_dyaw", 0.0)

        # yaw 부호 뒤집기(시뮬레이터/CSV 불일치시)
        self.declare_parameter("invert_yaw", False)

        # OpenCV 카메라 축 사용 여부 (X:우, Y:아래, Z:앞) → 여기선 CV식 대신
        # 차량 기준 (X:앞, Y:좌, Z:위)에서 CV픽셀로 변환할 때 맞춰줍니다.
        self.declare_parameter("use_cv_cam_frame", True)

        # drawing
        self.declare_parameter("dot_radius", 4)
        self.declare_parameter("draw_guides", True)
        self.declare_parameter("label_first", True)
        self.declare_parameter("debug_samples", 5)

        P = lambda k: self.get_parameter(k).get_parameter_value()
        self.fx, self.fy = float(P("fx").double_value), float(P("fy").double_value)
        self.cx, self.cy = float(P("cx").double_value), float(P("cy").double_value)

        self.cam_t = np.array([
            float(P("cam_tx").double_value),
            float(P("cam_ty").double_value),
            float(P("cam_tz").double_value)
        ], dtype=np.float32)

        self.cam_rpy = np.array([
            float(P("cam_roll").double_value),
            float(P("cam_pitch").double_value),
            float(P("cam_yaw").double_value)
        ], dtype=np.float32)

        self.map_dx = float(P("map_dx").double_value)
        self.map_dy = float(P("map_dy").double_value)
        self.map_dyaw = float(P("map_dyaw").double_value)
        self.invert_yaw = bool(P("invert_yaw").bool_value)
        self.use_cv_cam = bool(P("use_cv_cam_frame").bool_value)

        self.dot_r = int(P("dot_radius").integer_value)
        self.draw_guides = bool(P("draw_guides").bool_value)
        self.label_first = bool(P("label_first").bool_value)
        self.debug_samples = int(P("debug_samples").integer_value)

        self.camera_topic = P("camera_topic").string_value
        self.center_csv = P("center_csv").string_value
        self.inner_csv  = P("inner_csv").string_value
        self.outer_csv  = P("outer_csv").string_value

        self.z_inner  = float(P("z_inner").double_value)
        self.z_center = float(P("z_center").double_value)
        self.z_outer  = float(P("z_outer").double_value)

        # pose from /autodrive/.../ips (Point) and yaw fixed/mock (or from IMU if you have)
        self.pose_xy = (0.78, 3.16)   # fallback
        self.yaw = -1.571

        # QoS
        qos = QoSProfile(depth=1); qos.reliability=ReliabilityPolicy.BEST_EFFORT; qos.history=HistoryPolicy.KEEP_LAST
        self.sub_cam = self.create_subscription(Image, self.camera_topic, self.cb_img, qos)
        self.sub_pose = self.create_subscription(Point, "/autodrive/f1tenth_1/ips", self.cb_pose, 10)

        # load waypoints
        self.wps_inner  = self._load_csv2(self.inner_csv)
        self.wps_center = self._load_csv2(self.center_csv)
        self.wps_outer  = self._load_csv2(self.outer_csv)

        self.bridge = CvBridge()
        self.K = np.array([[self.fx,0,self.cx],[0,self.fy,self.cy],[0,0,1]],dtype=np.float32)

        self.get_logger().info(f"Loaded inner/center/outer: {len(self.wps_inner)}/{len(self.wps_center)}/{len(self.wps_outer)}")
        self.get_logger().info(f"Subscribing camera: {self.camera_topic}")

    # ------------------- callbacks -------------------
    def cb_pose(self, msg: Point):
        self.pose_xy = (msg.x, msg.y)

    def cb_img(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}")
            return

        # draw overlay
        try:
            over = self.overlay(bgr.copy())
            out = self.bridge.cv2_to_imgmsg(over, encoding="bgr8")
            # 그냥 화면 보일 수 있게 topic 바꾸지 않고 출력만:
            # 원한다면 /waypoints/overlay 로 publish 해도 됨.
            # 여기서는 showimage 재매핑을 위해 /waypoints/overlay 로 퍼블리시:
            if not hasattr(self, 'pub'):
                self.pub = self.create_publisher(Image, "/waypoints/overlay", 1)
            self.pub.publish(out)
        except Exception as e:
            self.get_logger().warn(f"overlay error: {e}")

    # ------------------- helpers -------------------
    def _load_csv2(self, path):
        pts=[]
        try:
            with open(path,"r") as f:
                rd=csv.reader(f)
                for r in rd:
                    if len(r)>=2:
                        pts.append((float(r[0]), float(r[1])))
        except Exception as e:
            self.get_logger().error(f"CSV load failed {path}: {e}")
        return np.array(pts, dtype=np.float32)

    def _veh_from_world(self, pw):
        """world->vehicle (x forward,y left,z up)"""
        x,y = self.pose_xy
        yaw = -self.yaw if self.invert_yaw else self.yaw

        # world->map correction
        Rm = rotz(self.map_dyaw); tm = np.array([self.map_dx,self.map_dy,0.0],dtype=np.float32)
        Pw = np.c_[pw, np.zeros((pw.shape[0],1),dtype=np.float32)]  # z=0 by default, caller overrides
        Pm = (Pw - tm) @ Rm.T

        # vehicle pose
        Rv = rotz(yaw).T
        tv = np.array([x,y,0.0],dtype=np.float32)

        Pv = (Pm - tv) @ Rv.T
        return Pv

    def _cam_from_vehicle(self, Pv):
        Rc = rotz(self.cam_rpy[2]) @ roty(self.cam_rpy[1]) @ rotx(self.cam_rpy[0])
        Pc = (Pv - self.cam_t) @ Rc.T
        if self.use_cv_cam:
            # vehicle(Xf,Yl,Zu) -> cv(Xr, Yd, Zahead): [Xcv,Ycv,Zcv] = [Yv, -Zv, Xv]
            Xv,Yv,Zv = Pc[:,0], Pc[:,1], Pc[:,2]
            Pc = np.stack([Yv, -Zv, Xv], axis=1)
        return Pc

    def _pixels_from_cam(self, Pc):
        Z = Pc[:,2].copy()
        eps = 1e-6
        Z[Z<eps] = np.nan  # 뒤/평면분기
        x = Pc[:,0]/Z; y = Pc[:,1]/Z
        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        return u, v, Z

    def _draw_points(self, img, u, v, color, radius):
        H,W = img.shape[:2]
        inb = 0
        for ui,vi in zip(u, v):
            if not (np.isfinite(ui) and np.isfinite(vi)):
                continue
            if 0 <= ui < W and 0 <= vi < H:
                cv2.circle(img,(int(ui),int(vi)), radius, color, -1, lineType=cv2.LINE_AA)
                inb += 1
        return inb

    def overlay(self, img):
        pose_str = f"pose=({self.pose_xy[0]:.2f},{self.pose_xy[1]:.2f}), yaw={self.yaw:.3f}"
        self.get_logger().debug(pose_str)

        # z-height 적용
        inner3  = np.c_[self.wps_inner,  np.full((len(self.wps_inner),1),  self.z_inner ,dtype=np.float32)]
        center3 = np.c_[self.wps_center, np.full((len(self.wps_center),1), self.z_center,dtype=np.float32)]
        outer3  = np.c_[self.wps_outer,  np.full((len(self.wps_outer),1),  self.z_outer ,dtype=np.float32)]

        # world->vehicle->camera->pixels
        def project(P3):
            Pv = self._veh_from_world(P3[:,:2])
            Pv[:,2] = P3[:,2]
            Pc = self._cam_from_vehicle(Pv)
            return self._pixels_from_cam(Pc), Pc

        (u_i,v_i,Z_i), Pc_i = project(inner3)
        (u_c,v_c,Z_c), Pc_c = project(center3)
        (u_o,v_o,Z_o), Pc_o = project(outer3)

        # 디버그 샘플 몇 개만
        try:
            ns = self.debug_samples
            for name,(u,v,Z,Pc) in [
                ("inner",(u_i,v_i,Z_i,Pc_i)),
                ("center",(u_c,v_c,Z_c,Pc_c)),
                ("outer",(u_o,v_o,Z_o,Pc_o))
            ]:
                samp = []
                for k in range(min(ns, len(u))):
                    if np.isfinite(u[k]) and np.isfinite(v[k]) and np.isfinite(Z[k]):
                        samp.append(f"{k}: (u,v,Z)=({u[k]:.1f},{v[k]:.1f},{Z[k]:.2f}) camXYZ=({Pc[k,0]:.2f},{Pc[k,1]:.2f},{Pc[k,2]:.2f})")
                if samp:
                    self.get_logger().info(f"{name} sample -> " + " | ".join(samp))
        except Exception as e:
            self.get_logger().warn(f"debug sample error: {e}")

        # draw guides
        if self.draw_guides:
            H,W = img.shape[:2]
            cv2.line(img,(W//2,0),(W//2,H),(128,128,128),1)
            cv2.line(img,(0,H//2),(W,H//2),(128,128,128),1)

        # draw points
        inb_i = self._draw_points(img, u_i, v_i, (0,0,255),   self.dot_r)       # red (inner)
        inb_c = self._draw_points(img, u_c, v_c, (0,255,255), self.dot_r)       # yellow (center)
        inb_o = self._draw_points(img, u_o, v_o, (255,0,0),   self.dot_r)       # blue (outer)

        if self.label_first and len(u_c)>0 and np.isfinite(u_c[0]) and np.isfinite(v_c[0]):
            cv2.putText(img, "wp0", (int(u_c[0])+6, int(v_c[0])-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,200), 1, cv2.LINE_AA)

        self.get_logger().info(
            f"drawn inner/center/outer in-bounds={inb_i}/{inb_c}/{inb_o} at pose=({self.pose_xy[0]:.2f},{self.pose_xy[1]:.2f}), yaw={self.yaw:.3f}"
        )
        return img

def main():
    rclpy.init()
    node = WpTriple()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
