"""
teleop_collision.py —— 实验04：弹性碰撞示教录制（稳定斜坡发射版）
====================================================================

设计目标：
  1. 两个小球一开始都在水平轨道上，不会左右乱跑；
  2. 小球被一维/二维 slide joint 约束：
       - 红球 ball1：只能沿 X 前后、Z 上下运动，不允许 Y 左右运动；
       - 蓝球 ball2：只能沿 X 前后运动，不允许 Y/Z 方向乱飞；
  3. 保留按键 7 的夹爪参与：
       - 靠近红球后按 7，红球会被“辅助夹持”到夹爪附近；
       - 把红球拉回斜坡某个高度；
       - 再按 7 松开，红球沿斜坡约束下滑，到水平轨道后与蓝球发生一维碰撞；
  4. 终端状态输出降频，避免刷屏影响观察。

State 12 维：
  [finger1, finger2, j1..j7, v1, v2, time]

Action 8 维：
  [j1_ctrl..j7_ctrl, gripper_ctrl]

用法示例：
  python teleop_collision.py --m1 0.10 --m2 0.10 --v 0.30 --distance 0.34
  python teleop_collision.py --m1 0.10 --m2 0.05 --v 0.30 --distance 0.30

按键：
  1/2   末端 +X / -X
  3/4   末端 -Y / +Y
  5/6   末端 +Z / -Z
  7     夹爪开合；闭合且靠近红球时辅助夹住，打开时释放
  8     开始 / 停止录制
  9     保存退出
  ESC   放弃退出

重要说明：
  - 斜坡和轨道几何主要用于可视化，不参与接触，避免穿模/弹飞；
  - 红球在斜坡段由代码约束到斜坡线，并按无摩擦斜坡加速度加速；
  - 蓝球由 slide joint 固定在水平轨道高度，避免一开始飞出去；
  - 两球之间不再使用 MuJoCo 接触求解器，而是使用一维解析弹性碰撞，避免接触约束注入能量。
"""

from __future__ import annotations

import os
import sys
import time
import shutil
import argparse
import unicodedata
from pathlib import Path
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import mujoco
import mujoco.viewer

try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False
    print("⚠️  未找到 keyboard，请运行: pip install keyboard")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️  未找到 opencv-python，将不会录制 mp4。安装: pip install opencv-python")


# ── 路径 / 基础常量 ───────────────────────────────────────────────────────────
FRANKA_DIR = r"D:\mujoco\mujoco_menagerie\franka_emika_panda"
DEMO_DIR = Path(r"D:\mujoco\demos_collision")

TABLE_H = 0.80                     # 桌子 box 中心高度；桌面 top = TABLE_H + TABLE_THICK
TABLE_THICK = 0.025
TABLE_TOP_Z = TABLE_H + TABLE_THICK
ROBOT_X = -0.25

G = 9.81
BALL_RADIUS = 0.025
BALL_Y = 0.0
BALL_FLAT_Z = TABLE_TOP_Z + BALL_RADIUS

# 斜坡 + 水平轨道坐标。你可以理解为用户描述的 (3,3)->(6,0) 的缩放版。
# 斜坡起点高且靠后，终点接到水平轨道；红蓝球初始位置都在终点右侧的平轨上。
RAMP_TOP_X = 0.04                  # 高处起点，对应“(3,3)”里的 x≈3
RAMP_BOTTOM_X = 0.38               # 低处终点，对应“(6,0)”里的 x≈6
RAMP_HEIGHT = 0.16                 # 斜坡高差，对应“(3,3)”里的 z≈3
RAMP_LEN_X = RAMP_BOTTOM_X - RAMP_TOP_X
RAMP_SLOPE = -RAMP_HEIGHT / RAMP_LEN_X
RAMP_ANGLE_RAD = np.arctan2(RAMP_HEIGHT, RAMP_LEN_X)
# 斜坡只是“发射器”。视觉高度保留，但有效加速度缩放，避免红球从较高处释放后速度过大。
# 若想更接近真实重力斜坡，可把 RAMP_ACCEL_SCALE 调回 1.0；
# 若觉得还是太快，可调到 0.15~0.20。
RAMP_ACCEL_SCALE = 0.25
RAMP_AX = G * np.sin(RAMP_ANGLE_RAD) * np.cos(RAMP_ANGLE_RAD) * RAMP_ACCEL_SCALE  # X 方向加速度

BALL1_X0 = RAMP_BOTTOM_X + 0.10    # 红球初始在平轨上，且 6 < X
TRACK_END_X = 1.35                 # 平轨终点，保证碰撞后有观察空间
TRACK_Y_HALF_WIDTH = 0.055

# 控制和视频
MOVE_SPEED = 0.08
GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 255.0
FPS = 30
VID_W, VID_H = 640, 480
STATUS_INTERVAL = 0.10             # 状态刷新间隔；状态行会原地覆盖，不会刷屏

# 辅助夹持参数
# 关键改动：不再只用 hand 原点判断夹持，而是优先使用左右手指中心/夹爪 site。
# 同时默认 FORCE_ATTACH_ON_CLOSE=True，保证按 7 闭合后红球一定能被“辅助夹住”。
ATTACH_X_TOL = 0.22                # X 方向吸附判断容忍
ATTACH_Z_TOL = 0.20                # Z 方向吸附判断容忍
ATTACH_Y_TOL = 0.18                # Y 方向吸附判断容忍
ATTACH_DIST_3D = 0.30              # 3D 距离吸附判断容忍
FORCE_ATTACH_ON_CLOSE = True       # 按 7 闭合时强制辅助绑定红球，避免夹爪穿模但球不动
ATTACH_Z_OFFSET = -0.005           # 红球中心相对夹爪中心的 z 偏移，可微调视觉效果

# 碰撞稳定化参数
# 核心思路：两球已经被约束成一维运动，所以碰撞不再交给 MuJoCo 接触求解器，
# 而是用一维弹性碰撞公式显式更新速度。这样可以避免“接触求解器 + 手动轨道约束”
# 叠加后给系统注入额外能量，导致两球同时高速弹飞。
USE_ANALYTIC_COLLISION = True
COLLISION_RESTITUTION = 0.98       # 1.0 为完全弹性；0.98 略有耗能，更接近真实碰撞且稳定
COLLISION_GAP = 0.003              # gap <= 该值且 v1>v2 时触发解析碰撞
COLLISION_COOLDOWN_STEPS = 18      # 防止重叠附近连续多次触发碰撞
COLLISION_SEPARATION = 0.0015      # 碰撞后两球表面主动留出的小间隙
MAX_RAMP_SPEED = 0.80              # 红球由斜坡获得的最大速度，防止高处释放过快
MAX_BALL_SPEED = 1.00              # 全局安全限速，防止误操作或数值误差导致球飞出

@dataclass
class JointIds:
    ball1_x_qid: int
    ball1_z_qid: int
    ball1_x_did: int
    ball1_z_did: int
    ball2_x_qid: int
    ball2_x_did: int
    hand_bid: int
    ball1_bid: int
    ball2_bid: int
    left_finger_bid: int = -1
    right_finger_bid: int = -1
    grasp_site_id: int = -1


# ── 路径检查 ─────────────────────────────────────────────────────────────────
def _require_franka_dir() -> None:
    if not os.path.exists(FRANKA_DIR):
        raise FileNotFoundError(
            f"找不到 FRANKA_DIR={FRANKA_DIR}\n"
            "请确认 mujoco_menagerie/franka_emika_panda 路径是否正确，"
            "或者修改脚本顶部的 FRANKA_DIR。"
        )
    for name in ["panda.xml", "scene.xml"]:
        p = os.path.join(FRANKA_DIR, name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少 {p}")


def quat_y(theta: float) -> str:
    """绕 Y 轴旋转 theta 弧度的 MuJoCo quat 字符串。"""
    return f"{np.cos(theta/2):.8f} 0 {np.sin(theta/2):.8f} 0"


def ramp_height_at_x(x: float) -> float:
    """给定世界 x，返回斜坡表面相对水平桌面的高度。"""
    if x <= RAMP_TOP_X:
        return RAMP_HEIGHT
    if x >= RAMP_BOTTOM_X:
        return 0.0
    return RAMP_HEIGHT * (RAMP_BOTTOM_X - x) / RAMP_LEN_X


def ramp_x_from_height(h: float) -> float:
    """给定高度 h，返回斜坡线上对应的 x。"""
    h = float(np.clip(h, 0.0, RAMP_HEIGHT))
    return RAMP_BOTTOM_X - (h / RAMP_HEIGHT) * RAMP_LEN_X


# ── 场景构建 ─────────────────────────────────────────────────────────────────
def build_model(m1: float = 0.10, m2: float = 0.10, distance: float = 0.34):
    """
    构建 Franka + 延长桌面 + 斜坡 + 水平一维轨道 + 两个受约束小球。

    distance 表示蓝球距离红球初始位置的中心距。
    """
    _require_franka_dir()

    ball2_x0 = BALL1_X0 + distance
    if ball2_x0 > TRACK_END_X - 0.08:
        raise ValueError(
            f"distance={distance:.3f} 太大，蓝球初始 x={ball2_x0:.3f} 超出轨道。"
            f"请减小 distance 或增大 TRACK_END_X。"
        )

    # ── 加载 Panda，删除 keyframe，移动基座到桌面高度 ─────────────────────────
    panda_tree = ET.parse(f"{FRANKA_DIR}/panda.xml")
    panda_root = panda_tree.getroot()
    for kf in panda_root.findall("keyframe"):
        panda_root.remove(kf)
    panda_root.find("worldbody").find("body").set("pos", f"{ROBOT_X} 0 {TABLE_H}")

    panda_tmp = f"{FRANKA_DIR}/_panda_collision.xml"
    panda_tree.write(panda_tmp, encoding="unicode", xml_declaration=False)

    # ── 加载 scene.xml，把 include 改为临时 Panda 文件 ────────────────────────
    scene_tree = ET.parse(f"{FRANKA_DIR}/scene.xml")
    scene_root = scene_tree.getroot()
    for inc in scene_root.findall("include"):
        inc.set("file", "_panda_collision.xml")
    for kf in scene_root.findall("keyframe"):
        scene_root.remove(kf)

    option = scene_root.find("option")
    if option is None:
        option = ET.SubElement(scene_root, "option")
    option.set("timestep", "0.002")
    option.set("gravity", "0 0 -9.81")

    wb = scene_root.find("worldbody")

    # ── 延长桌子。几何不参与碰撞，防止和斜坡/球产生不必要穿模弹飞。──────────
    wb.append(ET.fromstring(f"""
    <body name="exp_table" pos="0.45 0 0">
      <geom name="table_top" type="box" size="1.20 0.42 {TABLE_THICK:.4f}"
            pos="0 0 {TABLE_H:.4f}" rgba="0.50 0.36 0.24 1"
            contype="0" conaffinity="0" friction="0 0 0"/>
      <geom type="cylinder" size="0.030 0.39" pos="-0.95 -0.33 0.40" rgba="0.35 0.25 0.18 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.030 0.39" pos=" 0.95 -0.33 0.40" rgba="0.35 0.25 0.18 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.030 0.39" pos="-0.95  0.33 0.40" rgba="0.35 0.25 0.18 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.030 0.39" pos=" 0.95  0.33 0.40" rgba="0.35 0.25 0.18 1" contype="0" conaffinity="0"/>
    </body>
    """))

    # ── 斜坡可视化。用 quat 旋转，避免 compiler angle 单位问题。──────────────
    ramp_dx = RAMP_BOTTOM_X - RAMP_TOP_X
    ramp_dz = -RAMP_HEIGHT
    ramp_length = float(np.sqrt(ramp_dx ** 2 + ramp_dz ** 2))
    surf_mid_x = (RAMP_TOP_X + RAMP_BOTTOM_X) / 2.0
    surf_mid_z = TABLE_TOP_Z + RAMP_HEIGHT / 2.0
    ramp_half_thick = 0.012
    # box 的上表面沿 local z 偏移，中心要沿法向略微下移
    normal_x = np.sin(RAMP_ANGLE_RAD)
    normal_z = np.cos(RAMP_ANGLE_RAD)
    ramp_center_x = surf_mid_x - normal_x * ramp_half_thick
    ramp_center_z = surf_mid_z - normal_z * ramp_half_thick
    ramp_quat = quat_y(RAMP_ANGLE_RAD)

    wb.append(ET.fromstring(f"""
    <body name="ramp_visual" pos="0 0 0">
      <geom name="ramp_surface" type="box"
            size="{ramp_length/2:.4f} {TRACK_Y_HALF_WIDTH:.4f} {ramp_half_thick:.4f}"
            pos="{ramp_center_x:.4f} 0 {ramp_center_z:.4f}"
            quat="{ramp_quat}"
            rgba="0.86 0.80 0.58 0.85"
            contype="0" conaffinity="0" friction="0 0 0"/>
      <geom name="ramp_left_rail" type="capsule" size="0.007"
            fromto="{RAMP_TOP_X:.4f} {TRACK_Y_HALF_WIDTH:.4f} {TABLE_TOP_Z+RAMP_HEIGHT+0.035:.4f} {RAMP_BOTTOM_X:.4f} {TRACK_Y_HALF_WIDTH:.4f} {TABLE_TOP_Z+0.035:.4f}"
            rgba="0.10 0.35 0.85 0.85" contype="0" conaffinity="0"/>
      <geom name="ramp_right_rail" type="capsule" size="0.007"
            fromto="{RAMP_TOP_X:.4f} {-TRACK_Y_HALF_WIDTH:.4f} {TABLE_TOP_Z+RAMP_HEIGHT+0.035:.4f} {RAMP_BOTTOM_X:.4f} {-TRACK_Y_HALF_WIDTH:.4f} {TABLE_TOP_Z+0.035:.4f}"
            rgba="0.10 0.35 0.85 0.85" contype="0" conaffinity="0"/>
      <geom name="ramp_top_marker" type="sphere" size="0.012"
            pos="{RAMP_TOP_X:.4f} 0 {TABLE_TOP_Z+RAMP_HEIGHT+0.025:.4f}"
            rgba="1.0 0.85 0.15 1" contype="0" conaffinity="0"/>
      <geom name="ramp_bottom_marker" type="sphere" size="0.010"
            pos="{RAMP_BOTTOM_X:.4f} 0 {TABLE_TOP_Z+0.025:.4f}"
            rgba="0.15 0.9 0.25 1" contype="0" conaffinity="0"/>
    </body>
    """))

    # ── 水平轨道可视化：从斜坡底部一直延伸到蓝球之后。──────────────────────
    track_center_x = (RAMP_BOTTOM_X + TRACK_END_X) / 2.0
    track_len = TRACK_END_X - RAMP_BOTTOM_X
    rail_z = TABLE_TOP_Z + 0.035
    wb.append(ET.fromstring(f"""
    <body name="horizontal_track" pos="0 0 0">
      <geom name="track_floor" type="box"
            size="{track_len/2:.4f} {TRACK_Y_HALF_WIDTH:.4f} 0.004"
            pos="{track_center_x:.4f} 0 {TABLE_TOP_Z+0.003:.4f}"
            rgba="0.82 0.88 0.92 0.45" contype="0" conaffinity="0" friction="0 0 0"/>
      <geom name="track_left_rail" type="box"
            size="{track_len/2:.4f} 0.006 0.018"
            pos="{track_center_x:.4f} {TRACK_Y_HALF_WIDTH:.4f} {rail_z:.4f}"
            rgba="0.10 0.35 0.85 0.75" contype="0" conaffinity="0"/>
      <geom name="track_right_rail" type="box"
            size="{track_len/2:.4f} 0.006 0.018"
            pos="{track_center_x:.4f} {-TRACK_Y_HALF_WIDTH:.4f} {rail_z:.4f}"
            rgba="0.10 0.35 0.85 0.75" contype="0" conaffinity="0"/>
      <geom name="ball1_start_marker" type="box"
            size="0.004 {TRACK_Y_HALF_WIDTH:.4f} 0.018"
            pos="{BALL1_X0:.4f} 0 {rail_z+0.008:.4f}"
            rgba="0.95 0.25 0.15 0.70" contype="0" conaffinity="0"/>
      <geom name="ball2_start_marker" type="box"
            size="0.004 {TRACK_Y_HALF_WIDTH:.4f} 0.018"
            pos="{ball2_x0:.4f} 0 {rail_z+0.008:.4f}"
            rgba="0.15 0.75 0.25 0.70" contype="0" conaffinity="0"/>
    </body>
    """))

    # ── 两球。ball1 有 x/z 两个 slide；ball2 只有 x slide。──────────────────
    # 注意：这里关闭两球的 MuJoCo 接触，由 resolve_analytic_collision() 统一处理一维碰撞。
    # 原因是红球还受“斜坡/水平轨道手动约束”，如果再叠加 MuJoCo 接触求解，容易注入能量。
    b1_x_min = RAMP_TOP_X - BALL1_X0
    b1_x_max = TRACK_END_X - BALL1_X0
    b2_x_min = -0.04
    b2_x_max = TRACK_END_X - ball2_x0
    z_max = RAMP_HEIGHT + 0.12

    balls_xml = f"""
    <body name="ball1_body" pos="{BALL1_X0:.4f} {BALL_Y:.4f} {BALL_FLAT_Z:.4f}">
      <joint name="ball1_x" type="slide" axis="1 0 0"
             limited="true" range="{b1_x_min:.4f} {b1_x_max:.4f}"
             damping="0" frictionloss="0"/>
      <joint name="ball1_z" type="slide" axis="0 0 1"
             limited="true" range="0 {z_max:.4f}"
             damping="0" frictionloss="0"/>
      <geom name="ball1_geom" type="sphere" size="{BALL_RADIUS:.4f}"
            rgba="0.90 0.22 0.10 1" mass="{m1:.6f}"
            contype="0" conaffinity="0" condim="1"
            friction="0 0 0"/>
      <site name="ball1_site" pos="0 0 0" size="0.006" rgba="1.0 1.0 0.0 0.9"/>
    </body>

    <body name="ball2_body" pos="{ball2_x0:.4f} {BALL_Y:.4f} {BALL_FLAT_Z:.4f}">
      <joint name="ball2_x" type="slide" axis="1 0 0"
             limited="true" range="{b2_x_min:.4f} {b2_x_max:.4f}"
             damping="0" frictionloss="0"/>
      <geom name="ball2_geom" type="sphere" size="{BALL_RADIUS:.4f}"
            rgba="0.10 0.45 0.95 1" mass="{m2:.6f}"
            contype="0" conaffinity="0" condim="1"
            friction="0 0 0"/>
      <site name="ball2_site" pos="0 0 0" size="0.006" rgba="1.0 1.0 0.0 0.9"/>
    </body>
    """
    tmp_root = ET.fromstring(f"<root>{balls_xml}</root>")
    for child in list(tmp_root):
        wb.append(child)

    # ── 相机 ─────────────────────────────────────────────────────────────────
    cx = (ROBOT_X + ball2_x0) / 2.0
    cams = [
        f'<camera name="cam_side" pos="1.45 -1.35 1.65" xyaxes="0.70 0.70 0 -0.35 0.35 0.87" fovy="55"/>',
        f'<camera name="cam_front" pos="{cx:.4f} -1.70 1.45" xyaxes="1 0 0 0 0.42 0.91" fovy="52"/>',
        f'<camera name="cam_top" pos="{cx:.4f} 0 2.55" xyaxes="1 0 0 0 1 0" fovy="55"/>',
    ]
    for cam in cams:
        wb.append(ET.fromstring(cam))

    # ── 传感器 ───────────────────────────────────────────────────────────────
    sensor = scene_root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(scene_root, "sensor")
    for s in [
        '<jointpos name="ball1_x_pos" joint="ball1_x"/>',
        '<jointvel name="ball1_x_vel" joint="ball1_x"/>',
        '<jointpos name="ball1_z_pos" joint="ball1_z"/>',
        '<jointvel name="ball1_z_vel" joint="ball1_z"/>',
        '<jointpos name="ball2_x_pos" joint="ball2_x"/>',
        '<jointvel name="ball2_x_vel" joint="ball2_x"/>',
    ]:
        sensor.append(ET.fromstring(s))

    scene_tmp = f"{FRANKA_DIR}/_scene_collision.xml"
    scene_tree.write(scene_tmp, encoding="unicode", xml_declaration=False)
    try:
        m = mujoco.MjModel.from_xml_path(scene_tmp)
    finally:
        if os.path.exists(scene_tmp):
            os.remove(scene_tmp)
        if os.path.exists(panda_tmp):
            os.remove(panda_tmp)

    geom_info = {
        "ball2_x0": ball2_x0,
        "ramp_top_x": RAMP_TOP_X,
        "ramp_bottom_x": RAMP_BOTTOM_X,
        "ramp_height": RAMP_HEIGHT,
        "ramp_angle_deg": float(np.degrees(RAMP_ANGLE_RAD)),
        "track_end_x": TRACK_END_X,
    }
    return m, mujoco.MjData(m), geom_info


# ── MuJoCo ID 工具 ───────────────────────────────────────────────────────────
def jid(model, name: str) -> int:
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if i < 0:
        raise KeyError(f"找不到 joint: {name}")
    return i


def bid(model, name: str) -> int:
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if i < 0:
        raise KeyError(f"找不到 body: {name}")
    return i


def maybe_body_id(model, names) -> int:
    if isinstance(names, str):
        names = [names]
    for name in names:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i >= 0:
            return int(i)
    return -1


def maybe_site_id(model, names) -> int:
    if isinstance(names, str):
        names = [names]
    for name in names:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if i >= 0:
            return int(i)
    return -1


def collect_ids(model) -> JointIds:
    j1x = jid(model, "ball1_x")
    j1z = jid(model, "ball1_z")
    j2x = jid(model, "ball2_x")
    return JointIds(
        ball1_x_qid=int(model.jnt_qposadr[j1x]),
        ball1_z_qid=int(model.jnt_qposadr[j1z]),
        ball1_x_did=int(model.jnt_dofadr[j1x]),
        ball1_z_did=int(model.jnt_dofadr[j1z]),
        ball2_x_qid=int(model.jnt_qposadr[j2x]),
        ball2_x_did=int(model.jnt_dofadr[j2x]),
        hand_bid=bid(model, "hand"),
        ball1_bid=bid(model, "ball1_body"),
        ball2_bid=bid(model, "ball2_body"),
        left_finger_bid=maybe_body_id(model, ["left_finger", "panda_leftfinger", "leftfinger", "finger_left"]),
        right_finger_bid=maybe_body_id(model, ["right_finger", "panda_rightfinger", "rightfinger", "finger_right"]),
        grasp_site_id=maybe_site_id(model, ["gripper", "pinch", "ee_site", "attachment_site"]),
    )


# ── IK 和键盘控制 ────────────────────────────────────────────────────────────
def ik_step(model, data, target: np.ndarray):
    hand_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    err = target - data.xpos[hand_bid]
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, hand_bid)
    J = jacp[:, :7]
    dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
    dq = np.clip(dq * 400.0 * model.opt.timestep, -0.3, 0.3)
    data.ctrl[:7] = np.clip(data.qpos[:7] + dq, model.jnt_range[:7, 0], model.jnt_range[:7, 1])


def poll_movement(dt: float) -> np.ndarray:
    if not HAS_KEYBOARD:
        return np.zeros(3)
    delta = np.zeros(3)
    spd = MOVE_SPEED * dt
    if keyboard.is_pressed('1'):
        delta[0] += spd
    if keyboard.is_pressed('2'):
        delta[0] -= spd
    if keyboard.is_pressed('3'):
        delta[1] -= spd
    if keyboard.is_pressed('4'):
        delta[1] += spd
    if keyboard.is_pressed('5'):
        delta[2] += spd
    if keyboard.is_pressed('6'):
        delta[2] -= spd
    return delta


# ── 小球状态与约束 ───────────────────────────────────────────────────────────
def ball1_world(data, ids: JointIds) -> tuple[float, float, float, float]:
    x = BALL1_X0 + float(data.qpos[ids.ball1_x_qid])
    z = BALL_FLAT_Z + float(data.qpos[ids.ball1_z_qid])
    vx = float(data.qvel[ids.ball1_x_did])
    vz = float(data.qvel[ids.ball1_z_did])
    return x, z, vx, vz


def ball2_world(data, ids: JointIds, ball2_x0: float) -> tuple[float, float]:
    x = ball2_x0 + float(data.qpos[ids.ball2_x_qid])
    vx = float(data.qvel[ids.ball2_x_did])
    return x, vx


def set_ball2_world(data, ids: JointIds, ball2_x0: float, x: float, vx: float = 0.0):
    x_rel = np.clip(x - ball2_x0, -0.05, TRACK_END_X - ball2_x0)
    data.qpos[ids.ball2_x_qid] = x_rel
    data.qvel[ids.ball2_x_did] = float(np.clip(vx, -MAX_BALL_SPEED, MAX_BALL_SPEED))


def set_ball1_world(data, ids: JointIds, x: float, z: float, vx: float = 0.0, vz: float = 0.0):
    x_rel = np.clip(x - BALL1_X0, RAMP_TOP_X - BALL1_X0, TRACK_END_X - BALL1_X0)
    z_rel = np.clip(z - BALL_FLAT_Z, 0.0, RAMP_HEIGHT + 0.12)
    data.qpos[ids.ball1_x_qid] = x_rel
    data.qpos[ids.ball1_z_qid] = z_rel
    data.qvel[ids.ball1_x_did] = float(np.clip(vx, -MAX_BALL_SPEED, MAX_BALL_SPEED))
    data.qvel[ids.ball1_z_did] = float(np.clip(vz, -MAX_BALL_SPEED, MAX_BALL_SPEED))


def clamp_ball2(data, ids: JointIds, ball2_x0: float):
    # 保证蓝球不会跑出可视轨道，也不会出现极端速度刷屏。
    x2, v2 = ball2_world(data, ids, ball2_x0)
    if x2 > TRACK_END_X:
        set_ball2_world(data, ids, ball2_x0, TRACK_END_X, 0.0)
    elif x2 < ball2_x0 - 0.05:
        set_ball2_world(data, ids, ball2_x0, ball2_x0 - 0.05, 0.0)
    else:
        # 仅做安全限速，不改变正常碰撞速度。
        data.qvel[ids.ball2_x_did] = float(np.clip(v2, -MAX_BALL_SPEED, MAX_BALL_SPEED))


def resolve_analytic_collision(
    model,
    data,
    ids: JointIds,
    m1: float,
    m2: float,
    ball2_x0: float,
    step_count: int,
    last_collision_step: int,
) -> tuple[int, bool]:
    """解析处理红蓝球一维弹性碰撞。

    为什么不用 MuJoCo contact：红球在斜坡段和水平段由代码手动约束；
    若再叠加 MuJoCo 接触求解器，接触约束与手动约束会相互“打架”，
    容易出现两球碰后速度异常放大的现象。这里用一维碰撞公式显式更新速度，
    可以保证动量守恒，且 restitution<=1 时不会凭空增加动能。
    """
    if step_count - last_collision_step < COLLISION_COOLDOWN_STEPS:
        return last_collision_step, False

    x1, z1, v1, _ = ball1_world(data, ids)
    x2, v2 = ball2_world(data, ids, ball2_x0)

    # 只在红球进入水平轨道后处理碰撞；斜坡段不会碰到蓝球。
    if x1 < RAMP_BOTTOM_X - 0.003 or abs(z1 - BALL_FLAT_Z) > 0.010:
        return last_collision_step, False

    gap = x2 - x1 - 2.0 * BALL_RADIUS
    closing_speed = v1 - v2
    if gap > COLLISION_GAP or closing_speed <= 0.0:
        return last_collision_step, False

    denom = m1 + m2
    e = COLLISION_RESTITUTION
    # 一维恢复系数碰撞公式：e=1 时完全弹性；e<1 时略有能量损失，更稳定也更真实。
    v1_new = ((m1 - e * m2) / denom) * v1 + ((1.0 + e) * m2 / denom) * v2
    v2_new = ((1.0 + e) * m1 / denom) * v1 + ((m2 - e * m1) / denom) * v2
    v1_new = float(np.clip(v1_new, -MAX_BALL_SPEED, MAX_BALL_SPEED))
    v2_new = float(np.clip(v2_new, -MAX_BALL_SPEED, MAX_BALL_SPEED))

    # 位置修正：把两个球从微小重叠状态分开，防止下一帧重复碰撞。
    overlap = max(0.0, 2.0 * BALL_RADIUS - (x2 - x1) + COLLISION_SEPARATION)
    if overlap > 0:
        # 按质量反比分配修正量；质量小的球位移稍大。
        x1 -= overlap * (m2 / denom)
        x2 += overlap * (m1 / denom)

    set_ball1_world(data, ids, x1, BALL_FLAT_Z, v1_new, 0.0)
    set_ball2_world(data, ids, ball2_x0, x2, v2_new)
    mujoco.mj_forward(model, data)
    return step_count, True


def snap_release_to_ramp(data, ids: JointIds):
    """松爪时将红球吸附到斜坡线或水平轨道，避免空中掉落/飞出。"""
    x, z, vx, _ = ball1_world(data, ids)
    h = max(0.0, z - BALL_FLAT_Z)

    # 如果红球处于较高位置，优先投影到斜坡线上。
    if h > 0.01 or x < RAMP_BOTTOM_X:
        if x < RAMP_TOP_X or x > RAMP_BOTTOM_X:
            x = ramp_x_from_height(h)
        x = float(np.clip(x, RAMP_TOP_X, RAMP_BOTTOM_X - 0.002))
        h_on_ramp = ramp_height_at_x(x)
        z = BALL_FLAT_Z + h_on_ramp
        vx = max(0.0, vx)
        vz = RAMP_SLOPE * vx
        set_ball1_world(data, ids, x, z, vx, vz)
    else:
        set_ball1_world(data, ids, x, BALL_FLAT_Z, max(0.0, vx), 0.0)


def apply_ramp_constraint(model, data, ids: JointIds, attached: bool, dt: float):
    """
    红球未被夹住时：
      - 在斜坡区间内，约束到斜坡线，并施加无摩擦斜坡的 X 方向加速度；
      - 到达斜坡底部后，约束到水平轨道高度；
      - 不允许 Y 方向运动，因为模型本身没有 Y 自由度。
    """
    if attached:
        return

    x, z, vx, vz = ball1_world(data, ids)

    if x < RAMP_BOTTOM_X - 0.001:
        x = float(np.clip(x, RAMP_TOP_X, RAMP_BOTTOM_X - 0.001))
        h = ramp_height_at_x(x)
        vx = min(MAX_RAMP_SPEED, max(0.0, vx + RAMP_AX * dt))
        vz = RAMP_SLOPE * vx
        set_ball1_world(data, ids, x, BALL_FLAT_Z + h, vx, vz)
        mujoco.mj_forward(model, data)
    else:
        # 水平轨道段：红球只沿 X 运动，Z 固定在桌面高度。
        set_ball1_world(data, ids, x, BALL_FLAT_Z, vx, 0.0)
        mujoco.mj_forward(model, data)


def get_grasp_center(model, data, ids: JointIds) -> np.ndarray:
    """返回用于辅助夹持的夹爪中心。

    优先级：
      1. gripper / pinch / ee_site 等 site；
      2. left_finger 与 right_finger 两个 body 的中点；
      3. hand body 坐标。
    """
    if ids.grasp_site_id >= 0:
        return np.asarray(data.site_xpos[ids.grasp_site_id], dtype=float).copy()
    if ids.left_finger_bid >= 0 and ids.right_finger_bid >= 0:
        return ((np.asarray(data.xpos[ids.left_finger_bid], dtype=float)
                 + np.asarray(data.xpos[ids.right_finger_bid], dtype=float)) / 2.0).copy()
    return np.asarray(data.xpos[ids.hand_bid], dtype=float).copy()


def maybe_attach_ball(model, data, ids: JointIds, ee_target: np.ndarray | None = None) -> tuple[bool, float]:
    """判断是否可以辅助夹住红球，返回 (can_attach, distance)。

    同时考虑真实夹爪中心和 ee_target，避免 IK 滞后导致误判。
    如果 FORCE_ATTACH_ON_CLOSE=True，按 7 闭合时即使判断不准也会辅助绑定红球。
    """
    center = get_grasp_center(model, data, ids)
    x, z, _, _ = ball1_world(data, ids)
    ball = np.array([x, BALL_Y, z], dtype=float)

    candidates = [center]
    if ee_target is not None:
        candidates.append(np.array([ee_target[0], BALL_Y, ee_target[2]], dtype=float))

    best_dist = float("inf")
    ok = False
    for c in candidates:
        diff = c - ball
        dist = float(np.linalg.norm(diff))
        best_dist = min(best_dist, dist)
        ok_box = (abs(diff[0]) <= ATTACH_X_TOL and
                  abs(diff[1]) <= ATTACH_Y_TOL and
                  abs(diff[2]) <= ATTACH_Z_TOL)
        ok_sphere = dist <= ATTACH_DIST_3D
        ok = ok or ok_box or ok_sphere
    return bool(ok), best_dist


def update_attached_ball(model, data, ids: JointIds):
    """辅助夹持时，让红球跟随夹爪中心的 X/Z，不改变 Y。

    这里不依赖 MuJoCo 真实接触摩擦，而是显式把红球绑定到夹爪中心。
    这样既保留按键 7 的夹取/释放动作，又避免球体和夹爪穿模后无法移动。
    """
    center = get_grasp_center(model, data, ids)
    x = float(np.clip(center[0], RAMP_TOP_X, TRACK_END_X - 0.05))
    z = float(np.clip(center[2] + ATTACH_Z_OFFSET, BALL_FLAT_Z, BALL_FLAT_Z + RAMP_HEIGHT + 0.10))
    set_ball1_world(data, ids, x, z, 0.0, 0.0)
    mujoco.mj_forward(model, data)


def current_quantities(data, ids: JointIds, m1: float, m2: float, ball2_x0: float):
    x1, z1, v1, _ = ball1_world(data, ids)
    x2, v2 = ball2_world(data, ids, ball2_x0)
    gap = x2 - x1 - 2 * BALL_RADIUS
    p = m1 * v1 + m2 * v2
    e = 0.5 * m1 * v1 * v1 + 0.5 * m2 * v2 * v2
    h = max(0.0, z1 - BALL_FLAT_Z)
    v_theory = np.sqrt(max(0.0, 2 * G * RAMP_ACCEL_SCALE * h))
    return x1, z1, x2, v1, v2, gap, p, e, h, v_theory


def make_state_row(data, ids: JointIds, v1: float, v2: float):
    finger1 = float(data.qpos[7]) if data.qpos.shape[0] > 7 else 0.0
    finger2 = float(data.qpos[8]) if data.qpos.shape[0] > 8 else 0.0
    return [finger1, finger2] + data.qpos[:7].tolist() + [float(v1), float(v2), float(data.time)]


# ── 主录制流程 ───────────────────────────────────────────────────────────────
def record_one(
    m1: float,
    m2: float,
    v_target: float,
    distance: float,
    save_path: Path,
    video_path: Path,
    status_interval: float = STATUS_INTERVAL,
):
    print("\n" + "=" * 76)
    print("  实验04：弹性碰撞示教录制 —— 稳定斜坡发射版")
    print(f"  m1={m1:.3f} kg  m2={m2:.3f} kg  目标碰前速度 v≈{v_target:.3f} m/s")
    print(f"  红球初始在水平轨道 x={BALL1_X0:.3f}，蓝球中心距 distance={distance:.3f}")
    print(f"  斜坡：top=({RAMP_TOP_X:.3f}, h={RAMP_HEIGHT:.3f}) → bottom=({RAMP_BOTTOM_X:.3f}, h=0)，有效加速度缩放={RAMP_ACCEL_SCALE:.2f}")
    print("  约束：红球仅 X/Z；蓝球仅 X；碰撞使用一维解析弹性公式，避免数值弹飞")
    print(f"  保存：{save_path}")
    print("=" * 76)
    print("  1/2  +X/-X   3/4  -Y/+Y   5/6  +Z/-Z")
    print("  7    夹爪开合：闭合时辅助夹住红球；打开时释放到斜坡/水平轨道")
    print("  8    开始 / 停止录制")
    print("  9    保存退出   ESC 放弃退出")
    print("=" * 76)
    print("\n推荐流程：")
    print("  ① 两球初始都在水平轨道上；")
    print("  ② 移动夹爪到红球附近，按 7；若判断不准也会辅助吸附，终端出现 ATTACHED；")
    print("  ③ 用 2/5 把红球拉回斜坡线上某个高度；")
    print("  ④ 按 8 开始录制；")
    print("  ⑤ 按 7 松爪，红球沿斜坡下滑到平轨并撞蓝球；")
    print("  ⑥ 碰撞后观察一小段，按 8 停止，再按 9 保存。\n")

    model, data, geom_info = build_model(m1=m1, m2=m2, distance=distance)
    ids = collect_ids(model)
    ball2_x0 = geom_info["ball2_x0"]

    mujoco.mj_resetData(model, data)
    home_q = np.array([0.0, 0.5, 0.0, -2.0, 0.0, 2.5, 0.785])
    data.qpos[:7] = home_q.copy()
    data.ctrl[:7] = home_q.copy()
    if model.nu > 7:
        data.ctrl[7] = GRIPPER_OPEN
    # 确保两球初始稳定在水平轨道上
    set_ball1_world(data, ids, BALL1_X0, BALL_FLAT_Z, 0.0, 0.0)
    data.qpos[ids.ball2_x_qid] = 0.0
    data.qvel[ids.ball2_x_did] = 0.0
    mujoco.mj_forward(model, data)

    ee_target = data.xpos[ids.hand_bid].copy()
    gripper_val = GRIPPER_OPEN
    attached = False
    is_recording = False
    saved = False

    states = []
    actions = []
    metrics_history = []

    renderer = mujoco.Renderer(model, VID_H, VID_W)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_side")
    vw = None
    record_every = max(1, int(1.0 / (FPS * model.opt.timestep)))

    prev_time = time.perf_counter()
    prev_7 = prev_8 = prev_9 = prev_esc = False
    step_count = 0
    last_status_t = 0.0
    max_v1 = 0.0
    max_v2 = 0.0
    last_collision_step = -10_000
    last_collision_t = -10_000.0
    status_line_len = 0

    def _char_width(ch: str) -> int:
        """估计字符在终端中的显示宽度：中文/全角字符通常占 2 列。"""
        if unicodedata.east_asian_width(ch) in ("F", "W", "A"):
            return 2
        return 1

    def _display_width(text: str) -> int:
        return sum(_char_width(ch) for ch in text)

    def _truncate_to_columns(text: str, max_cols: int) -> str:
        """按终端显示列宽截断，避免 PowerShell 自动换行。"""
        if max_cols <= 0:
            return ""
        out = []
        used = 0
        for ch in text:
            w = _char_width(ch)
            if used + w > max_cols:
                break
            out.append(ch)
            used += w
        return "".join(out)

    def clear_status_line():
        """清空当前终端行。

        不能只按上一条 msg 的 len 清空，因为中文字符在 PowerShell 里通常占 2 列，
        len(msg) 小于实际显示宽度，会留下 v理≈0.65 这类尾巴。
        """
        nonlocal status_line_len
        cols = shutil.get_terminal_size((120, 20)).columns
        sys.stdout.write("\r" + " " * max(1, cols - 1) + "\r")
        sys.stdout.flush()
        status_line_len = 0

    def event_print(msg: str):
        clear_status_line()
        print(msg, flush=True)

    def write_status(msg: str):
        nonlocal status_line_len
        cols = shutil.get_terminal_size((120, 20)).columns
        # 按显示宽度截断，而不是按 Python 字符数截断。中文字符会占 2 列。
        max_cols = max(30, cols - 2)
        msg = _truncate_to_columns(msg, max_cols)
        width = _display_width(msg)
        # 再补空格到行尾，确保比上一条更短时也不会残留尾巴。
        pad = " " * max(0, cols - 1 - width)
        sys.stdout.write("\r" + msg + pad + "\r" + msg)
        sys.stdout.flush()
        status_line_len = width

    def key_callback(keycode):
        # keyboard 不可用时的备用单步控制。
        nonlocal ee_target
        if HAS_KEYBOARD:
            return
        step = 0.02
        mapping = {
            ord('1'): np.array([ step, 0, 0]),
            ord('2'): np.array([-step, 0, 0]),
            ord('3'): np.array([0, -step, 0]),
            ord('4'): np.array([0,  step, 0]),
            ord('5'): np.array([0, 0,  step]),
            ord('6'): np.array([0, 0, -step]),
        }
        if keycode in mapping:
            ee_target = ee_target + mapping[keycode]

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.distance = 2.6
        viewer.cam.elevation = -22
        viewer.cam.azimuth = 145

        while viewer.is_running() and not saved:
            now = time.perf_counter()
            dt = min(now - prev_time, 0.05)
            prev_time = now

            ee_target += poll_movement(dt)

            cur_esc = HAS_KEYBOARD and keyboard.is_pressed('esc')
            if cur_esc and not prev_esc:
                event_print("❌ 放弃退出，未保存。")
                break
            prev_esc = cur_esc

            # 7：夹爪开合 + 辅助夹持/释放
            cur_7 = HAS_KEYBOARD and keyboard.is_pressed('7')
            if cur_7 and not prev_7:
                if gripper_val == GRIPPER_OPEN:
                    gripper_val = GRIPPER_CLOSE
                    can_attach, attach_dist = maybe_attach_ball(model, data, ids, ee_target=ee_target)
                    if can_attach or FORCE_ATTACH_ON_CLOSE:
                        attached = True
                        update_attached_ball(model, data, ids)
                        if can_attach:
                            event_print(f"夹爪: 闭合，红球 ATTACHED（距离={attach_dist:.3f}m）")
                        else:
                            event_print(f"夹爪: 闭合，红球 ATTACHED（辅助吸附，距离={attach_dist:.3f}m）")
                    else:
                        attached = False
                        event_print(f"夹爪: 闭合，但距离红球较远（距离={attach_dist:.3f}m），未夹住")
                else:
                    gripper_val = GRIPPER_OPEN
                    if attached:
                        attached = False
                        snap_release_to_ramp(data, ids)
                        mujoco.mj_forward(model, data)
                        event_print("夹爪: 打开，红球 RELEASED")
                    else:
                        event_print("夹爪: 打开")
            prev_7 = cur_7

            # 如果夹住期间移动夹爪，红球跟随 X/Z。
            if attached:
                update_attached_ball(model, data, ids)

            # 8：录制开关
            cur_8 = HAS_KEYBOARD and keyboard.is_pressed('8')
            if cur_8 and not prev_8:
                is_recording = not is_recording
                if is_recording:
                    states.clear()
                    actions.clear()
                    metrics_history.clear()
                    max_v1 = max_v2 = 0.0
                    if HAS_CV2:
                        vw = cv2.VideoWriter(
                            str(video_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            FPS,
                            (VID_W, VID_H),
                        )
                    event_print("▶ 开始录制...")
                else:
                    if vw:
                        vw.release()
                        vw = None
                    event_print(f"■ 停止录制，已录 {len(states)} 步")
            prev_8 = cur_8

            # 9：保存退出
            cur_9 = HAS_KEYBOARD and keyboard.is_pressed('9')
            if cur_9 and not prev_9:
                if len(states) > 100:
                    if vw:
                        vw.release()
                        vw = None
                    np.savez(
                        str(save_path),
                        states=np.asarray(states, dtype=np.float32),
                        actions=np.asarray(actions, dtype=np.float32),
                        metrics=np.asarray(metrics_history, dtype=np.float32),
                        m1=np.array([m1], dtype=np.float32),
                        m2=np.array([m2], dtype=np.float32),
                        v_target=np.array([v_target], dtype=np.float32),
                        distance=np.array([distance], dtype=np.float32),
                        ball_radius=np.array([BALL_RADIUS], dtype=np.float32),
                        ball1_x0=np.array([BALL1_X0], dtype=np.float32),
                        ball2_x0=np.array([ball2_x0], dtype=np.float32),
                        ramp_top_x=np.array([RAMP_TOP_X], dtype=np.float32),
                        ramp_bottom_x=np.array([RAMP_BOTTOM_X], dtype=np.float32),
                        ramp_height=np.array([RAMP_HEIGHT], dtype=np.float32),
                        ramp_angle_deg=np.array([np.degrees(RAMP_ANGLE_RAD)], dtype=np.float32),
                        collision_restitution=np.array([COLLISION_RESTITUTION], dtype=np.float32),
                        analytic_collision=np.array([1 if USE_ANALYTIC_COLLISION else 0], dtype=np.int32),
                        max_ramp_speed=np.array([MAX_RAMP_SPEED], dtype=np.float32),
                        max_ball_speed=np.array([MAX_BALL_SPEED], dtype=np.float32),
                    )
                    event_print(f"✅ 已保存 {len(states)} 步 → {save_path}")
                    if HAS_CV2:
                        event_print(f"✅ 视频 → {video_path}")
                    saved = True
                else:
                    event_print(f"步数不足（{len(states)} 步），请先按 8 录制。")
            prev_9 = cur_9

            # ── 控制机械臂 ────────────────────────────────────────────────────
            ik_step(model, data, ee_target)
            if model.nu > 7:
                data.ctrl[7] = gripper_val

            mujoco.mj_step(model, data)
            step_count += 1

            # 步进后应用斜坡/水平轨道约束，避免红球飞出/穿模。
            if attached:
                update_attached_ball(model, data, ids)
            else:
                apply_ramp_constraint(model, data, ids, attached=False, dt=model.opt.timestep)
                last_collision_step, collided = resolve_analytic_collision(
                    model, data, ids, m1, m2, ball2_x0, step_count, last_collision_step
                )
                if collided:
                    last_collision_t = now
            clamp_ball2(data, ids, ball2_x0)
            mujoco.mj_forward(model, data)

            x1, z1, x2, v1, v2, gap, p, e, h, v_theory = current_quantities(data, ids, m1, m2, ball2_x0)
            max_v1 = max(max_v1, abs(v1))
            max_v2 = max(max_v2, abs(v2))

            if is_recording:
                states.append(make_state_row(data, ids, v1, v2))
                actions.append(data.ctrl[:8].tolist())
                # metrics: [x1, x2, v1, v2, P, E, gap, z1, h]
                metrics_history.append([x1, x2, v1, v2, p, e, gap, z1, h])

                if HAS_CV2 and vw and step_count % record_every == 0:
                    renderer.update_scene(data, camera=cam_id)
                    vw.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

            viewer.sync()

            # 终端输出降频，避免每步刷屏。
            if now - last_status_t >= status_interval:
                last_status_t = now
                attach_flag = "ATTACHED" if attached else "free"
                rec_flag = "REC" if is_recording else "idle"
                col_flag = "COL" if (now - last_collision_t) < 0.7 else ""
                gc = get_grasp_center(model, data, ids)
                msg = (
                    f"夹={'闭' if gripper_val == GRIPPER_CLOSE else '开'} {attach_flag} {rec_flag} {col_flag} {len(states)}步 | "
                    f"g=({gc[0]:.2f},{gc[2]:.2f}) x1={x1:.2f} z1={z1:.2f} x2={x2:.2f} "
                    f"gap={gap:.2f} v1={v1:+.2f} v2={v2:+.2f} h={h:.2f} v理≈{v_theory:.2f}"
                )
                write_status(msg)

    try:
        clear_status_line()
    except Exception:
        pass
    renderer.close()
    if vw:
        vw.release()
    return saved


# ── 主入口 ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="实验04：弹性碰撞示教录制（稳定斜坡发射版）")
    parser.add_argument("--m1", type=float, default=0.10, help="红球质量 kg")
    parser.add_argument("--m2", type=float, default=0.10, help="蓝球质量 kg")
    parser.add_argument("--v", type=float, default=0.30, help="目标碰前速度，仅作元数据记录")
    parser.add_argument("--distance", type=float, default=0.34, help="红蓝球初始中心距")
    parser.add_argument("--status-interval", type=float, default=STATUS_INTERVAL, help="终端状态输出间隔秒")
    args = parser.parse_args()

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    save_path = DEMO_DIR / (
        f"collision_ramp_stable_m1{args.m1:.2f}_m2{args.m2:.2f}_v{args.v:.2f}_d{args.distance:.2f}.npz"
    )
    video_path = DEMO_DIR / (
        f"collision_ramp_stable_m1{args.m1:.2f}_m2{args.m2:.2f}_v{args.v:.2f}_d{args.distance:.2f}.mp4"
    )

    if save_path.exists():
        print(f"⚠️  已存在：{save_path}")
        ans = input("   覆盖重录？(y/N): ").strip().lower()
        if ans != "y":
            print("取消。")
            return

    ok = record_one(
        m1=args.m1,
        m2=args.m2,
        v_target=args.v,
        distance=args.distance,
        save_path=save_path,
        video_path=video_path,
        status_interval=args.status_interval,
    )
    print("\n✅ 录制成功！" if ok else "\n❌ 文件未保存。")


if __name__ == "__main__":
    main()
