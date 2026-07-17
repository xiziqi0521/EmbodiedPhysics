"""
teleop_buoyancy_lerobot.py —— 浮力实验示教录制（lerobot v3.0 格式）
=====================================================================
实验 07：阿基米德原理验证
    机械臂抓取漂浮在水面的方块，缓慢将其沉入水中

物理模拟：
    MuJoCo 不支持流体，通过在每步施加浮力来模拟：
    F_buoy = ρ_water * g * V_submerged（向上）
    V_submerged = block_size^3 * clamp(submerged_ratio, 0, 1)

state 14维：[j0~j6, gripper, block_x, block_y, block_z, ee_x, ee_y, ee_z]
action  8维：ctrl[:8]
三路相机：cam_front / cam_side / cam_top，96×96

用法：
    python teleop_buoyancy_lerobot.py --n 20 --out D:/mujuco/demos_buoyancy

按键：
    1/2 前/后  3/4 左/右  5/6 上/下
    7   夹爪开合
    8   开始/停止录制
    9   保存当前条
    0   放弃当前条
    ESC 退出
"""

import os, sys, time, argparse, json
import numpy as np
import mujoco, mujoco.viewer
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import imageio

try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False
    print("pip install keyboard")

# ── 常量 ──────────────────────────────────────────────────────────────────────
FRANKA_DIR    = r"D:\mujuco\mujoco_menagerie\franka_emika_panda"
TABLE_H       = 0.80
ROBOT_X       = -0.25
TANK_X        = 0.35       # 水槽中心 x
TANK_Y        = 0.00
TANK_W        = 0.28       # 水槽宽度（x方向）
TANK_D        = 0.22       # 水槽深度（y方向）
TANK_H        = 0.20       # 水槽高度
WATER_DEPTH   = 0.16       # 水深
BALL_RADIUS   = 0.025      # 球半径（m），直径5cm
BLOCK_DENSITY = 150.0      # 球密度（kg/m³），完全沉没净向上力约0.55N
WATER_DENSITY = 1000.0     # 水密度（kg/m³）
G             = 9.81

MOVE_SPEED    = 0.04       # 移动速度（慢一些方便沉水）
GRIPPER_OPEN  = 0.0
GRIPPER_CLOSE = 255.0
IMG_SIZE      = 96
MIN_STEPS     = 100
FPS           = 15
STATE_DIM     = 14
ACTION_DIM    = 8
TASK_STR      = 'grasp floating block and slowly submerge it into water'


# ── 浮力计算 ──────────────────────────────────────────────────────────────────
def compute_buoyancy(ball_z, water_surface_z, ball_radius, ball_density, water_density):
    """计算球体所受浮力（N，向上为正）- 用球冠体积近似"""
    import math
    r = ball_radius
    h = float(np.clip(water_surface_z - (ball_z - r), 0, 2*r))
    # 球冠体积 V = pi*h^2*(3r-h)/3
    v_sub = math.pi * h**2 * (3*r - h) / 3
    f_buoy = water_density * G * v_sub
    return f_buoy


# ── lerobot v3.0 写入器（与 teleop_arc_lerobot.py 相同）────────────────────
class LeRobotWriter:
    def __init__(self, out_dir):
        self.root = Path(out_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'meta' / 'episodes' / 'chunk-000').mkdir(parents=True, exist_ok=True)
        (self.root / 'data' / 'chunk-000').mkdir(parents=True, exist_ok=True)
        for cam in ['cam_front', 'cam_side', 'cam_top']:
            (self.root / 'videos' / f'observation.images.{cam}' / 'chunk-000').mkdir(parents=True, exist_ok=True)

        self.eps_parquet = self.root / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet'
        self.episodes_meta = []
        self.global_index  = 0

        if self.eps_parquet.exists():
            df = pd.read_parquet(self.eps_parquet)
            self.episodes_meta = df.to_dict('records')
            data_files = sorted((self.root / 'data' / 'chunk-000').glob('file-*.parquet'))
            if data_files:
                last_df = pd.read_parquet(data_files[-1])
                self.global_index = int(last_df['index'].iloc[-1]) + 1

        self.all_actions = []
        self.all_states  = []
        for f in sorted((self.root / 'data' / 'chunk-000').glob('file-*.parquet')):
            df = pd.read_parquet(f)
            self.all_actions.append(np.array(df['action'].tolist()))
            self.all_states.append(np.array(df['observation.state'].tolist()))

    @property
    def n_episodes(self):
        return len(self.episodes_meta)

    def save_episode(self, frames_front, frames_side, frames_top, states, actions):
        ep_idx = self.n_episodes
        T = len(actions)
        timestamps = np.arange(T, dtype=np.float32) / FPS

        for cam_name, frames in [('cam_front', frames_front),
                                  ('cam_side',  frames_side),
                                  ('cam_top',   frames_top)]:
            vid_path = self.root / 'videos' / f'observation.images.{cam_name}' / 'chunk-000' / f'file-{ep_idx:03d}.mp4'
            writer = imageio.get_writer(str(vid_path), format='ffmpeg', fps=FPS,
                                        codec='libx264', output_params=['-pix_fmt', 'yuv420p'])
            for f in frames:
                writer.append_data(f)
            writer.close()

        rows = []
        for t in range(T):
            rows.append({
                'timestamp':          float(timestamps[t]),
                'frame_index':        int(t),
                'episode_index':      int(ep_idx),
                'index':              int(self.global_index + t),
                'task_index':         int(0),
                'observation.state':  states[t].tolist(),
                'action':             actions[t].tolist(),
                'next.done':          bool(t == T - 1),
                'next.reward':        float(1.0 if t == T - 1 else 0.0),
            })
        pd.DataFrame(rows).to_parquet(
            self.root / 'data' / 'chunk-000' / f'file-{ep_idx:03d}.parquet', index=False)

        self.episodes_meta.append({
            'episode_index':       int(ep_idx),
            'tasks':               [TASK_STR],
            'length':              int(T),
            'dataset_from_index':  int(self.global_index),
            'dataset_to_index':    int(self.global_index + T),
            'videos/observation.images.cam_front/chunk_index':    0,
            'videos/observation.images.cam_front/file_index':     ep_idx,
            'videos/observation.images.cam_front/from_timestamp': float(timestamps[0]),
            'videos/observation.images.cam_front/to_timestamp':   float(timestamps[-1]),
            'videos/observation.images.cam_side/chunk_index':     0,
            'videos/observation.images.cam_side/file_index':      ep_idx,
            'videos/observation.images.cam_side/from_timestamp':  float(timestamps[0]),
            'videos/observation.images.cam_side/to_timestamp':    float(timestamps[-1]),
            'videos/observation.images.cam_top/chunk_index':      0,
            'videos/observation.images.cam_top/file_index':       ep_idx,
            'videos/observation.images.cam_top/from_timestamp':   float(timestamps[0]),
            'videos/observation.images.cam_top/to_timestamp':     float(timestamps[-1]),
        })
        pd.DataFrame(self.episodes_meta).to_parquet(self.eps_parquet, index=False)

        self.global_index += T
        self.all_actions.append(np.array(actions))
        self.all_states.append(np.array(states))
        self._write_meta()
        return T

    def _write_meta(self):
        all_a = np.vstack(self.all_actions)
        all_s = np.vstack(self.all_states)

        def stat(arr):
            return {'mean': arr.mean(0).tolist(), 'std': arr.std(0).tolist(),
                    'min':  arr.min(0).tolist(),  'max': arr.max(0).tolist()}

        stats = {
            'action':            stat(all_a),
            'observation.state': stat(all_s),
            'observation.images.cam_front': {
                'mean': [[[0.485]], [[0.456]], [[0.406]]],
                'std':  [[[0.229]], [[0.224]], [[0.225]]],
                'min':  [[[0.0]],  [[0.0]],  [[0.0]]],
                'max':  [[[1.0]],  [[1.0]],  [[1.0]]],
            },
            'observation.images.cam_side': {
                'mean': [[[0.485]], [[0.456]], [[0.406]]],
                'std':  [[[0.229]], [[0.224]], [[0.225]]],
                'min':  [[[0.0]],  [[0.0]],  [[0.0]]],
                'max':  [[[1.0]],  [[1.0]],  [[1.0]]],
            },
            'observation.images.cam_top': {
                'mean': [[[0.485]], [[0.456]], [[0.406]]],
                'std':  [[[0.229]], [[0.224]], [[0.225]]],
                'min':  [[[0.0]],  [[0.0]],  [[0.0]]],
                'max':  [[[1.0]],  [[1.0]],  [[1.0]]],
            },
        }
        with open(self.root / 'meta' / 'stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        n_ep = self.n_episodes
        info = {
            'codebase_version': 'v3.0',
            'robot_type':       'franka',
            'total_episodes':   n_ep,
            'total_frames':     int(self.global_index),
            'total_tasks':      1,
            'fps':              FPS,
            'splits':           {'train': f'0:{n_ep}'},
            'data_path':        'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'video_path':       'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
            'features': {
                'action': {'dtype': 'float32', 'shape': [ACTION_DIM],
                           'names': ['j0','j1','j2','j3','j4','j5','j6','gripper']},
                'observation.state': {'dtype': 'float32', 'shape': [STATE_DIM],
                                      'names': ['j0','j1','j2','j3','j4','j5','j6','gripper',
                                                'block_x','block_y','block_z',
                                                'ee_x','ee_y','ee_z']},
                'observation.images.cam_front': {
                    'dtype': 'video', 'shape': [IMG_SIZE, IMG_SIZE, 3],
                    'names': ['height','width','channel'],
                    'video_info': {'video.fps': float(FPS), 'video.codec': 'h264',
                                   'video.pix_fmt': 'yuv420p',
                                   'video.is_depth_map': False, 'has_audio': False}},
                'observation.images.cam_side': {
                    'dtype': 'video', 'shape': [IMG_SIZE, IMG_SIZE, 3],
                    'names': ['height','width','channel'],
                    'video_info': {'video.fps': float(FPS), 'video.codec': 'h264',
                                   'video.pix_fmt': 'yuv420p',
                                   'video.is_depth_map': False, 'has_audio': False}},
                'observation.images.cam_top': {
                    'dtype': 'video', 'shape': [IMG_SIZE, IMG_SIZE, 3],
                    'names': ['height','width','channel'],
                    'video_info': {'video.fps': float(FPS), 'video.codec': 'h264',
                                   'video.pix_fmt': 'yuv420p',
                                   'video.is_depth_map': False, 'has_audio': False}},
                'timestamp':     {'dtype': 'float32', 'shape': [1], 'names': None},
                'frame_index':   {'dtype': 'int64',   'shape': [1], 'names': None},
                'episode_index': {'dtype': 'int64',   'shape': [1], 'names': None},
                'index':         {'dtype': 'int64',   'shape': [1], 'names': None},
                'task_index':    {'dtype': 'int64',   'shape': [1], 'names': None},
                'next.done':     {'dtype': 'bool',    'shape': [1], 'names': None},
                'next.reward':   {'dtype': 'float32', 'shape': [1], 'names': None},
            },
        }
        with open(self.root / 'meta' / 'info.json', 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)
        pd.DataFrame([{'task_index': 0, 'task': TASK_STR}]).to_parquet(
            self.root / 'meta' / 'tasks.parquet', index=False)

    def print_summary(self):
        print(f"\n  dataset: {self.root}")
        print(f"  episodes={self.n_episodes}  frames={self.global_index}")


# ── 场景构建 ──────────────────────────────────────────────────────────────────
def build_model():
    # 水面高度（世界坐标）
    water_surface_z = TABLE_H + WATER_DEPTH
    # 方块漂浮平衡深度：ρ_block/ρ_water * block_size
    float_depth = (BLOCK_DENSITY / WATER_DENSITY) * 2 * BALL_RADIUS
    block_init_z = water_surface_z - float_depth + BALL_RADIUS

    panda_tree = ET.parse(f"{FRANKA_DIR}/panda.xml")
    panda_root = panda_tree.getroot()
    for kf in panda_root.findall("keyframe"): panda_root.remove(kf)
    panda_root.find("worldbody").find("body").set("pos", f"{ROBOT_X} 0 {TABLE_H}")
    panda_nokf = f"{FRANKA_DIR}/_panda_buoy.xml"
    panda_tree.write(panda_nokf, encoding="unicode", xml_declaration=False)

    scene_tree = ET.parse(f"{FRANKA_DIR}/scene.xml")
    scene_root = scene_tree.getroot()
    for inc in scene_root.findall("include"): inc.set("file", "_panda_buoy.xml")
    for kf in scene_root.findall("keyframe"): scene_root.remove(kf)
    wb = scene_root.find("worldbody")

    # 桌子
    wb.append(ET.fromstring(f"""
    <body name="exp_table" pos="0 0 0">
      <geom type="box" size="0.55 0.35 0.025" pos="0.05 0 {TABLE_H}"
            rgba="0.50 0.36 0.24 1" mass="30"/>
      <geom type="cylinder" size="0.030 0.39" pos="-0.45 -0.28 0.40" rgba="0.35 0.25 0.18 1"/>
      <geom type="cylinder" size="0.030 0.39" pos=" 0.55 -0.28 0.40" rgba="0.35 0.25 0.18 1"/>
      <geom type="cylinder" size="0.030 0.39" pos="-0.45  0.28 0.40" rgba="0.35 0.25 0.18 1"/>
      <geom type="cylinder" size="0.030 0.39" pos=" 0.55  0.28 0.40" rgba="0.35 0.25 0.18 1"/>
    </body>"""))

    # 水槽（玻璃透明容器）
    tw = TANK_W / 2
    td = TANK_D / 2
    th = TANK_H
    wall_t = 0.005
    tank_bottom_z = TABLE_H + 0.025
    wb.append(ET.fromstring(f"""
    <body name="tank" pos="{TANK_X} {TANK_Y} 0">
      <!-- 底部 -->
      <geom type="box" size="{tw} {td} {wall_t}"
            pos="0 0 {tank_bottom_z + wall_t}"
            rgba="0.2 0.5 0.9 0.3" contype="1" conaffinity="1"/>
      <!-- 前壁 -->
      <geom type="box" size="{tw} {wall_t} {th/2}"
            pos="0 {-td} {tank_bottom_z + th/2}"
            rgba="0.2 0.5 0.9 0.25" contype="1" conaffinity="1"/>
      <!-- 后壁 -->
      <geom type="box" size="{tw} {wall_t} {th/2}"
            pos="0 {td} {tank_bottom_z + th/2}"
            rgba="0.2 0.5 0.9 0.25" contype="1" conaffinity="1"/>
      <!-- 左壁 -->
      <geom type="box" size="{wall_t} {td} {th/2}"
            pos="{-tw} 0 {tank_bottom_z + th/2}"
            rgba="0.2 0.5 0.9 0.25" contype="1" conaffinity="1"/>
      <!-- 右壁 -->
      <geom type="box" size="{wall_t} {td} {th/2}"
            pos="{tw} 0 {tank_bottom_z + th/2}"
            rgba="0.2 0.5 0.9 0.25" contype="1" conaffinity="1"/>
      <!-- 水面（半透明蓝色平面，仅视觉）-->
      <geom type="box" size="{tw-wall_t} {td-wall_t} 0.001"
            pos="0 0 {water_surface_z}"
            rgba="0.1 0.4 0.9 0.4" contype="0" conaffinity="0"/>
    </body>"""))

    # 漂浮小球（自由体）
    import math as _math
    ball_vol   = 4/3 * _math.pi * BALL_RADIUS**3
    block_mass = BLOCK_DENSITY * ball_vol
    wb.append(ET.fromstring(f"""
    <body name="block_body" pos="{TANK_X} 0 {block_init_z:.4f}">
      <freejoint name="block_joint"/>
      <geom name="ball_geom" type="sphere"
            size="{BALL_RADIUS}"
            rgba="0.9 0.3 0.1 1" mass="{block_mass:.4f}"
            contype="1" conaffinity="1"
            friction="0.8 0.1 0.1"/>
    </body>"""))

    # 三路相机
    cx = (ROBOT_X + TANK_X) / 2
    wb.append(ET.fromstring(
        f'<camera name="cam_front" pos="{cx:.3f} -1.10 1.20" '
        f'xyaxes="1 0 0 0 0.25 0.97" fovy="55"/>'))
    wb.append(ET.fromstring(
        f'<camera name="cam_side" pos="1.20 -0.10 1.20" '
        f'xyaxes="0 1 0 -0.25 0 0.97" fovy="60"/>'))
    wb.append(ET.fromstring(
        f'<camera name="cam_top" pos="{cx:.3f} 0 1.80" '
        f'xyaxes="1 0 0 0 1 0" fovy="60"/>'))

    scene_tmp = f"{FRANKA_DIR}/_scene_buoy.xml"
    scene_tree.write(scene_tmp, encoding="unicode", xml_declaration=False)
    try:
        m = mujoco.MjModel.from_xml_path(scene_tmp)
    finally:
        if os.path.exists(scene_tmp):  os.remove(scene_tmp)
        if os.path.exists(panda_nokf): os.remove(panda_nokf)

    water_surface_z_val = water_surface_z
    return m, mujoco.MjData(m), water_surface_z_val


def ik_step(m, d, target):
    bid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    err  = target - d.xpos[bid]
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jacp, jacr, bid)
    J  = jacp[:, :7]
    dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
    dq = np.clip(dq * 400.0 * m.opt.timestep, -0.3, 0.3)
    d.ctrl[:7] = np.clip(d.qpos[:7] + dq, m.jnt_range[:7, 0], m.jnt_range[:7, 1])


def apply_buoyancy(m, d, block_bid, water_surface_z):
    """每步手动施加浮力和水阻尼"""
    ball_z = d.xpos[block_bid][2]
    f_buoy = compute_buoyancy(ball_z, water_surface_z, BALL_RADIUS,
                               BLOCK_DENSITY, WATER_DENSITY)
    block_id = block_bid
    d.xfrc_applied[block_id, 2] = f_buoy
    ball_bottom = ball_z - BALL_RADIUS
    if ball_bottom < water_surface_z:
        submerge_ratio = np.clip((water_surface_z - ball_bottom) / (2*BALL_RADIUS), 0, 1)
        damp = 8.0 * submerge_ratio
        d.xfrc_applied[block_id, 0] -= damp * d.cvel[block_id, 3]
        d.xfrc_applied[block_id, 1] -= damp * d.cvel[block_id, 4]
        d.xfrc_applied[block_id, 2] -= damp * d.cvel[block_id, 5]


def poll_movement(dt):
    if not HAS_KEYBOARD: return np.zeros(3)
    spd = MOVE_SPEED * dt
    delta = np.zeros(3)
    if keyboard.is_pressed('1'): delta[0] += spd
    if keyboard.is_pressed('2'): delta[0] -= spd
    if keyboard.is_pressed('3'): delta[1] -= spd
    if keyboard.is_pressed('4'): delta[1] += spd
    if keyboard.is_pressed('5'): delta[2] += spd
    if keyboard.is_pressed('6'): delta[2] -= spd
    return delta


def record_session(n_demos, writer):
    m, d, water_surface_z = build_model()

    hand_bid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    block_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "block_body")
    id_front  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam_front")
    id_side   = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam_side")
    id_top    = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam_top")

    home_q = np.array([0.0, 0.5, 0.0, -2.0, 0.0, 2.5, 0.785])
    RECORD_EVERY = max(1, int(round(1.0 / (FPS * m.opt.timestep))))

    rf = mujoco.Renderer(m, IMG_SIZE, IMG_SIZE)
    rs = mujoco.Renderer(m, IMG_SIZE, IMG_SIZE)
    rt = mujoco.Renderer(m, IMG_SIZE, IMG_SIZE)

    def reset_sim():
        mujoco.mj_resetData(m, d)
        d.qpos[:7] = home_q.copy()
        d.ctrl[:7] = home_q.copy()
        d.ctrl[7]  = GRIPPER_OPEN
        # 重置方块到水面浮平衡位置
        float_depth = (BLOCK_DENSITY / WATER_DENSITY) * 2 * BALL_RADIUS
        block_z = water_surface_z - float_depth + BALL_RADIUS
        # freejoint: qpos[7:14] = [x, y, z, qw, qx, qy, qz]
        block_jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "block_joint")
        jnt_qposadr = m.jnt_qposadr[block_jnt_id]
        d.qpos[jnt_qposadr:jnt_qposadr+3] = [TANK_X, 0.0, block_z]
        d.qpos[jnt_qposadr+3] = 1.0  # qw
        d.qpos[jnt_qposadr+4:jnt_qposadr+7] = 0.0  # qx qy qz
        mujoco.mj_forward(m, d)
        # ee 目标：方块正上方 10cm，夹爪打开，准备抓取
        hover_z = block_z + BALL_RADIUS + 0.10
        ee_init = np.array([TANK_X, 0.0, hover_z])
        return ee_init, GRIPPER_OPEN

    ee_target, gripper_val = reset_sim()
    ep_done = 0
    is_recording = False
    step_cnt = 0
    bf, bs, bt, bst, bac = [], [], [], [], []
    prev_7 = prev_8 = prev_9 = prev_0 = False

    STEP_FB = 0.02
    def key_callback(keycode):
        nonlocal ee_target
        if not HAS_KEYBOARD:
            fb = {ord('1'):[STEP_FB,0,0], ord('2'):[-STEP_FB,0,0],
                  ord('3'):[0,-STEP_FB,0], ord('4'):[0,STEP_FB,0],
                  ord('5'):[0,0,STEP_FB], ord('6'):[0,0,-STEP_FB]}
            if keycode in fb: ee_target = ee_target + np.array(fb[keycode])

    print(f"\n{'='*65}")
    print(f"  浮力实验  水面高度={water_surface_z:.3f}m")
    print(f"  球密度={BLOCK_DENSITY}kg/m³  水密度={WATER_DENSITY}kg/m³")
    print(f"  state 14维: [j0~j6, gripper, block_xyz, ee_xyz]")
    print(f"  操作流程: 移动到方块上方 → [7]夹住 → 缓慢下压入水 → [9]保存")
    print(f"  注意: 使用6键(下降)将方块压入水中，动作要慢！")
    print(f"{'='*65}\n")

    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as v:
        v.cam.distance = 2.0
        v.cam.elevation = -20
        v.cam.azimuth = 160
        prev_time = time.perf_counter()

        while v.is_running() and ep_done < n_demos:
            now = time.perf_counter()
            dt  = min(now - prev_time, 0.05)
            prev_time = now
            ee_target += poll_movement(dt)

            cur_7 = HAS_KEYBOARD and keyboard.is_pressed('7')
            if cur_7 and not prev_7:
                gripper_val = GRIPPER_CLOSE if gripper_val == GRIPPER_OPEN else GRIPPER_OPEN
                print(f"\n  夹爪: {'闭合' if gripper_val==GRIPPER_CLOSE else '打开'}")
            prev_7 = cur_7

            cur_8 = HAS_KEYBOARD and keyboard.is_pressed('8')
            if cur_8 and not prev_8:
                is_recording = not is_recording
                if is_recording:
                    bf.clear(); bs.clear(); bt.clear()
                    bst.clear(); bac.clear()
                    step_cnt = 0
                    print(f"\n  ▶ 开始录制第 {ep_done+1}/{n_demos} 条...")
                else:
                    print(f"\n  ■ 暂停，已录 {len(bac)} 帧（9保存 / 0放弃）")
            prev_8 = cur_8

            cur_9 = HAS_KEYBOARD and keyboard.is_pressed('9')
            if cur_9 and not prev_9:
                if len(bac) >= MIN_STEPS:
                    is_recording = False
                    T = writer.save_episode(
                        bf, bs, bt,
                        np.array(bst, dtype=np.float32),
                        np.array(bac, dtype=np.float32))
                    ep_done += 1
                    print(f"\n  ✅ 第 {ep_done}/{n_demos} 条已保存（{T} 帧）")
                    writer.print_summary()
                    ee_target, gripper_val = reset_sim()
                    if ep_done < n_demos:
                        print(f"\n  → 准备第 {ep_done+1}/{n_demos} 条")
                else:
                    print(f"\n  ⚠️  仅 {len(bac)} 帧，需 >={MIN_STEPS}")
            prev_9 = cur_9

            cur_0 = HAS_KEYBOARD and keyboard.is_pressed('0')
            if cur_0 and not prev_0:
                is_recording = False
                bf.clear(); bs.clear(); bt.clear()
                bst.clear(); bac.clear()
                ee_target, gripper_val = reset_sim()
                print(f"\n  🔄 已放弃，重录第 {ep_done+1}/{n_demos} 条")
            prev_0 = cur_0

            # 施加浮力
            apply_buoyancy(m, d, block_bid, water_surface_z)

            ik_step(m, d, ee_target)
            d.ctrl[7] = gripper_val
            mujoco.mj_step(m, d)
            step_cnt += 1

            block_pos = d.xpos[block_bid]
            ee_pos    = d.xpos[hand_bid]
            block_z   = block_pos[2]
            submerge  = max(0, water_surface_z - (block_z - BALL_RADIUS))
            submerge_pct = min(100, submerge / (2*BALL_RADIUS) * 100)

            if is_recording and step_cnt % RECORD_EVERY == 0:
                def render(renderer, cam_id):
                    renderer.update_scene(d, camera=cam_id)
                    return renderer.render().copy()
                bf.append(render(rf, id_front))
                bs.append(render(rs, id_side))
                bt.append(render(rt, id_top))

                state = (d.qpos[:7].tolist() +
                         [float(d.qpos[7])] +
                         block_pos.tolist() +
                         ee_pos.tolist())
                bst.append(state)
                bac.append(d.ctrl[:8].tolist())

            v.sync()
            sys.stdout.write(
                f"\r  EE=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f})"
                f"  Ball_z={block_pos[2]:.3f}(水面={water_surface_z:.3f})"
                f"  沉入={submerge_pct:.0f}%"
                f"  夹={'闭' if gripper_val==GRIPPER_CLOSE else '开'}"
                f"  {'🔴录制' if is_recording else '⚫待机'}"
                f"  {len(bac)}帧  {ep_done}/{n_demos}条  "
            )
            sys.stdout.flush()

    rf.close(); rs.close(); rt.close()
    return ep_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",   type=int, default=20)
    ap.add_argument("--out", type=str, default=r"D:\mujuco\demos_buoyancy")
    args = ap.parse_args()

    writer = LeRobotWriter(args.out)
    print(f"dataset: {Path(args.out).resolve()}")
    print(f"已有 {writer.n_episodes} 条  本次录 {args.n} 条  {FPS}Hz")

    saved = record_session(args.n, writer)
    print(f"\n本次保存 {saved}/{args.n} 条")
    writer.print_summary()


if __name__ == "__main__":
    main()
