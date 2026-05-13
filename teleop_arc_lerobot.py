"""
teleop_arc_lerobot.py —— 圆弧轨道示教录制（直接输出 lerobot v3.0 格式）
=========================================================================
state 14维：[j0~j6, gripper, ball_x, ball_y, ball_z, ee_x, ee_y, ee_z]
action  8维：ctrl[:8]
三路相机：cam_front / cam_side / cam_top，96×96，直接存 mp4

输出目录结构（lerobot v3.0）：
  demos_arc_lerobot/
  ├── meta/
  │   ├── info.json
  │   ├── stats.json
  │   ├── tasks.parquet
  │   └── episodes/chunk-000/file-000.parquet
  ├── data/chunk-000/file-NNN.parquet
  └── videos/
      ├── observation.images.cam_front/chunk-000/file-NNN.mp4
      ├── observation.images.cam_side/chunk-000/file-NNN.mp4
      └── observation.images.cam_top/chunk-000/file-NNN.mp4

用法：
    python teleop_arc_lerobot.py --theta 20
    python teleop_arc_lerobot.py --theta 20 --n 10 --out D:/mujuco/demos_arc_lerobot

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
ARC_X         = 0.30
MOVE_SPEED    = 0.06
GRIPPER_OPEN  = 0.0
GRIPPER_CLOSE = 255.0
G             = 9.81
IMG_SIZE      = 96
MIN_STEPS     = 100
FPS           = 15
STATE_DIM     = 14
ACTION_DIM    = 8
TASK_STR      = 'grasp ball on arc track and release at target angle'


# ── lerobot v3.0 写入器 ────────────────────────────────────────────────────────
class LeRobotWriter:
    def __init__(self, out_dir):
        self.root = Path(out_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'meta' / 'episodes' / 'chunk-000').mkdir(parents=True, exist_ok=True)
        (self.root / 'data' / 'chunk-000').mkdir(parents=True, exist_ok=True)
        for cam in ['cam_front', 'cam_side', 'cam_top']:
            (self.root / 'videos' / f'observation.images.{cam}' / 'chunk-000').mkdir(parents=True, exist_ok=True)

        # 加载已有 episodes 信息
        self.eps_parquet = self.root / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet'
        self.episodes_meta = []
        self.global_index  = 0

        if self.eps_parquet.exists():
            df = pd.read_parquet(self.eps_parquet)
            self.episodes_meta = df.to_dict('records')
            # 读取最后一集的结束 index
            data_files = sorted((self.root / 'data' / 'chunk-000').glob('file-*.parquet'))
            if data_files:
                last_df = pd.read_parquet(data_files[-1])
                self.global_index = int(last_df['index'].iloc[-1]) + 1

        self.all_actions = []
        self.all_states  = []
        # 重新加载已有数据的统计
        for f in sorted((self.root / 'data' / 'chunk-000').glob('file-*.parquet')):
            df = pd.read_parquet(f)
            self.all_actions.append(np.array(df['action'].tolist()))
            self.all_states.append(np.array(df['observation.state'].tolist()))

    @property
    def n_episodes(self):
        return len(self.episodes_meta)

    def save_episode(self, frames_front, frames_side, frames_top, states, actions, theta):
        ep_idx = self.n_episodes
        T = len(actions)
        timestamps = np.arange(T, dtype=np.float32) / FPS

        # 写视频
        for cam_name, frames in [('cam_front', frames_front),
                                  ('cam_side',  frames_side),
                                  ('cam_top',   frames_top)]:
            vid_path = self.root / 'videos' / f'observation.images.{cam_name}' / 'chunk-000' / f'file-{ep_idx:03d}.mp4'
            writer = imageio.get_writer(str(vid_path), format='ffmpeg', fps=FPS,
                                        codec='libx264', output_params=['-pix_fmt', 'yuv420p'])
            for f in frames:
                writer.append_data(f)
            writer.close()

        # 写 data parquet
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
        data_df = pd.DataFrame(rows)
        data_df.to_parquet(self.root / 'data' / 'chunk-000' / f'file-{ep_idx:03d}.parquet', index=False)

        # 更新 meta/episodes
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
        with open(self.root / 'meta' / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        n_ep = self.n_episodes
        total_frames = self.global_index
        info = {
            'codebase_version': 'v3.0',
            'robot_type':       'franka',
            'total_episodes':   n_ep,
            'total_frames':     total_frames,
            'total_tasks':      1,
            'fps':              FPS,
            'splits':           {'train': f'0:{n_ep}'},
            'data_path':        'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'video_path':       'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
            'features': {
                'action': {
                    'dtype': 'float32', 'shape': [ACTION_DIM],
                    'names': ['j0','j1','j2','j3','j4','j5','j6','gripper'],
                },
                'observation.state': {
                    'dtype': 'float32', 'shape': [STATE_DIM],
                    'names': ['j0','j1','j2','j3','j4','j5','j6','gripper',
                              'ball_x','ball_y','ball_z',
                              'ee_x','ee_y','ee_z'],
                },
                'observation.images.cam_front': {
                    'dtype': 'video', 'shape': [IMG_SIZE, IMG_SIZE, 3],
                    'names': ['height','width','channel'],
                    'video_info': {'video.fps': float(FPS), 'video.codec': 'h264',
                                   'video.pix_fmt': 'yuv420p',
                                   'video.is_depth_map': False, 'has_audio': False},
                },
                'observation.images.cam_side': {
                    'dtype': 'video', 'shape': [IMG_SIZE, IMG_SIZE, 3],
                    'names': ['height','width','channel'],
                    'video_info': {'video.fps': float(FPS), 'video.codec': 'h264',
                                   'video.pix_fmt': 'yuv420p',
                                   'video.is_depth_map': False, 'has_audio': False},
                },
                'observation.images.cam_top': {
                    'dtype': 'video', 'shape': [IMG_SIZE, IMG_SIZE, 3],
                    'names': ['height','width','channel'],
                    'video_info': {'video.fps': float(FPS), 'video.codec': 'h264',
                                   'video.pix_fmt': 'yuv420p',
                                   'video.is_depth_map': False, 'has_audio': False},
                },
                'timestamp':     {'dtype': 'float32', 'shape': [1], 'names': None},
                'frame_index':   {'dtype': 'int64',   'shape': [1], 'names': None},
                'episode_index': {'dtype': 'int64',   'shape': [1], 'names': None},
                'index':         {'dtype': 'int64',   'shape': [1], 'names': None},
                'task_index':    {'dtype': 'int64',   'shape': [1], 'names': None},
                'next.done':     {'dtype': 'bool',    'shape': [1], 'names': None},
                'next.reward':   {'dtype': 'float32', 'shape': [1], 'names': None},
            },
        }
        with open(self.root / 'meta' / 'info.json', 'w') as f:
            json.dump(info, f, indent=2)

        pd.DataFrame([{'task_index': 0, 'task': TASK_STR}]).to_parquet(
            self.root / 'meta' / 'tasks.parquet', index=False)

    def print_summary(self):
        print(f"\n  dataset: {self.root}")
        print(f"  episodes={self.n_episodes}  frames={self.global_index}")


# ── 场景构建（与 teleop_arc_v2.py 完全一致）─────────────────────────────────
def build_model(R, m_ball=0.10):
    arc_pivot_h = TABLE_H + R + 0.05
    cx = (ROBOT_X + ARC_X) / 2

    panda_tree = ET.parse(f"{FRANKA_DIR}/panda.xml")
    panda_root = panda_tree.getroot()
    for kf in panda_root.findall("keyframe"): panda_root.remove(kf)
    panda_root.find("worldbody").find("body").set("pos", f"{ROBOT_X} 0 {TABLE_H}")
    panda_nokf = f"{FRANKA_DIR}/_panda_arc.xml"
    panda_tree.write(panda_nokf, encoding="unicode", xml_declaration=False)

    scene_tree = ET.parse(f"{FRANKA_DIR}/scene.xml")
    scene_root = scene_tree.getroot()
    for inc in scene_root.findall("include"): inc.set("file", "_panda_arc.xml")
    for kf in scene_root.findall("keyframe"): scene_root.remove(kf)
    wb = scene_root.find("worldbody")

    wb.append(ET.fromstring(f"""
    <body name="exp_table" pos="0.05 0 0">
      <geom type="box" size="0.55 0.35 0.025" pos="0 0 {TABLE_H}" rgba="0.50 0.36 0.24 1" mass="30"/>
      <geom type="cylinder" size="0.030 0.39" pos="-0.45 -0.28 0.40" rgba="0.35 0.25 0.18 1"/>
      <geom type="cylinder" size="0.030 0.39" pos=" 0.45 -0.28 0.40" rgba="0.35 0.25 0.18 1"/>
      <geom type="cylinder" size="0.030 0.39" pos="-0.45  0.28 0.40" rgba="0.35 0.25 0.18 1"/>
      <geom type="cylinder" size="0.030 0.39" pos=" 0.45  0.28 0.40" rgba="0.35 0.25 0.18 1"/>
    </body>"""))

    TUBE_R = 0.012
    angles = np.linspace(np.radians(-75), np.radians(75), 121)
    R_cl   = R + TUBE_R
    arc_xml = f"""
    <body name="arc_stand" pos="{ARC_X} 0 0">
      <geom type="cylinder" size="0.012" fromto="0 0 {TABLE_H+0.025} 0 0 {arc_pivot_h}"
            rgba="0.25 0.25 0.30 1" mass="1.0" contype="0" conaffinity="0"/>
      <geom type="sphere" size="0.015" pos="0 0 {arc_pivot_h}"
            rgba="0.30 0.30 0.36 1" contype="0" conaffinity="0"/>
    """
    for i in range(120):
        a0, a1 = angles[i], angles[i+1]
        x0=R_cl*np.sin(a0); z0=arc_pivot_h-R_cl*np.cos(a0)
        x1=R_cl*np.sin(a1); z1=arc_pivot_h-R_cl*np.cos(a1)
        arc_xml += (f'      <geom type="capsule" size="{TUBE_R}" '
                    f'fromto="{x0:.4f} 0 {z0:.4f} {x1:.4f} 0 {z1:.4f}" '
                    f'rgba="0.15 0.40 0.85 1" contype="0" conaffinity="0"/>\n')
    for ad in range(-60, 61, 10):
        ang=np.radians(ad)
        ro=R+TUBE_R*2+0.030; bx=ro*np.sin(ang); bz=arc_pivot_h-ro*np.cos(ang)
        ri=R+TUBE_R*2+0.005; ix=ri*np.sin(ang); iz=arc_pivot_h-ri*np.cos(ang)
        col=("1 1 1 1" if ad==0 else "0.2 0.95 0.2 1" if abs(ad)<=20 else
             "1.0 0.80 0.0 1" if abs(ad)<=40 else "1.0 0.20 0.2 1")
        arc_xml += (f'      <geom type="sphere" size="0.011" pos="{bx:.4f} 0 {bz:.4f}" '
                    f'rgba="{col}" contype="0" conaffinity="0"/>\n'
                    f'      <geom type="cylinder" size="0.003" '
                    f'fromto="{ix:.4f} 0 {iz:.4f} {bx:.4f} 0 {bz:.4f}" '
                    f'rgba="{col}" contype="0" conaffinity="0"/>\n')
    arc_xml += f"""
      <body name="arc_mover" pos="0 0 {arc_pivot_h}">
        <joint name="arc_hinge" type="hinge" axis="0 1 0" range="-1.50 1.50" damping="0.002"/>
        <geom type="cylinder" size="0.003" fromto="0 0 0 0 0 {-R}"
              rgba="0 0 0 0" mass="0.001" contype="0" conaffinity="0"/>
        <body name="ball_body" pos="0 0 {-R}">
          <geom name="ball_geom" type="sphere" size="0.025" rgba="0.90 0.30 0.10 1"
                mass="{m_ball}" contype="1" conaffinity="1" condim="6"
                friction="0.05 0.005 0.005"/>
        </body>
      </body>
    </body>"""
    wb.append(ET.fromstring(arc_xml))

    wb.append(ET.fromstring(
        f'<camera name="cam_front" pos="{cx:.3f} -1.20 1.40" '
        f'xyaxes="1 0 0 0 0.30 0.95" fovy="55"/>'))
    wb.append(ET.fromstring(
        f'<camera name="cam_side" pos="1.50 -0.30 1.40" '
        f'xyaxes="0 1 0 -0.30 0 0.95" fovy="65"/>'))
    wb.append(ET.fromstring(
        f'<camera name="cam_top" pos="{cx:.3f} 0 2.10" '
        f'xyaxes="1 0 0 0 1 0" fovy="65"/>'))

    sensor = scene_root.find("sensor")
    if sensor is None: sensor = ET.SubElement(scene_root, "sensor")
    sensor.append(ET.fromstring('<jointpos name="arc_angle" joint="arc_hinge"/>'))
    sensor.append(ET.fromstring('<jointvel name="arc_vel"   joint="arc_hinge"/>'))

    scene_tmp = f"{FRANKA_DIR}/_scene_arc.xml"
    scene_tree.write(scene_tmp, encoding="unicode", xml_declaration=False)
    try:
        m = mujoco.MjModel.from_xml_path(scene_tmp)
    finally:
        if os.path.exists(scene_tmp):  os.remove(scene_tmp)
        if os.path.exists(panda_nokf): os.remove(panda_nokf)
    return m, mujoco.MjData(m)


def ik_step(m, d, target):
    bid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    err  = target - d.xpos[bid]
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jacp, jacr, bid)
    J  = jacp[:, :7]
    dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
    dq = np.clip(dq * 400.0 * m.opt.timestep, -0.3, 0.3)
    d.ctrl[:7] = np.clip(d.qpos[:7] + dq, m.jnt_range[:7, 0], m.jnt_range[:7, 1])


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


def record_session(R, theta, m_ball, n_demos, writer):
    v_theory = np.sqrt(2 * G * R * (1 - np.cos(np.radians(theta))))

    m, d = build_model(R, m_ball)
    hand_bid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,   "hand")
    ball_bid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,   "ball_body")
    sid_angle = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "arc_angle")
    sid_vel   = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "arc_vel")
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
        d.qpos[:7] = home_q.copy(); d.ctrl[:7] = home_q.copy()
        d.ctrl[7] = GRIPPER_OPEN
        mujoco.mj_forward(m, d)
        return d.xpos[hand_bid].copy(), GRIPPER_OPEN

    ee_target, gripper_val = reset_sim()
    ep_done = 0; is_recording = False; step_cnt = 0
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
    print(f"  θ={theta:.0f}°  R={R:.2f}m  v_theory={v_theory:.3f}m/s")
    print(f"  state 14维: [j0~j6, gripper, ball_xyz, ee_xyz]")
    print(f"  操作: 移动→夹球→抬到θ→[8]开录→松开→等摆→[9]保存  [0]放弃")
    print(f"{'='*65}\n")

    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as v:
        v.cam.distance=2.8; v.cam.elevation=-18; v.cam.azimuth=150
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
                        np.array(bac, dtype=np.float32),
                        theta)
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

            ik_step(m, d, ee_target)
            d.ctrl[7] = gripper_val
            mujoco.mj_step(m, d)
            step_cnt += 1

            angle_rad = float(d.sensordata[sid_angle])
            omega     = float(d.sensordata[sid_vel])
            angle_deg = np.degrees(angle_rad)
            v_ball    = abs(omega) * R

            if is_recording and step_cnt % RECORD_EVERY == 0:
                def render(renderer, cam_id):
                    renderer.update_scene(d, camera=cam_id)
                    return renderer.render().copy()
                bf.append(render(rf, id_front))
                bs.append(render(rs, id_side))
                bt.append(render(rt, id_top))

                # state 14维：[j0~j6, gripper, ball_xyz, ee_xyz]
                ball_pos = d.xpos[ball_bid].tolist()
                ee_pos   = d.xpos[hand_bid].tolist()
                state = (d.qpos[:7].tolist() +
                         [float(d.qpos[7])] +
                         ball_pos +
                         ee_pos)
                bst.append(state)
                bac.append(d.ctrl[:8].tolist())

            v.sync()
            ee = d.xpos[hand_bid]
            sys.stdout.write(
                f"\r  EE=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})"
                f"  θ={angle_deg:+6.1f}°(目标{theta:+.0f}°)"
                f"  v={v_ball:.3f}m/s"
                f"  夹={'闭' if gripper_val==GRIPPER_CLOSE else '开'}"
                f"  {'🔴录制' if is_recording else '⚫待机'}"
                f"  {len(bac)}帧  {ep_done}/{n_demos}条  "
            )
            sys.stdout.flush()

    rf.close(); rs.close(); rt.close()
    return ep_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R",     type=float, default=0.40)
    ap.add_argument("--theta", type=float, required=True)
    ap.add_argument("--m",     type=float, default=0.10)
    ap.add_argument("--n",     type=int,   default=10)
    ap.add_argument("--out",   type=str,   default=r"D:\mujuco\demos_arc_lerobot")
    args = ap.parse_args()

    writer = LeRobotWriter(args.out)
    print(f"dataset: {Path(args.out).resolve()}")
    print(f"已有 {writer.n_episodes} 条  本次录 {args.n} 条  θ={args.theta}°  {FPS}Hz")

    saved = record_session(args.R, args.theta, args.m, args.n, writer)
    print(f"\n本次保存 {saved}/{args.n} 条")
    writer.print_summary()


if __name__ == "__main__":
    main()
