"""
augment_lerobot.py —— lerobot v3.0 格式数据增强脚本
=====================================================
将现有 demos 扩充：20条 → 50条

增强策略：
  - action  关节角加高斯噪声 ±0.01 rad，夹爪维度不加噪声
  - state   ball_xyz 加 ±0.005m，ee_xyz 加 ±0.003m，关节角加 ±0.005 rad
  - 视频帧  亮度/对比度/饱和度随机抖动（不改变语义）
  - 每条原始 demo 生成若干增强副本

用法：
    python augment_lerobot.py \
        --src D:/mujuco/demos_arc_lerobot \
        --dst D:/mujuco/demos_arc_aug \
        --target 50 \
        --seed 42
"""

import argparse, json, shutil, random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("pip install opencv-python  # 用于视频增强")

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


# ── 噪声参数 ──────────────────────────────────────────────────────────────────
JOINT_STD    = 0.008   # 关节角噪声 (rad)
BALL_STD     = 0.004   # 球位置噪声 (m)
EE_STD       = 0.003   # 末端位置噪声 (m)
ACTION_STD   = 0.008   # action 关节角噪声 (rad)

# 视频亮度/对比度增强范围
BRIGHTNESS_RANGE = (-15, 15)   # pixel value offset
CONTRAST_RANGE   = (0.90, 1.10)


def augment_state(states: np.ndarray, rng) -> np.ndarray:
    """state 14维: [j0~j6, gripper, ball_xyz, ee_xyz]"""
    aug = states.copy()
    T = len(aug)
    # 关节角 (0:7) 加连续噪声（smoothed）
    joint_noise = rng.normal(0, JOINT_STD, (T, 7))
    # 平滑噪声，让相邻帧变化连续
    for i in range(1, T):
        joint_noise[i] = 0.7 * joint_noise[i-1] + 0.3 * joint_noise[i]
    aug[:, :7] += joint_noise
    # gripper (7) 不加噪声
    # ball_xyz (8:11)
    ball_offset = rng.normal(0, BALL_STD, 3)  # 同一集偏移一致
    aug[:, 8:11] += ball_offset
    # ee_xyz (11:14)
    ee_noise = rng.normal(0, EE_STD, (T, 3))
    for i in range(1, T):
        ee_noise[i] = 0.7 * ee_noise[i-1] + 0.3 * ee_noise[i]
    aug[:, 11:14] += ee_noise
    return aug.astype(np.float32)


def augment_action(actions: np.ndarray, rng) -> np.ndarray:
    """action 8维: [j0~j6, gripper]"""
    aug = actions.copy()
    T = len(aug)
    noise = rng.normal(0, ACTION_STD, (T, 7))
    for i in range(1, T):
        noise[i] = 0.7 * noise[i-1] + 0.3 * noise[i]
    aug[:, :7] += noise
    # gripper (7) 不加噪声
    return aug.astype(np.float32)


def augment_video(src_path: Path, dst_path: Path, rng):
    """读取 mp4，加亮度/对比度扰动，保存到 dst_path"""
    if not HAS_CV2 or not HAS_IMAGEIO:
        # 没有 cv2/imageio，直接复制
        shutil.copy(src_path, dst_path)
        return

    cap = cv2.VideoCapture(str(src_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        shutil.copy(src_path, dst_path)
        return

    # 随机增强参数（同一集保持一致）
    brightness = rng.integers(*BRIGHTNESS_RANGE)
    contrast   = rng.uniform(*CONTRAST_RANGE)

    aug_frames = []
    for f in frames:
        f = f.astype(np.float32)
        f = f * contrast + brightness
        f = np.clip(f, 0, 255).astype(np.uint8)
        aug_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

    writer = imageio.get_writer(str(dst_path), format='ffmpeg', fps=int(fps),
                                codec='libx264', output_params=['-pix_fmt', 'yuv420p'])
    for f in aug_frames:
        writer.append_data(f)
    writer.close()


def copy_video(src_path: Path, dst_path: Path):
    shutil.copy(src_path, dst_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src',    type=str, required=True, help='原始数据集路径')
    ap.add_argument('--dst',    type=str, required=True, help='输出数据集路径')
    ap.add_argument('--target', type=int, default=50,    help='目标 episode 数量')
    ap.add_argument('--seed',   type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    rng = np.random.default_rng(args.seed)

    # ── 读取原始数据集信息 ────────────────────────────────────────────────────
    with open(src / 'meta/info.json') as f:
        info = json.load(f)

    eps_df = pd.read_parquet(src / 'meta/episodes/chunk-000/file-000.parquet')
    n_orig = len(eps_df)
    n_aug  = args.target - n_orig

    print(f'原始 episodes: {n_orig}')
    print(f'目标 episodes: {args.target}')
    print(f'需生成增强:    {n_aug}')

    if n_aug <= 0:
        print('已达到目标数量，无需增强')
        return

    # ── 创建输出目录 ──────────────────────────────────────────────────────────
    dst.mkdir(parents=True, exist_ok=True)
    (dst / 'meta/episodes/chunk-000').mkdir(parents=True, exist_ok=True)
    (dst / 'data/chunk-000').mkdir(parents=True, exist_ok=True)
    for cam in ['cam_front', 'cam_side', 'cam_top']:
        (dst / f'videos/observation.images.{cam}/chunk-000').mkdir(parents=True, exist_ok=True)

    # ── 复制原始数据 ──────────────────────────────────────────────────────────
    print('\n复制原始数据...')
    all_episodes_meta = []
    all_actions = []
    all_states  = []
    global_index = 0

    for ep_idx in tqdm(range(n_orig)):
        # 读取原始 parquet
        data_df = pd.read_parquet(src / f'data/chunk-000/file-{ep_idx:03d}.parquet')
        T = len(data_df)

        # 更新 index
        data_df['index'] = range(global_index, global_index + T)
        data_df['episode_index'] = ep_idx
        data_df.to_parquet(dst / f'data/chunk-000/file-{ep_idx:03d}.parquet', index=False)

        # 复制视频
        for cam in ['cam_front', 'cam_side', 'cam_top']:
            src_vid = src / f'videos/observation.images.{cam}/chunk-000/file-{ep_idx:03d}.mp4'
            dst_vid = dst / f'videos/observation.images.{cam}/chunk-000/file-{ep_idx:03d}.mp4'
            shutil.copy(src_vid, dst_vid)

        timestamps = data_df['timestamp'].values
        all_episodes_meta.append({
            'episode_index':      ep_idx,
            'tasks':              ['grasp ball on arc track and release at target angle'],
            'length':             T,
            'dataset_from_index': global_index,
            'dataset_to_index':   global_index + T,
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

        actions = np.array(data_df['action'].tolist())
        states  = np.array(data_df['observation.state'].tolist())
        all_actions.append(actions)
        all_states.append(states)
        global_index += T

    # ── 生成增强数据 ──────────────────────────────────────────────────────────
    print(f'\n生成 {n_aug} 条增强数据...')

    # 循环选择原始 episode 来增强
    aug_ep_idx = n_orig
    src_indices = list(range(n_orig))

    with tqdm(total=n_aug) as pbar:
        i = 0
        while aug_ep_idx < args.target:
            src_ep = src_indices[i % n_orig]
            i += 1

            # 读取原始数据
            data_df = pd.read_parquet(src / f'data/chunk-000/file-{src_ep:03d}.parquet')
            T = len(data_df)
            states  = np.array(data_df['observation.state'].tolist())
            actions = np.array(data_df['action'].tolist())
            timestamps = data_df['timestamp'].values

            # 增强
            aug_states  = augment_state(states, rng)
            aug_actions = augment_action(actions, rng)

            # 写增强 parquet
            aug_df = data_df.copy()
            aug_df['observation.state'] = aug_states.tolist()
            aug_df['action']            = aug_actions.tolist()
            aug_df['episode_index']     = aug_ep_idx
            aug_df['index']             = range(global_index, global_index + T)
            aug_df.to_parquet(dst / f'data/chunk-000/file-{aug_ep_idx:03d}.parquet', index=False)

            # 增强视频
            for cam in ['cam_front', 'cam_side', 'cam_top']:
                src_vid = src / f'videos/observation.images.{cam}/chunk-000/file-{src_ep:03d}.mp4'
                dst_vid = dst / f'videos/observation.images.{cam}/chunk-000/file-{aug_ep_idx:03d}.mp4'
                augment_video(src_vid, dst_vid, rng)

            all_episodes_meta.append({
                'episode_index':      aug_ep_idx,
                'tasks':              ['grasp ball on arc track and release at target angle'],
                'length':             T,
                'dataset_from_index': global_index,
                'dataset_to_index':   global_index + T,
                'videos/observation.images.cam_front/chunk_index':    0,
                'videos/observation.images.cam_front/file_index':     aug_ep_idx,
                'videos/observation.images.cam_front/from_timestamp': float(timestamps[0]),
                'videos/observation.images.cam_front/to_timestamp':   float(timestamps[-1]),
                'videos/observation.images.cam_side/chunk_index':     0,
                'videos/observation.images.cam_side/file_index':      aug_ep_idx,
                'videos/observation.images.cam_side/from_timestamp':  float(timestamps[0]),
                'videos/observation.images.cam_side/to_timestamp':    float(timestamps[-1]),
                'videos/observation.images.cam_top/chunk_index':      0,
                'videos/observation.images.cam_top/file_index':       aug_ep_idx,
                'videos/observation.images.cam_top/from_timestamp':   float(timestamps[0]),
                'videos/observation.images.cam_top/to_timestamp':     float(timestamps[-1]),
            })

            all_actions.append(aug_actions)
            all_states.append(aug_states)
            global_index += T
            aug_ep_idx += 1
            pbar.update(1)

    # ── 写 meta ───────────────────────────────────────────────────────────────
    print('\n写入 meta...')
    pd.DataFrame(all_episodes_meta).to_parquet(
        dst / 'meta/episodes/chunk-000/file-000.parquet', index=False)

    all_a = np.vstack(all_actions)
    all_s = np.vstack(all_states)

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
    with open(dst / 'meta/stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    info_new = dict(info)
    info_new['total_episodes'] = args.target
    info_new['total_frames']   = int(global_index)
    info_new['splits']         = {'train': f'0:{args.target}'}
    with open(dst / 'meta/info.json', 'w') as f:
        json.dump(info_new, f, indent=2)

    pd.DataFrame([{'task_index': 0, 'task': 'grasp ball on arc track and release at target angle'}]
                 ).to_parquet(dst / 'meta/tasks.parquet', index=False)

    print(f'\n✅ 扩充完成！')
    print(f'  原始: {n_orig} 条')
    print(f'  增强: {args.target} 条')
    print(f'  总帧数: {global_index}')
    print(f'  输出: {dst}')
    print(f'\naction std(max): {all_a.std(0).max():.5f}')


if __name__ == '__main__':
    main()
