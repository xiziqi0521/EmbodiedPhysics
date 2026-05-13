"""
zarr_to_lerobot.py —— 将圆弧轨道 Zarr 数据集转换为 lerobot v3.0 格式
=========================================================================
用法：
    python zarr_to_lerobot.py

输出目录：/root/autodl-tmp/demos_lerobot/
"""

import json, os
import numpy as np
import pandas as pd
from pathlib import Path
import zarr
from PIL import Image
from tqdm import tqdm

# ── 路径配置 ──────────────────────────────────────────────────────────────────
ZARR_PATH  = '/root/autodl-tmp/demos_arc_final.zarr'
OUTPUT_DIR = Path('/root/autodl-tmp/demos_lerobot')
REPO_ID    = 'local/arc_demos'
FPS        = 15
TASK_STR   = 'grasp ball and release at target angle on arc track'

# ── 读取 Zarr 数据 ─────────────────────────────────────────────────────────────
print('读取 Zarr 数据集...')
root = zarr.open(ZARR_PATH, mode='r')

actions      = root['data/action'][:]           # (T, 8)
states       = root['data/obs/state'][:]        # (T, 9)
cam_front    = root['data/obs/cam_front'][:]    # (T, 96, 96, 3)
cam_side     = root['data/obs/cam_side'][:]     # (T, 96, 96, 3)
cam_top      = root['data/obs/cam_top'][:]      # (T, 96, 96, 3)
episode_ends = root['meta/episode_ends'][:]     # (N,)

n_episodes = len(episode_ends)
n_frames   = actions.shape[0]
print(f'  episodes={n_episodes}  frames={n_frames}')

# ── 创建目录结构 ───────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'meta').mkdir(exist_ok=True)
(OUTPUT_DIR / 'episodes' / 'chunk-000').mkdir(parents=True, exist_ok=True)

for cam in ['cam_front', 'cam_side', 'cam_top']:
    for ep_idx in range(n_episodes):
        (OUTPUT_DIR / 'images' / cam / f'episode-{ep_idx:06d}').mkdir(
            parents=True, exist_ok=True)

# ── 写入图像和 parquet ─────────────────────────────────────────────────────────
print('写入图像和 parquet...')

ep_starts = np.concatenate([[0], episode_ends[:-1]])
global_index = 0
all_actions  = []
all_states   = []
episodes_meta = []

for ep_idx in tqdm(range(n_episodes)):
    start = int(ep_starts[ep_idx])
    end   = int(episode_ends[ep_idx])
    T     = end - start

    rows = []
    for t in range(T):
        global_t   = start + t
        timestamp  = float(t) / FPS

        # 写图像
        for cam_name, cam_data in [
            ('cam_front', cam_front),
            ('cam_side',  cam_side),
            ('cam_top',   cam_top),
        ]:
            img = Image.fromarray(cam_data[global_t])
            img.save(OUTPUT_DIR / 'images' / cam_name /
                     f'episode-{ep_idx:06d}' / f'frame-{t:06d}.png')

        rows.append({
            'timestamp':           np.float32(timestamp),
            'frame_index':         np.int64(t),
            'episode_index':       np.int64(ep_idx),
            'index':               np.int64(global_index),
            'task_index':          np.int64(0),
            'observation.state':   states[global_t].tolist(),
            'action':              actions[global_t].tolist(),
            'next.done':           bool(t == T - 1),
            'next.reward':         np.float32(1.0 if t == T - 1 else 0.0),
        })
        global_index += 1

    df = pd.DataFrame(rows)
    df.to_parquet(
        OUTPUT_DIR / 'episodes' / 'chunk-000' / f'episode-{ep_idx:06d}.parquet',
        index=False)

    all_actions.append(actions[start:end])
    all_states.append(states[start:end])
    episodes_meta.append({
        'episode_index': ep_idx,
        'tasks': [TASK_STR],
        'length': T,
    })

# ── 计算统计信息 ───────────────────────────────────────────────────────────────
print('计算统计信息...')

def stat(arr):
    return {
        'mean': arr.mean(0).tolist(),
        'std':  arr.std(0).tolist(),
        'min':  arr.min(0).tolist(),
        'max':  arr.max(0).tolist(),
    }

all_actions_np = np.vstack(all_actions)
all_states_np  = np.vstack(all_states)

stats = {
    'action':            stat(all_actions_np),
    'observation.state': stat(all_states_np),
}

# ── 写入 meta 文件 ─────────────────────────────────────────────────────────────
print('写入 meta 文件...')

info = {
    'codebase_version': 'v3.0',
    'robot_type':       'franka',
    'total_episodes':   n_episodes,
    'total_frames':     n_frames,
    'total_tasks':      1,
    'fps':              FPS,
    'splits':           {'train': f'0:{n_episodes}'},
    'data_path':        'episodes/chunk-{chunk_index:03d}/episode-{file_index:06d}.parquet',
    'image_path':       'images/{image_key}/episode-{episode_index:06d}/frame-{frame_index:06d}.png',
    'features': {
        'action': {
            'dtype': 'float32',
            'shape': [8],
            'names': ['j0','j1','j2','j3','j4','j5','j6','gripper'],
        },
        'observation.state': {
            'dtype': 'float32',
            'shape': [9],
            'names': ['j0','j1','j2','j3','j4','j5','j6','gripper','target_angle'],
        },
        'observation.images.cam_front': {
            'dtype': 'image',
            'shape': [96, 96, 3],
            'names': ['height', 'width', 'channel'],
        },
        'observation.images.cam_side': {
            'dtype': 'image',
            'shape': [96, 96, 3],
            'names': ['height', 'width', 'channel'],
        },
        'observation.images.cam_top': {
            'dtype': 'image',
            'shape': [96, 96, 3],
            'names': ['height', 'width', 'channel'],
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

with open(OUTPUT_DIR / 'meta' / 'info.json', 'w') as f:
    json.dump(info, f, indent=2)

with open(OUTPUT_DIR / 'meta' / 'stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

with open(OUTPUT_DIR / 'meta' / 'tasks.jsonl', 'w') as f:
    f.write(json.dumps({'task_index': 0, 'task': TASK_STR}) + '\n')

with open(OUTPUT_DIR / 'meta' / 'episodes.jsonl', 'w') as f:
    for ep in episodes_meta:
        f.write(json.dumps(ep) + '\n')

# ── 验证 ───────────────────────────────────────────────────────────────────────
print('\n验证数据集...')
ok = True
for fname in ['info.json', 'stats.json', 'tasks.jsonl', 'episodes.jsonl']:
    p = OUTPUT_DIR / 'meta' / fname
    s = '✅' if p.exists() else '❌'
    ok = ok and p.exists()
    print(f'  {s}  meta/{fname}')

ep0 = OUTPUT_DIR / 'episodes' / 'chunk-000' / 'episode-000000.parquet'
if ep0.exists():
    df = pd.read_parquet(ep0)
    print(f'  ✅  episodes/chunk-000/episode-000000.parquet — {len(df)} 帧')
else:
    print('  ❌  找不到 episode-000000.parquet')
    ok = False

img0 = OUTPUT_DIR / 'images' / 'cam_front' / 'episode-000000' / 'frame-000000.png'
print(f'  {"✅" if img0.exists() else "❌"}  images/cam_front/episode-000000/frame-000000.png')

print(f'\n{"✅ 转换完成！" if ok else "❌ 有问题，请检查"}')
print(f'输出路径: {OUTPUT_DIR}')
print(f'总 episodes: {n_episodes}  总 frames: {n_frames}')
print(f'\naction std(max): {all_actions_np.std(0).max():.5f}')
print(f'state  std(max): {all_states_np.std(0).max():.5f}')
