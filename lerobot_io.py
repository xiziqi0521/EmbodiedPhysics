# -*- coding: utf-8 -*-
"""
lerobot_io.py —— lerobot v3.0 数据集读写公共工具
=====================================================================
被 D1~D5 各构造脚本、指标计算脚本、相关性分析脚本共同引用。

数据集目录结构（与 teleop_buoyancy_lerobot.py 的 LeRobotWriter 保持一致）：
    root/
      meta/
        episodes/chunk-000/file-000.parquet   # 每条episode的元信息
        stats.json                            # 归一化统计量
        info.json                             # 数据集整体信息
        tasks.parquet
      data/
        chunk-000/file-{ep:03d}.parquet       # 每条episode一个文件，逐帧记录
      videos/
        observation.images.{cam}/chunk-000/file-{ep:03d}.mp4

state 14维: [j0~j6, gripper, block_x, block_y, block_z, ee_x, ee_y, ee_z]
action  8维: [j0~j6, gripper]
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 15
STATE_DIM = 14
ACTION_DIM = 8
IMG_SIZE = 96
CAMS = ["cam_front", "cam_side", "cam_top"]

STATE_NAMES = ["j0", "j1", "j2", "j3", "j4", "j5", "j6", "gripper",
               "block_x", "block_y", "block_z", "ee_x", "ee_y", "ee_z"]
ACTION_NAMES = ["j0", "j1", "j2", "j3", "j4", "j5", "j6", "gripper"]


# ── 读取 ──────────────────────────────────────────────────────────────────
class LeRobotDataset:
    """只读方式加载一个 lerobot v3.0 数据集，episode 粒度访问。"""

    def __init__(self, root):
        self.root = Path(root)
        self.eps_parquet = self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        if not self.eps_parquet.exists():
            raise FileNotFoundError(f"找不到 episodes 元数据: {self.eps_parquet}")
        self.episodes_meta = pd.read_parquet(self.eps_parquet).to_dict("records")
        self.episodes_meta.sort(key=lambda r: r["episode_index"])

        with open(self.root / "meta" / "info.json", "r") as f:
            self.info = json.load(f)

    @property
    def n_episodes(self):
        return len(self.episodes_meta)

    def episode_data_path(self, ep_idx):
        return self.root / "data" / "chunk-000" / f"file-{ep_idx:03d}.parquet"

    def episode_video_path(self, ep_idx, cam):
        return self.root / "videos" / f"observation.images.{cam}" / "chunk-000" / f"file-{ep_idx:03d}.mp4"

    def load_episode(self, ep_idx):
        """返回 (states[T,14], actions[T,8], meta_dict)"""
        df = pd.read_parquet(self.episode_data_path(ep_idx))
        states = np.array(df["observation.state"].tolist(), dtype=np.float32)
        actions = np.array(df["action"].tolist(), dtype=np.float32)
        meta = next(m for m in self.episodes_meta if m["episode_index"] == ep_idx)
        return states, actions, meta

    def load_episode_frames(self, ep_idx, cam):
        """用 imageio 读出某个相机某条episode的全部帧，返回 list[np.ndarray(H,W,3)]"""
        import imageio.v3 as iio
        vid_path = self.episode_video_path(ep_idx, cam)
        return list(iio.imiter(str(vid_path)))

    def all_episode_indices(self):
        return [m["episode_index"] for m in self.episodes_meta]


# ── 写入（新数据集，从零开始，episode_index 从0连续编号）───────────────────
class LeRobotWriter:
    """
    与 teleop_buoyancy_lerobot.py 中的 LeRobotWriter 行为一致，
    但额外支持指定 task_str，用于给 D1~D5 数据集打上不同的任务/构造说明标签。
    """

    def __init__(self, out_dir, task_str, fps=FPS):
        self.root = Path(out_dir)
        self.task_str = task_str
        self.fps = fps
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        for cam in CAMS:
            (self.root / "videos" / f"observation.images.{cam}" / "chunk-000").mkdir(
                parents=True, exist_ok=True)

        self.eps_parquet = self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        self.episodes_meta = []
        self.global_index = 0
        self.all_actions = []
        self.all_states = []

    @property
    def n_episodes(self):
        return len(self.episodes_meta)

    def save_episode_from_video_copy(self, src_ds: LeRobotDataset, src_ep_idx,
                                      states, actions, frame_indices=None):
        """
        写入一条 episode。states/actions 是最终要落盘的序列（可能是抽帧/加噪后的结果）。
        若 frame_indices 不为 None，说明视频帧要跟着 states/actions 一起做同样的抽取
        （用于 D1/D2/D3 这种不需要重新仿真、只是重采样的场景）。
        若 frame_indices 为 None，说明视频改由调用方直接传入渲染好的帧（用于 D4/D5 重新仿真场景），
        此时请改用 save_episode_with_frames。
        """
        assert frame_indices is not None, "未提供 frame_indices 时请使用 save_episode_with_frames"
        ep_idx = self.n_episodes
        T = len(actions)
        timestamps = np.arange(T, dtype=np.float32) / self.fps

        import imageio
        for cam in CAMS:
            src_frames = src_ds.load_episode_frames(src_ep_idx, cam)
            sel_frames = [src_frames[i] for i in frame_indices]
            vid_path = self.root / "videos" / f"observation.images.{cam}" / "chunk-000" / f"file-{ep_idx:03d}.mp4"
            writer = imageio.get_writer(str(vid_path), format="ffmpeg", fps=self.fps,
                                         codec="libx264", output_params=["-pix_fmt", "yuv420p"])
            for fr in sel_frames:
                writer.append_data(fr)
            writer.close()

        self._write_episode_common(ep_idx, states, actions, timestamps)
        return T

    def save_episode_with_frames(self, frames_dict, states, actions):
        """
        frames_dict: {'cam_front': [frame,...], 'cam_side': [...], 'cam_top': [...]}
        用于 D4/D5：视频是重新仿真渲染出来的，不是从旧视频抽取的。
        """
        ep_idx = self.n_episodes
        T = len(actions)
        timestamps = np.arange(T, dtype=np.float32) / self.fps

        import imageio
        for cam in CAMS:
            frames = frames_dict[cam]
            vid_path = self.root / "videos" / f"observation.images.{cam}" / "chunk-000" / f"file-{ep_idx:03d}.mp4"
            writer = imageio.get_writer(str(vid_path), format="ffmpeg", fps=self.fps,
                                         codec="libx264", output_params=["-pix_fmt", "yuv420p"])
            for fr in frames:
                writer.append_data(fr)
            writer.close()

        self._write_episode_common(ep_idx, states, actions, timestamps)
        return T

    def _write_episode_common(self, ep_idx, states, actions, timestamps):
        T = len(actions)
        rows = []
        for t in range(T):
            rows.append({
                "timestamp": float(timestamps[t]),
                "frame_index": int(t),
                "episode_index": int(ep_idx),
                "index": int(self.global_index + t),
                "task_index": int(0),
                "observation.state": np.asarray(states[t], dtype=np.float32).tolist(),
                "action": np.asarray(actions[t], dtype=np.float32).tolist(),
                "next.done": bool(t == T - 1),
                "next.reward": float(1.0 if t == T - 1 else 0.0),
            })
        pd.DataFrame(rows).to_parquet(
            self.root / "data" / "chunk-000" / f"file-{ep_idx:03d}.parquet", index=False)

        self.episodes_meta.append({
            "episode_index": int(ep_idx),
            "tasks": [self.task_str],
            "length": int(T),
            "dataset_from_index": int(self.global_index),
            "dataset_to_index": int(self.global_index + T),
            **{
                f"videos/observation.images.{cam}/chunk_index": 0
                for cam in CAMS
            },
            **{
                f"videos/observation.images.{cam}/file_index": ep_idx
                for cam in CAMS
            },
            **{
                f"videos/observation.images.{cam}/from_timestamp": float(timestamps[0])
                for cam in CAMS
            },
            **{
                f"videos/observation.images.{cam}/to_timestamp": float(timestamps[-1])
                for cam in CAMS
            },
        })
        pd.DataFrame(self.episodes_meta).to_parquet(self.eps_parquet, index=False)

        self.global_index += T
        self.all_actions.append(np.array(actions, dtype=np.float32))
        self.all_states.append(np.array(states, dtype=np.float32))
        self._write_meta()

    def _write_meta(self):
        all_a = np.vstack(self.all_actions)
        all_s = np.vstack(self.all_states)

        def stat(arr):
            return {"mean": arr.mean(0).tolist(), "std": arr.std(0).tolist(),
                    "min": arr.min(0).tolist(), "max": arr.max(0).tolist()}

        img_stat = {
            "mean": [[[0.485]], [[0.456]], [[0.406]]],
            "std": [[[0.229]], [[0.224]], [[0.225]]],
            "min": [[[0.0]], [[0.0]], [[0.0]]],
            "max": [[[1.0]], [[1.0]], [[1.0]]],
        }
        stats = {
            "action": stat(all_a),
            "observation.state": stat(all_s),
            **{f"observation.images.{cam}": img_stat for cam in CAMS},
        }
        with open(self.root / "meta" / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        n_ep = self.n_episodes
        info = {
            "codebase_version": "v3.0",
            "robot_type": "franka",
            "total_episodes": n_ep,
            "total_frames": int(self.global_index),
            "total_tasks": 1,
            "fps": self.fps,
            "splits": {"train": f"0:{n_ep}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {
                "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": ACTION_NAMES},
                "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": STATE_NAMES},
                **{
                    f"observation.images.{cam}": {
                        "dtype": "video", "shape": [IMG_SIZE, IMG_SIZE, 3],
                        "names": ["height", "width", "channel"],
                        "video_info": {"video.fps": float(self.fps), "video.codec": "h264",
                                       "video.pix_fmt": "yuv420p",
                                       "video.is_depth_map": False, "has_audio": False}}
                    for cam in CAMS
                },
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
                "next.done": {"dtype": "bool", "shape": [1], "names": None},
                "next.reward": {"dtype": "float32", "shape": [1], "names": None},
            },
        }
        with open(self.root / "meta" / "info.json", "w") as f:
            json.dump(info, f, indent=2)
        pd.DataFrame([{"task_index": 0, "task": self.task_str}]).to_parquet(
            self.root / "meta" / "tasks.parquet", index=False)

    def print_summary(self):
        print(f"\n  dataset: {self.root}")
        print(f"  episodes={self.n_episodes}  frames={self.global_index}")


def copy_dataset_shell(src_root, dst_root):
    """复制一份完整数据集（用于某些D只改meta不改视频/数据的场景，目前未使用，留作工具函数）"""
    shutil.copytree(src_root, dst_root, dirs_exist_ok=True)
