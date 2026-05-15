"""
teleop_collision_lerobot.py —— 实验04：弹性碰撞 LeRobot v3 数据集采集脚本
================================================================================

用途：
    在不改变现有 teleop_collision.py 物理实验建模与运动逻辑的前提下，
    直接把手动示教采集结果保存为 LeRobot v3 风格数据集：

    demos_collision_lerobot/
    ├── meta/
    │   ├── info.json
    │   ├── stats.json
    │   ├── tasks.parquet
    │   ├── episodes/chunk-000/file-000.parquet
    │   └── collision_metrics/chunk-000/file-000.parquet
    ├── data/chunk-000/file-NNN.parquet
    └── videos/
        ├── observation.images.cam_front/chunk-000/file-NNN.mp4
        ├── observation.images.cam_side/chunk-000/file-NNN.mp4
        └── observation.images.cam_top/chunk-000/file-NNN.mp4

重要原则：
    1. 本脚本不重新写弹性碰撞物理模型。
    2. 本脚本直接 import 同目录下的 teleop_collision.py，并复用其中的：
       build_model / collect_ids / ik_step / poll_movement / 辅助夹持 / 斜坡约束 /
       一维解析碰撞 / current_quantities / make_state_row 等函数。
    3. 因此，请把本文件和已经调好的 teleop_collision.py 放在同一个目录下运行。

State 12 维：
    [finger1, finger2, j0, j1, j2, j3, j4, j5, j6, ball1_vx, ball2_vx, time]

Action 8 维：
    [j0_ctrl, j1_ctrl, j2_ctrl, j3_ctrl, j4_ctrl, j5_ctrl, j6_ctrl, gripper_ctrl]

用法：
    python teleop_collision_lerobot.py --n 10
    python teleop_collision_lerobot.py --n 20 --m1 0.10 --m2 0.10 --v 0.30 --distance 0.34
    python teleop_collision_lerobot.py --out D:/mujoco/demos_collision_lerobot --n 10
    python teleop_collision_lerobot.py --n 5 --m1 0.10 --m2 0.05 --img-size 256 --out D:\mujoco\demos_collision_lerobot_256

按键：
    1/2   末端 +X / -X
    3/4   末端 -Y / +Y
    5/6   末端 +Z / -Z
    7     夹爪开合；闭合时辅助夹住红球，打开时释放
    8     开始 / 停止录制当前条
    9     保存当前条到 LeRobot v3 数据集
    0     放弃当前条并重置
    ESC   退出

依赖：
    pip install mujoco keyboard pandas imageio imageio-ffmpeg pyarrow
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd

# 必须与本文件在同一目录，且是你已经调好的稳定版本。
import teleop_collision as sim

try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False
    print("⚠️  未找到 keyboard，请运行: pip install keyboard")


# ── LeRobot 数据集常量 ───────────────────────────────────────────────────────
IMG_SIZE = 256                  # 默认从 96 提升到 256，便于人工检查轨迹；可用 --img-size 覆盖
FPS = 15
MIN_FRAMES = 30
STATE_DIM = 12
ACTION_DIM = 8
TASK_STR = "grasp red ball, release it on the ramp, and perform one-dimensional elastic collision with blue ball"
CAMERAS = ["cam_front", "cam_side", "cam_top"]
VIDEO_CODEC = "libx264"
VIDEO_PIX_FMT = "yuv420p"
VIDEO_CRF = 18                    # H.264 质量参数：越小越清晰、文件越大；18 通常清晰
VIDEO_PRESET = "medium"           # 编码速度/压缩效率折中

STATE_NAMES = [
    "finger1", "finger2",
    "j0", "j1", "j2", "j3", "j4", "j5", "j6",
    "ball1_vx", "ball2_vx", "time",
]
ACTION_NAMES = ["j0", "j1", "j2", "j3", "j4", "j5", "j6", "gripper"]


# ── 检查当前 teleop_collision.py 是否是稳定斜坡版 ───────────────────────────
def _require_collision_api() -> None:
    required = [
        "build_model", "collect_ids", "ik_step", "poll_movement",
        "set_ball1_world", "apply_ramp_constraint", "clamp_ball2",
        "maybe_attach_ball", "update_attached_ball", "snap_release_to_ramp",
        "current_quantities", "get_grasp_center",
    ]
    missing = [name for name in required if not hasattr(sim, name)]
    if missing:
        raise RuntimeError(
            "当前目录下的 teleop_collision.py 不是稳定斜坡发射版，缺少函数："
            + ", ".join(missing)
            + "\n请把已经调好的 teleop_collision.py 与本脚本放在同一目录下。"
        )


def _make_state_row(model, data, ids, v1: float, v2: float) -> List[float]:
    """兼容不同版本 teleop_collision.py 的 make_state_row 签名。"""
    if not hasattr(sim, "make_state_row"):
        finger1 = float(data.qpos[7]) if data.qpos.shape[0] > 7 else 0.0
        finger2 = float(data.qpos[8]) if data.qpos.shape[0] > 8 else 0.0
        return [finger1, finger2] + data.qpos[:7].tolist() + [float(v1), float(v2), float(data.time)]
    try:
        return list(sim.make_state_row(data, ids, v1, v2))
    except TypeError:
        return list(sim.make_state_row(model, data, v1, v2))


# ── 图片统计缓存：用真实仿真图像计算 mean/std，而不是 ImageNet 默认值 ───────
class RunningImageStats:
    def __init__(self, root: Path):
        self.cache_path = root / "meta" / "image_stats_cache.json"
        self.stats: Dict[str, Dict[str, Any]] = {}
        for cam in CAMERAS:
            self.stats[cam] = {
                "sum": [0.0, 0.0, 0.0],
                "sum_sq": [0.0, 0.0, 0.0],
                "count": 0,
            }
        if self.cache_path.exists():
            with self.cache_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            for cam in CAMERAS:
                if cam in loaded:
                    self.stats[cam] = loaded[cam]

    def update(self, cam: str, frames: List[np.ndarray]) -> None:
        if not frames:
            return
        arr = np.asarray(frames, dtype=np.float32) / 255.0  # T,H,W,C, RGB
        flat = arr.reshape(-1, 3)
        st = self.stats[cam]
        st["sum"] = (np.asarray(st["sum"], dtype=np.float64) + flat.sum(axis=0)).tolist()
        st["sum_sq"] = (np.asarray(st["sum_sq"], dtype=np.float64) + (flat ** 2).sum(axis=0)).tolist()
        st["count"] = int(st.get("count", 0) + flat.shape[0])

    def to_feature_stats(self, cam: str) -> Dict[str, Any]:
        st = self.stats[cam]
        count = int(st.get("count", 0))
        if count <= 0:
            # 兜底值；正常采集一条后会被真实图像统计替代。
            mean = np.array([0.5, 0.5, 0.5], dtype=np.float64)
            std = np.array([0.25, 0.25, 0.25], dtype=np.float64)
        else:
            s = np.asarray(st["sum"], dtype=np.float64)
            ss = np.asarray(st["sum_sq"], dtype=np.float64)
            mean = s / count
            var = np.maximum(ss / count - mean ** 2, 1e-8)
            std = np.sqrt(var)
        return {
            "mean": [[[float(mean[0])]], [[float(mean[1])]], [[float(mean[2])]]],
            "std":  [[[float(std[0])]],  [[float(std[1])]],  [[float(std[2])]]],
            "min":  [[[0.0]], [[0.0]], [[0.0]]],
            "max":  [[[1.0]], [[1.0]], [[1.0]]],
        }

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)


# ── LeRobot v3 写入器 ────────────────────────────────────────────────────────
class LeRobotCollisionWriter:
    def __init__(self, out_dir: str | Path):
        self.root = Path(out_dir)
        self.root.mkdir(parents=True, exist_ok=True)

        (self.root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (self.root / "meta" / "collision_metrics" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        for cam in CAMERAS:
            (self.root / "videos" / f"observation.images.{cam}" / "chunk-000").mkdir(parents=True, exist_ok=True)

        self.eps_parquet = self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        self.metrics_parquet = self.root / "meta" / "collision_metrics" / "chunk-000" / "file-000.parquet"
        self.episodes_meta: List[Dict[str, Any]] = []
        self.collision_metrics_meta: List[Dict[str, Any]] = []
        self.global_index = 0

        if self.eps_parquet.exists():
            df = pd.read_parquet(self.eps_parquet)
            self.episodes_meta = df.to_dict("records")
            data_files = sorted((self.root / "data" / "chunk-000").glob("file-*.parquet"))
            if data_files:
                last_df = pd.read_parquet(data_files[-1])
                self.global_index = int(last_df["index"].iloc[-1]) + 1

        if self.metrics_parquet.exists():
            self.collision_metrics_meta = pd.read_parquet(self.metrics_parquet).to_dict("records")

        self.all_actions: List[np.ndarray] = []
        self.all_states: List[np.ndarray] = []
        for f in sorted((self.root / "data" / "chunk-000").glob("file-*.parquet")):
            df = pd.read_parquet(f)
            self.all_actions.append(np.asarray(df["action"].tolist(), dtype=np.float32))
            self.all_states.append(np.asarray(df["observation.state"].tolist(), dtype=np.float32))

        self.image_stats = RunningImageStats(self.root)

        # 防止把 96×96 的旧数据集和 256×256 的新数据混在同一目录里。
        info_path = self.root / "meta" / "info.json"
        if info_path.exists() and self.n_episodes > 0:
            try:
                with info_path.open("r", encoding="utf-8") as f:
                    old_info = json.load(f)
                old_shape = old_info.get("features", {}).get("observation.images.cam_front", {}).get("shape", None)
                if old_shape and (int(old_shape[0]) != int(IMG_SIZE) or int(old_shape[1]) != int(IMG_SIZE)):
                    raise RuntimeError(
                        f"当前输出目录已有 {old_shape[0]}×{old_shape[1]} 视频数据，"
                        f"但本次设置为 {IMG_SIZE}×{IMG_SIZE}。\n"
                        "请使用相同 --img-size 继续追加，或者换一个新的 --out 目录；"
                        "如果旧数据不要了，也可以删除原目录后重新采集。"
                    )
            except RuntimeError:
                raise
            except Exception as exc:
                print(f"⚠️  读取已有 info.json 检查分辨率失败：{exc}")

    @property
    def n_episodes(self) -> int:
        return len(self.episodes_meta)

    def _video_path(self, cam_name: str, ep_idx: int) -> Path:
        return self.root / "videos" / f"observation.images.{cam_name}" / "chunk-000" / f"file-{ep_idx:03d}.mp4"

    def save_episode(
        self,
        frames_front: List[np.ndarray],
        frames_side: List[np.ndarray],
        frames_top: List[np.ndarray],
        states: np.ndarray,
        actions: np.ndarray,
        metrics: np.ndarray,
        params: Dict[str, Any],
    ) -> int:
        ep_idx = self.n_episodes
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        metrics = np.asarray(metrics, dtype=np.float32)
        T = int(actions.shape[0])

        if states.shape != (T, STATE_DIM):
            raise ValueError(f"states shape 应为 ({T},{STATE_DIM})，实际 {states.shape}")
        if actions.shape != (T, ACTION_DIM):
            raise ValueError(f"actions shape 应为 ({T},{ACTION_DIM})，实际 {actions.shape}")
        if not (len(frames_front) == len(frames_side) == len(frames_top) == T):
            raise ValueError(
                f"视频帧数与 actions 不一致：front={len(frames_front)}, side={len(frames_side)}, "
                f"top={len(frames_top)}, actions={T}"
            )

        # 写三路视频，帧是 RGB uint8。
        frame_sets = {
            "cam_front": frames_front,
            "cam_side": frames_side,
            "cam_top": frames_top,
        }
        for cam_name, frames in frame_sets.items():
            vid_path = self._video_path(cam_name, ep_idx)
            writer = imageio.get_writer(
                str(vid_path),
                format="ffmpeg",
                fps=FPS,
                codec=VIDEO_CODEC,
                output_params=["-pix_fmt", VIDEO_PIX_FMT, "-crf", str(VIDEO_CRF), "-preset", VIDEO_PRESET],
            )
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            self.image_stats.update(cam_name, frames)

        timestamps = np.arange(T, dtype=np.float32) / float(FPS)
        rows = []
        for t in range(T):
            rows.append({
                "timestamp": float(timestamps[t]),
                "frame_index": int(t),
                "episode_index": int(ep_idx),
                "index": int(self.global_index + t),
                "task_index": int(0),
                "observation.state": states[t].tolist(),
                "action": actions[t].tolist(),
                "next.done": bool(t == T - 1),
                "next.reward": float(1.0 if t == T - 1 else 0.0),
            })
        pd.DataFrame(rows).to_parquet(
            self.root / "data" / "chunk-000" / f"file-{ep_idx:03d}.parquet",
            index=False,
        )

        summary = summarize_collision_metrics(metrics, params)

        ep_meta = {
            "episode_index": int(ep_idx),
            "tasks": [TASK_STR],
            "length": int(T),
            "dataset_from_index": int(self.global_index),
            "dataset_to_index": int(self.global_index + T),
            "videos/observation.images.cam_front/chunk_index": 0,
            "videos/observation.images.cam_front/file_index": int(ep_idx),
            "videos/observation.images.cam_front/from_timestamp": float(timestamps[0]),
            "videos/observation.images.cam_front/to_timestamp": float(timestamps[-1]),
            "videos/observation.images.cam_side/chunk_index": 0,
            "videos/observation.images.cam_side/file_index": int(ep_idx),
            "videos/observation.images.cam_side/from_timestamp": float(timestamps[0]),
            "videos/observation.images.cam_side/to_timestamp": float(timestamps[-1]),
            "videos/observation.images.cam_top/chunk_index": 0,
            "videos/observation.images.cam_top/file_index": int(ep_idx),
            "videos/observation.images.cam_top/from_timestamp": float(timestamps[0]),
            "videos/observation.images.cam_top/to_timestamp": float(timestamps[-1]),
            # 下面是弹性碰撞实验专属元信息，便于后续筛除失败轨迹。
            "m1": float(params.get("m1", 0.0)),
            "m2": float(params.get("m2", 0.0)),
            "v_target": float(params.get("v_target", 0.0)),
            "distance": float(params.get("distance", 0.0)),
            "collision_success": bool(summary.get("success", False)),
            "min_gap": float(summary.get("min_gap", 0.0)),
            "max_v1_abs": float(summary.get("max_v1_abs", 0.0)),
            "max_v2_abs": float(summary.get("max_v2_abs", 0.0)),
            "momentum_error_pct": float(summary.get("momentum_error_pct", 0.0)),
            "energy_error_pct": float(summary.get("energy_error_pct", 0.0)),
        }
        self.episodes_meta.append(ep_meta)
        pd.DataFrame(self.episodes_meta).to_parquet(self.eps_parquet, index=False)

        metric_row = {"episode_index": int(ep_idx), **params, **summary}
        self.collision_metrics_meta.append(metric_row)
        pd.DataFrame(self.collision_metrics_meta).to_parquet(self.metrics_parquet, index=False)

        self.global_index += T
        self.all_actions.append(actions)
        self.all_states.append(states)
        self.image_stats.save()
        self._write_meta()
        return T

    def _write_meta(self) -> None:
        if self.all_actions:
            all_a = np.vstack(self.all_actions).astype(np.float32)
            all_s = np.vstack(self.all_states).astype(np.float32)
        else:
            all_a = np.zeros((1, ACTION_DIM), dtype=np.float32)
            all_s = np.zeros((1, STATE_DIM), dtype=np.float32)

        def stat(arr: np.ndarray) -> Dict[str, Any]:
            return {
                "mean": arr.mean(axis=0).tolist(),
                "std": (arr.std(axis=0) + 1e-8).tolist(),
                "min": arr.min(axis=0).tolist(),
                "max": arr.max(axis=0).tolist(),
            }

        stats = {
            "action": stat(all_a),
            "observation.state": stat(all_s),
        }
        for cam in CAMERAS:
            stats[f"observation.images.{cam}"] = self.image_stats.to_feature_stats(cam)

        with (self.root / "meta" / "stats.json").open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        n_ep = self.n_episodes
        total_frames = int(self.global_index)
        info = {
            "codebase_version": "v3.0",
            "robot_type": "franka",
            "total_episodes": int(n_ep),
            "total_frames": total_frames,
            "total_tasks": 1,
            "fps": int(FPS),
            "splits": {"train": f"0:{n_ep}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {
                "action": {
                    "dtype": "float32",
                    "shape": [ACTION_DIM],
                    "names": ACTION_NAMES,
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [STATE_DIM],
                    "names": STATE_NAMES,
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
        for cam in CAMERAS:
            info["features"][f"observation.images.{cam}"] = {
                "dtype": "video",
                "shape": [IMG_SIZE, IMG_SIZE, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": float(FPS),
                    "video.codec": "h264",
                    "video.pix_fmt": VIDEO_PIX_FMT,
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
        with (self.root / "meta" / "info.json").open("w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        pd.DataFrame([{"task_index": 0, "task": TASK_STR}]).to_parquet(
            self.root / "meta" / "tasks.parquet",
            index=False,
        )

    def print_summary(self) -> None:
        print(f"\n  dataset: {self.root.resolve()}")
        print(f"  episodes={self.n_episodes}  frames={self.global_index}")


# ── 碰撞指标摘要，用于后续筛除失败样本 ─────────────────────────────────────
def _robust_mean(x: np.ndarray, default: float = 0.0) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(default)
    if x.size < 5:
        return float(x.mean())
    return float(np.median(x))


def summarize_collision_metrics(metrics: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    """metrics 列：[x1,z1,x2,v1,v2,gap,P,E,h,v_theory,collision_event]."""
    if metrics is None or len(metrics) == 0:
        return {"success": False, "reason": "empty metrics"}
    arr = np.asarray(metrics, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 8:
        return {"success": False, "reason": f"bad metrics shape {arr.shape}"}

    v1 = arr[:, 3]
    v2 = arr[:, 4]
    gap = arr[:, 5]
    p = arr[:, 6]
    e = arr[:, 7]
    event_col = arr[:, 10] if arr.shape[1] > 10 else np.zeros(arr.shape[0])

    event_idx = np.where(event_col > 0.5)[0]
    if len(event_idx) > 0:
        ci = int(event_idx[0])
    else:
        ci = int(np.argmin(gap))

    pre = slice(max(0, ci - 5), max(1, ci))
    post = slice(min(arr.shape[0] - 1, ci + 1), min(arr.shape[0], ci + 10))
    if post.start >= post.stop:
        post = slice(ci, arr.shape[0])

    p_before = _robust_mean(p[pre])
    p_after = _robust_mean(p[post])
    e_before = _robust_mean(e[pre])
    e_after = _robust_mean(e[post])

    p_err = abs(p_after - p_before) / max(abs(p_before), 1e-8) * 100.0
    e_err = abs(e_after - e_before) / max(abs(e_before), 1e-8) * 100.0
    min_gap = float(np.nanmin(gap))
    max_v1 = float(np.nanmax(np.abs(v1)))
    max_v2 = float(np.nanmax(np.abs(v2)))
    success = bool((min_gap <= 0.03) and (max_v1 >= 0.03) and (max_v2 >= 0.02))

    return {
        "success": success,
        "contact_index": ci,
        "min_gap": min_gap,
        "max_v1_abs": max_v1,
        "max_v2_abs": max_v2,
        "v1_before": _robust_mean(v1[pre]),
        "v2_before": _robust_mean(v2[pre]),
        "v1_after": _robust_mean(v1[post]),
        "v2_after": _robust_mean(v2[post]),
        "momentum_before": p_before,
        "momentum_after": p_after,
        "momentum_error_pct": float(p_err),
        "energy_before": e_before,
        "energy_after": e_after,
        "energy_error_pct": float(e_err),
        "collision_event_recorded": bool(len(event_idx) > 0),
    }


# ── 终端状态行工具 ──────────────────────────────────────────────────────────
def _char_width(ch: str) -> int:
    return 2 if unicodedata.east_asian_width(ch) in ("F", "W", "A") else 1


def _display_width(text: str) -> int:
    return sum(_char_width(ch) for ch in text)


def _truncate_to_columns(text: str, max_cols: int) -> str:
    out: List[str] = []
    used = 0
    for ch in text:
        w = _char_width(ch)
        if used + w > max_cols:
            break
        out.append(ch)
        used += w
    return "".join(out)


class StatusLine:
    def __init__(self):
        self.width = 0

    def clear(self) -> None:
        cols = shutil.get_terminal_size((120, 20)).columns
        sys.stdout.write("\r" + " " * max(1, cols - 1) + "\r")
        sys.stdout.flush()
        self.width = 0

    def event(self, msg: str) -> None:
        self.clear()
        print(msg, flush=True)

    def write(self, msg: str) -> None:
        cols = shutil.get_terminal_size((120, 20)).columns
        msg = _truncate_to_columns(msg, max(30, cols - 2))
        width = _display_width(msg)
        pad = " " * max(0, cols - 1 - width)
        sys.stdout.write("\r" + msg + pad + "\r" + msg)
        sys.stdout.flush()
        self.width = width


# ── 主采集流程 ──────────────────────────────────────────────────────────────
def record_session(
    *,
    m1: float,
    m2: float,
    v_target: float,
    distance: float,
    n_demos: int,
    writer: LeRobotCollisionWriter,
    status_interval: float,
) -> int:
    _require_collision_api()

    print("\n" + "=" * 78)
    print("  实验04：弹性碰撞 LeRobot v3 数据集采集")
    print(f"  m1={m1:.3f} kg  m2={m2:.3f} kg  v_target={v_target:.3f} m/s  distance={distance:.3f}")
    print("  物理建模与运动逻辑：完全复用同目录 teleop_collision.py")
    print(f"  输出数据集：{writer.root.resolve()}")
    print("=" * 78)
    print("  1/2 +X/-X   3/4 -Y/+Y   5/6 +Z/-Z")
    print("  7 夹爪开合/辅助夹持；8 开始/停止；9 保存当前条；0 放弃当前条；ESC 退出")
    print("=" * 78 + "\n")

    model, data, geom_info = sim.build_model(m1=m1, m2=m2, distance=distance)
    ids = sim.collect_ids(model)
    ball2_x0 = float(geom_info["ball2_x0"])

    hand_bid = ids.hand_bid
    home_q = np.array([0.0, 0.5, 0.0, -2.0, 0.0, 2.5, 0.785])
    record_every = max(1, int(round(1.0 / (FPS * model.opt.timestep))))

    id_front = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_front")
    id_side = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_side")
    id_top = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_top")
    if min(id_front, id_side, id_top) < 0:
        raise RuntimeError("模型中缺少 cam_front / cam_side / cam_top，相机名需与 teleop_collision.py 一致。")

    rf = mujoco.Renderer(model, IMG_SIZE, IMG_SIZE)
    rs = mujoco.Renderer(model, IMG_SIZE, IMG_SIZE)
    rt = mujoco.Renderer(model, IMG_SIZE, IMG_SIZE)

    status = StatusLine()

    def reset_sim() -> Tuple[np.ndarray, float, bool]:
        mujoco.mj_resetData(model, data)
        data.qpos[:7] = home_q.copy()
        data.ctrl[:7] = home_q.copy()
        if model.nu > 7:
            data.ctrl[7] = sim.GRIPPER_OPEN
        # 保持与 teleop_collision.py 相同的初始稳定状态。
        sim.set_ball1_world(data, ids, sim.BALL1_X0, sim.BALL_FLAT_Z, 0.0, 0.0)
        data.qpos[ids.ball2_x_qid] = 0.0
        data.qvel[ids.ball2_x_did] = 0.0
        mujoco.mj_forward(model, data)
        return data.xpos[hand_bid].copy(), sim.GRIPPER_OPEN, False

    def render(renderer: mujoco.Renderer, cam_id: int) -> np.ndarray:
        renderer.update_scene(data, camera=cam_id)
        return renderer.render().copy()  # RGB uint8

    ee_target, gripper_val, attached = reset_sim()
    is_recording = False
    ep_done = 0
    step_count = 0
    last_status_t = 0.0
    last_collision_step = -10_000
    last_collision_t = -10_000.0

    frames_front: List[np.ndarray] = []
    frames_side: List[np.ndarray] = []
    frames_top: List[np.ndarray] = []
    states: List[List[float]] = []
    actions: List[List[float]] = []
    metrics: List[List[float]] = []

    prev_7 = prev_8 = prev_9 = prev_0 = prev_esc = False

    def clear_buffers() -> None:
        frames_front.clear()
        frames_side.clear()
        frames_top.clear()
        states.clear()
        actions.clear()
        metrics.clear()

    def key_callback(keycode):
        nonlocal ee_target
        if HAS_KEYBOARD:
            return
        step = 0.02
        mapping = {
            ord("1"): np.array([ step, 0, 0]),
            ord("2"): np.array([-step, 0, 0]),
            ord("3"): np.array([0, -step, 0]),
            ord("4"): np.array([0,  step, 0]),
            ord("5"): np.array([0, 0,  step]),
            ord("6"): np.array([0, 0, -step]),
        }
        if keycode in mapping:
            ee_target = ee_target + mapping[keycode]

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            viewer.cam.distance = 2.6
            viewer.cam.elevation = -22
            viewer.cam.azimuth = 145
            prev_time = time.perf_counter()

            while viewer.is_running() and ep_done < n_demos:
                now = time.perf_counter()
                dt = min(now - prev_time, 0.05)
                prev_time = now
                ee_target += sim.poll_movement(dt)

                cur_esc = HAS_KEYBOARD and keyboard.is_pressed("esc")
                if cur_esc and not prev_esc:
                    status.event("❌ 退出采集。")
                    break
                prev_esc = cur_esc

                cur_7 = HAS_KEYBOARD and keyboard.is_pressed("7")
                if cur_7 and not prev_7:
                    if gripper_val == sim.GRIPPER_OPEN:
                        gripper_val = sim.GRIPPER_CLOSE
                        can_attach, attach_dist = sim.maybe_attach_ball(model, data, ids, ee_target=ee_target)
                        if can_attach or getattr(sim, "FORCE_ATTACH_ON_CLOSE", True):
                            attached = True
                            sim.update_attached_ball(model, data, ids)
                            status.event(f"夹爪: 闭合，红球 ATTACHED（距离={attach_dist:.3f}m）")
                        else:
                            attached = False
                            status.event(f"夹爪: 闭合，但距离红球较远（距离={attach_dist:.3f}m），未夹住")
                    else:
                        gripper_val = sim.GRIPPER_OPEN
                        if attached:
                            attached = False
                            sim.snap_release_to_ramp(data, ids)
                            mujoco.mj_forward(model, data)
                            status.event("夹爪: 打开，红球 RELEASED")
                        else:
                            status.event("夹爪: 打开")
                prev_7 = cur_7

                if attached:
                    sim.update_attached_ball(model, data, ids)

                cur_8 = HAS_KEYBOARD and keyboard.is_pressed("8")
                if cur_8 and not prev_8:
                    is_recording = not is_recording
                    if is_recording:
                        clear_buffers()
                        status.event(f"▶ 开始录制第 {ep_done + 1}/{n_demos} 条...")
                    else:
                        status.event(f"■ 暂停，已录 {len(actions)} 帧（9保存 / 0放弃）")
                prev_8 = cur_8

                cur_9 = HAS_KEYBOARD and keyboard.is_pressed("9")
                if cur_9 and not prev_9:
                    if len(actions) >= MIN_FRAMES:
                        is_recording = False
                        params = {
                            "m1": float(m1),
                            "m2": float(m2),
                            "v_target": float(v_target),
                            "distance": float(distance),
                            "ball_radius": float(sim.BALL_RADIUS),
                            "ball1_x0": float(sim.BALL1_X0),
                            "ball2_x0": float(ball2_x0),
                            "ramp_top_x": float(sim.RAMP_TOP_X),
                            "ramp_bottom_x": float(sim.RAMP_BOTTOM_X),
                            "ramp_height": float(sim.RAMP_HEIGHT),
                            "ramp_angle_deg": float(np.degrees(sim.RAMP_ANGLE_RAD)),
                            "collision_restitution": float(getattr(sim, "COLLISION_RESTITUTION", 1.0)),
                            "analytic_collision": bool(getattr(sim, "USE_ANALYTIC_COLLISION", False)),
                            "max_ramp_speed": float(getattr(sim, "MAX_RAMP_SPEED", 0.0)),
                            "max_ball_speed": float(getattr(sim, "MAX_BALL_SPEED", 0.0)),
                        }
                        T = writer.save_episode(
                            frames_front,
                            frames_side,
                            frames_top,
                            np.asarray(states, dtype=np.float32),
                            np.asarray(actions, dtype=np.float32),
                            np.asarray(metrics, dtype=np.float32),
                            params,
                        )
                        ep_done += 1
                        status.event(f"✅ 第 {ep_done}/{n_demos} 条已保存到 LeRobot v3（{T} 帧）")
                        writer.print_summary()
                        ee_target, gripper_val, attached = reset_sim()
                        clear_buffers()
                        if ep_done < n_demos:
                            status.event(f"→ 准备第 {ep_done + 1}/{n_demos} 条")
                    else:
                        status.event(f"⚠️  仅 {len(actions)} 帧，需 >= {MIN_FRAMES} 帧；继续录制或按 0 放弃。")
                prev_9 = cur_9

                cur_0 = HAS_KEYBOARD and keyboard.is_pressed("0")
                if cur_0 and not prev_0:
                    is_recording = False
                    clear_buffers()
                    ee_target, gripper_val, attached = reset_sim()
                    status.event(f"🔄 已放弃，重录第 {ep_done + 1}/{n_demos} 条")
                prev_0 = cur_0

                # ── 以下控制/物理更新逻辑与 teleop_collision.py 保持一致 ─────
                sim.ik_step(model, data, ee_target)
                if model.nu > 7:
                    data.ctrl[7] = gripper_val

                mujoco.mj_step(model, data)
                step_count += 1

                collided = False
                if attached:
                    sim.update_attached_ball(model, data, ids)
                else:
                    sim.apply_ramp_constraint(model, data, ids, attached=False, dt=model.opt.timestep)
                    if hasattr(sim, "resolve_analytic_collision"):
                        last_collision_step, collided = sim.resolve_analytic_collision(
                            model, data, ids, m1, m2, ball2_x0, step_count, last_collision_step
                        )
                        if collided:
                            last_collision_t = now

                sim.clamp_ball2(data, ids, ball2_x0)
                mujoco.mj_forward(model, data)

                x1, z1, x2, v1, v2, gap, p, energy, height, v_theory = sim.current_quantities(
                    data, ids, m1, m2, ball2_x0
                )

                if is_recording and step_count % record_every == 0:
                    frames_front.append(render(rf, id_front))
                    frames_side.append(render(rs, id_side))
                    frames_top.append(render(rt, id_top))
                    states.append(_make_state_row(model, data, ids, v1, v2))
                    actions.append(data.ctrl[:8].tolist())
                    # metrics: [x1,z1,x2,v1,v2,gap,P,E,h,v_theory,collision_event]
                    metrics.append([
                        float(x1), float(z1), float(x2), float(v1), float(v2),
                        float(gap), float(p), float(energy), float(height),
                        float(v_theory), 1.0 if collided else 0.0,
                    ])

                viewer.sync()

                if now - last_status_t >= status_interval:
                    last_status_t = now
                    gc = sim.get_grasp_center(model, data, ids)
                    attach_flag = "ATTACHED" if attached else "free"
                    rec_flag = "REC" if is_recording else "idle"
                    col_flag = "COL" if (now - last_collision_t) < 0.7 else ""
                    msg = (
                        f"夹={'闭' if gripper_val == sim.GRIPPER_CLOSE else '开'} {attach_flag} {rec_flag} {col_flag} "
                        f"{len(actions)}帧 {ep_done}/{n_demos}条 | "
                        f"g=({gc[0]:.2f},{gc[2]:.2f}) x1={x1:.2f} z1={z1:.2f} x2={x2:.2f} "
                        f"gap={gap:.2f} v1={v1:+.2f} v2={v2:+.2f} h={height:.2f} v理≈{v_theory:.2f}"
                    )
                    status.write(msg)

    finally:
        try:
            status.clear()
        except Exception:
            pass
        rf.close()
        rs.close()
        rt.close()

    return ep_done


# ── CLI ─────────────────────────────────────────────────────────────────────
def main() -> None:
    global IMG_SIZE, FPS, VIDEO_CRF
    ap = argparse.ArgumentParser(description="实验04：弹性碰撞 LeRobot v3 数据集采集脚本")
    ap.add_argument("--m1", type=float, default=0.10, help="红球质量 kg")
    ap.add_argument("--m2", type=float, default=0.10, help="蓝球质量 kg")
    ap.add_argument("--v", type=float, default=0.30, help="目标碰前速度，仅作元数据记录")
    ap.add_argument("--distance", type=float, default=0.34, help="红蓝球初始中心距")
    ap.add_argument("--n", type=int, default=10, help="本次计划采集 episode 数")
    ap.add_argument("--out", type=str, default=r"D:\mujoco\demos_collision_lerobot", help="LeRobot v3 数据集输出目录")
    ap.add_argument("--status-interval", type=float, default=getattr(sim, "STATUS_INTERVAL", 0.10), help="终端状态刷新间隔秒")
    ap.add_argument("--img-size", type=int, default=256, help="LeRobot 视频分辨率，默认 256；原组员模板是 96，但人工检查会很糊")
    ap.add_argument("--video-crf", type=int, default=18, help="H.264 清晰度参数，越小越清晰/越大；建议 16-23，默认 18")
    ap.add_argument("--video-preset", type=str, default="medium", help="H.264 编码 preset，如 ultrafast/fast/medium/slow")
    ap.add_argument("--franka-dir", type=str, default=None, help="可选：覆盖 teleop_collision.FRANKA_DIR")
    args = ap.parse_args()

    global IMG_SIZE, VIDEO_CRF, VIDEO_PRESET
    if args.img_size <= 0:
        raise ValueError("--img-size 必须为正整数")
    IMG_SIZE = int(args.img_size)
    VIDEO_CRF = int(args.video_crf)
    VIDEO_PRESET = str(args.video_preset)
    if IMG_SIZE < 160:
        print(f"⚠️  当前 --img-size={IMG_SIZE}，人工查看可能仍然偏糊；建议至少 256。")
    if IMG_SIZE % 16 != 0:
        print(f"⚠️  --img-size={IMG_SIZE} 不是 16 的倍数，ffmpeg 可能会自动补边或重采样；建议 256/320/480。")

    if args.franka_dir:
        sim.FRANKA_DIR = args.franka_dir

    writer = LeRobotCollisionWriter(args.out)
    print(f"dataset: {Path(args.out).resolve()}")
    print(f"已有 {writer.n_episodes} 条，本次计划录 {args.n} 条，采样率 {FPS}Hz，图像 {IMG_SIZE}×{IMG_SIZE}")

    saved = record_session(
        m1=args.m1,
        m2=args.m2,
        v_target=args.v,
        distance=args.distance,
        n_demos=args.n,
        writer=writer,
        status_interval=args.status_interval,
    )
    print(f"\n本次保存 {saved}/{args.n} 条")
    writer.print_summary()


if __name__ == "__main__":
    main()
