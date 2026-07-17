# -*- coding: utf-8 -*-
"""
compute_smoothness_metrics.py —— 平滑度候选指标计算
=====================================================================
对应文档《四、平滑度候选指标》：

    以下 4 个候选指标全部计算，只在 D0、D4、D5 这三组「未被抽帧稀释」的
    数据上算（每条 demo 算一次，取组内均值），不在 D1/D2/D3 上计算。

    1. 路径效率比：ee_xyz 实际路径长度（逐帧欧氏距离累加）÷ 起止点直线距离，
       越接近 1 越平滑高效，反映「空间上走不走弯路」
    2. Jerk 均方值：对 j0~j6 关节角序列做三阶差分（位置→速度→加速度→jerk），
       按时间间隔归一化后算均方，反映「时间上动作顺不顺」
    3. SPARC（谱弧长）：对 ee_xyz 速度模长做 FFT，在幅度谱上算弧长，
       数值越接近 0 越平滑，是运动科学里常用的成熟指标
    4. 速度局部极值计数：对 ee_xyz 速度模长曲线用 scipy.signal.find_peaks
       数局部极大值个数，极值越多轨迹越「抖」，计算最简单但对噪声敏感

用法（可在无 MuJoCo 的环境运行，只依赖已录制好的 D0/D4/D5 数据集）：
    python compute_smoothness_metrics.py \
        --d0 /path/to/D0 --d4 /path/to/D4 --d5 /path/to/D5 \
        --out smoothness_metrics.csv --fps 15
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from lerobot_io import LeRobotDataset

# state 维度索引: [j0..j6(0-6), gripper(7), block_xyz(8-10), ee_xyz(11-13)]
IDX_JOINTS = slice(0, 7)
IDX_EE = slice(11, 14)


def path_efficiency_ratio(ee_xyz):
    """实际路径长度 / 起止点直线距离。ee_xyz: [T,3]"""
    diffs = np.diff(ee_xyz, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    path_len = seg_lens.sum()
    straight_len = np.linalg.norm(ee_xyz[-1] - ee_xyz[0])
    if straight_len < 1e-9:
        return np.nan  # 起止点重合，比值无意义
    return float(path_len / straight_len)


def jerk_mean_square(joint_traj, dt):
    """
    joint_traj: [T,7] 关节角序列
    三阶差分：位置->速度->加速度->jerk，按时间间隔归一化，再算均方（对7个关节和所有时刻取平均）
    """
    vel = np.diff(joint_traj, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    if jerk.shape[0] == 0:
        return np.nan
    return float(np.mean(jerk ** 2))


def sparc(speed, fs, padlevel=4, fc=10.0, amp_th=0.05):
    """
    谱弧长 SPARC（Spectral Arc Length），标准实现（Balasubramanian et al.）。
    speed: 一维速度模长序列
    fs: 采样率(Hz)
    返回负值，越接近0越平滑；数值越负越不平滑。
    """
    if len(speed) < 3:
        return np.nan
    N = len(speed)
    Nfft = int(2 ** (np.ceil(np.log2(N)) + padlevel))
    freq = np.arange(0, Nfft) * fs / Nfft
    Mf = np.abs(np.fft.fft(speed, Nfft))
    Mf = Mf / (np.max(Mf) + 1e-12)

    # 只取到截止频率 fc 以内、且幅值超过阈值 amp_th 的部分（标准SPARC做法）
    fc_idx = int(np.searchsorted(freq, fc))
    if fc_idx < 2:
        fc_idx = min(2, len(freq))
    freq_sel = freq[:fc_idx]
    Mf_sel = Mf[:fc_idx]

    above_th = np.where(Mf_sel >= amp_th)[0]
    if len(above_th) == 0:
        return np.nan
    inx = range(above_th[0], above_th[-1] + 1)
    freq_band = freq_sel[inx]
    Mf_band = Mf_sel[inx]
    if len(freq_band) < 2:
        return np.nan

    d_freq = np.diff(freq_band) / (freq_band[-1] - freq_band[0] + 1e-12)
    d_mf = np.diff(Mf_band)
    arc_length = -np.sum(np.sqrt(d_freq ** 2 + d_mf ** 2))
    return float(arc_length)


def velocity_local_extrema_count(speed):
    """ee_xyz速度模长曲线上的局部极大值个数"""
    if len(speed) < 3:
        return 0
    peaks, _ = find_peaks(speed)
    return int(len(peaks))


def compute_episode_metrics(states, fps):
    ee_xyz = states[:, IDX_EE]
    joints = states[:, IDX_JOINTS]
    dt = 1.0 / fps

    ee_vel = np.diff(ee_xyz, axis=0) / dt          # [T-1, 3]
    ee_speed = np.linalg.norm(ee_vel, axis=1)       # [T-1]

    return {
        "path_efficiency_ratio": path_efficiency_ratio(ee_xyz),
        "jerk_mean_square": jerk_mean_square(joints, dt),
        "sparc": sparc(ee_speed, fs=fps),
        "velocity_local_extrema_count": velocity_local_extrema_count(ee_speed),
    }


def compute_dataset_metrics(root, dataset_name, fps):
    ds = LeRobotDataset(root)
    rows = []
    for ep_idx in ds.all_episode_indices():
        states, actions, meta = ds.load_episode(ep_idx)
        m = compute_episode_metrics(states, fps)
        m["dataset"] = dataset_name
        m["episode_index"] = ep_idx
        m["n_frames"] = len(actions)
        rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d0", type=str, required=True)
    ap.add_argument("--d4", type=str, required=True)
    ap.add_argument("--d5", type=str, required=True)
    ap.add_argument("--out", type=str, default="smoothness_metrics.csv")
    ap.add_argument("--fps", type=float, default=15.0)
    args = ap.parse_args()

    all_rows = []
    all_rows += compute_dataset_metrics(args.d0, "D0", args.fps)
    all_rows += compute_dataset_metrics(args.d4, "D4", args.fps)
    all_rows += compute_dataset_metrics(args.d5, "D5", args.fps)

    df = pd.DataFrame(all_rows)
    cols = ["dataset", "episode_index", "n_frames",
            "path_efficiency_ratio", "jerk_mean_square", "sparc",
            "velocity_local_extrema_count"]
    df = df[cols]
    df.to_csv(args.out, index=False)
    print(f"逐episode指标已保存到: {args.out}")
    print(df.to_string(index=False))

    # 组内均值（文档要求：每条demo算一次，取组内均值）
    summary = df.groupby("dataset")[
        ["path_efficiency_ratio", "jerk_mean_square", "sparc", "velocity_local_extrema_count"]
    ].mean().reset_index()
    summary_path = str(Path(args.out).with_name(Path(args.out).stem + "_summary.csv"))
    summary.to_csv(summary_path, index=False)
    print(f"\n组内均值（D0/D4/D5 × 4指标）已保存到: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
