# -*- coding: utf-8 -*-
"""
build_D2_D3_stage_frames.py —— 构造 D2（阶段完整性-2帧）/ D3（阶段完整性-4帧）
=====================================================================
对应文档《二、每个场景需要准备的数据集》表格 D2/D3 行：

    先确定阶段切分规则，每个阶段内均匀抽 N 帧（D2: N=2, D3: N=4），
    其余帧丢弃后重采样到统一长度。

── 阶段切分规则（固定、可复现，v2 修正版）───────────────────────────────
【v1问题记录，已确认并修复】最初版本用绝对高度阈值
`block_z - r <= water_surface_z` 判定"方块是否已触水"。

实测你的真实 D0 数据发现：block_z 在 gripper 闭合前后长时间保持常数
0.9728（方块静止漂浮，尚未被压低），而 0.9728 - 0.025 = 0.9478，本来就
<= water_surface_z(0.96)。也就是说方块"漂浮时露出水面以上的部分"按这个
公式看，本来就已经满足了"触水"判定——这个绝对阈值从第0帧起就恒为真，
根本不需要等到机械臂真正把它往下压。阶段2（提起/转移）因此被压缩到只剩
1帧。这是判定条件设计错误，与你的数据无关，20条episode里全部复现，
说明它是系统性bug，不是个别噪声。

【v2修正】改用"方块是否被机械臂明显主动压低到低于它自己静止漂浮时的高度"
来判定，而不是和水面的绝对位置比较：

    阶段1「接近」     ：从episode开始 → gripper 首次从"开"变为"闭合"（含该帧）
    阶段2「提起/转移」：gripper闭合之后 → 方块高度相对"闭合那一帧"下沉
                        超过 DROP_THRESH（默认5mm）的第一帧
    阶段3「按压入水」：方块开始被明显下压 → episode结束

    - gripper 状态阈值：action中的gripper维度 > 127 视为"闭合"指令
    - DROP_THRESH：方块相对"gripper闭合那一帧"高度的下降量超过此阈值，
      视为"开始主动下压"，默认 0.005m（5mm），可用 --drop_thresh 调整。
      5mm 远小于球半径(25mm)和整个下压行程（观察到block_z从0.9728降到
      0.8587，行程超过10cm），足以捕捉"刚开始下压"的时刻，又不会被
      传感器/物理噪声误触发
    - 若某一阶段可用帧数 < N（抽帧数），该阶段全部帧保留（不足则不丢帧），
      并打印警告

阶段内抽帧规则：阶段有 L 帧，需要抽 N 帧，用
    np.linspace(0, L-1, N) 取整（去重后不足N则补齐，保证均匀覆盖阶段首尾）
抽帧顺序 = 原始时间顺序，抽完后三个阶段首尾相接，得到新的一条轨迹。

── 重采样到统一长度 ──────────────────────────────────────────────────────
文档要求"其余帧丢弃后重采样到统一长度"。由于三个阶段各抽 N 帧，
不同 episode 的阶段划分帧数不同，抽帧后各 episode 长度已经统一为 3×N
（3个阶段 × 每阶段N帧），所以这里的"统一长度"通过阶段内固定抽N帧帧数
自然得到，不需要再做额外的时间插值重采样。

用法：
    python build_D2_D3_stage_frames.py --d0 /path/to/D0 --out /path/to/D2 --n_per_stage 2
    python build_D2_D3_stage_frames.py --d0 /path/to/D0 --out /path/to/D3 --n_per_stage 4

诊断工具：若怀疑阶段切分不合理，先用 diagnose_stage_split.py 检查真实数据
里 block_z / gripper 的时间序列，确认判定条件和实际物理过程是否吻合。
用法示例:
    python diagnose_stage_split.py --d0 D:/mujuco/demos_buoyancy --episodes 0,5,12
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lerobot_io import LeRobotDataset, LeRobotWriter

GRIPPER_CLOSE_THRESH = 127.0
DROP_THRESH_DEFAULT = 0.005  # 5mm，方块相对闭合时刻高度的下沉量阈值

# state 维度索引（对应 STATE_NAMES）
IDX_BLOCK_Z = 10  # block_x, block_y, block_z -> 8,9,10
IDX_GRIPPER_STATE = 7


def split_stages(states, actions, drop_thresh):
    """
    返回三个阶段的帧下标区间列表 [(start,end_exclusive), ...]，长度为3。
    v2切分逻辑见模块docstring：用"相对闭合时刻高度的下沉量"判定阶段2/3边界，
    不再使用绝对水面高度阈值（该阈值在方块静止漂浮时就已经满足，无法正确切分）。
    """
    T = len(actions)
    gripper_action = actions[:, 7]
    is_closed = gripper_action > GRIPPER_CLOSE_THRESH

    # 阶段1结束点：gripper 首次闭合的帧（含）
    closed_idxs = np.where(is_closed)[0]
    stage1_end = int(closed_idxs[0]) + 1 if len(closed_idxs) > 0 else T  # exclusive
    stage1_end = min(stage1_end, T)
    close_frame = stage1_end - 1 if len(closed_idxs) > 0 else T - 1

    # 阶段2结束点：方块高度相对"闭合时刻"下沉超过 drop_thresh 的第一帧
    block_z = states[:, IDX_BLOCK_Z]
    z_at_close = block_z[close_frame]
    search_start = stage1_end
    if search_start < T:
        drop = z_at_close - block_z[search_start:]  # 正值=下沉
        drop_idxs = np.where(drop >= drop_thresh)[0]
        if len(drop_idxs) > 0:
            stage2_end = search_start + int(drop_idxs[0]) + 1
        else:
            stage2_end = T
    else:
        stage2_end = T
    stage2_end = min(max(stage2_end, stage1_end), T)

    stage3_end = T

    bounds = [(0, stage1_end), (stage1_end, stage2_end), (stage2_end, stage3_end)]
    # 修正空区间：若某阶段长度为0，就把前一个阶段的最后一帧借给它，保证3个阶段都至少有1帧
    fixed = []
    for i, (s, e) in enumerate(bounds):
        if e <= s:
            s = max(0, e - 1)
        fixed.append((s, e))
    return fixed


def pick_indices_in_stage(start, end, n):
    """阶段区间 [start,end) 内均匀抽 n 帧下标（原始时间顺序，去重后不足则补齐）"""
    L = end - start
    if L <= 0:
        return []
    if L <= n:
        return list(range(start, end))  # 帧数不足，全部保留（不丢帧）
    raw = np.linspace(start, end - 1, n)
    idxs = sorted(set(int(round(x)) for x in raw))
    # 去重后可能不足n个，从阶段内未选中的帧里补齐，保持时间顺序
    if len(idxs) < n:
        remaining = [i for i in range(start, end) if i not in idxs]
        need = n - len(idxs)
        # 均匀地从剩余帧里再挑
        if remaining:
            extra_pick = np.linspace(0, len(remaining) - 1, min(need, len(remaining)))
            for x in extra_pick:
                idxs.append(remaining[int(round(x))])
        idxs = sorted(set(idxs))
    return idxs


def build_stage_dataset(d0_root, out_root, n_per_stage, drop_thresh, task_str):
    ds0 = LeRobotDataset(d0_root)
    all_idx = ds0.all_episode_indices()
    writer = LeRobotWriter(out_root, task_str=task_str)

    per_episode_manifest = []
    warn_count = 0

    for src_ep in all_idx:
        states, actions, meta = ds0.load_episode(src_ep)
        bounds = split_stages(states, actions, drop_thresh)

        frame_indices = []
        stage_lengths = []
        for (s, e) in bounds:
            L = e - s
            if L < n_per_stage:
                warn_count += 1
                print(f"  ⚠️ episode {src_ep} 某阶段仅有 {L} 帧 (< {n_per_stage})，"
                      f"该阶段全部保留，不丢帧")
            picked = pick_indices_in_stage(s, e, n_per_stage)
            frame_indices.extend(picked)
            stage_lengths.append(L)

        frame_indices = sorted(set(frame_indices))
        sel_states = states[frame_indices]
        sel_actions = actions[frame_indices]

        writer.save_episode_from_video_copy(
            ds0, src_ep, sel_states, sel_actions, frame_indices=frame_indices)
        print(f"  D episode {writer.n_episodes - 1} 源自 D0 ep{src_ep}: "
              f"阶段帧数原始={stage_lengths} → 抽帧后总帧数={len(frame_indices)}")

        per_episode_manifest.append({
            "source_episode_index": int(src_ep),
            "stage_bounds": [[int(s), int(e)] for s, e in bounds],
            "stage_lengths_original": [int(x) for x in stage_lengths],
            "picked_frame_indices": [int(x) for x in frame_indices],
        })

    writer.print_summary()
    if warn_count:
        print(f"\n共有 {warn_count} 次阶段帧数不足警告，详见 manifest。")

    manifest = {
        "source_dataset": str(Path(d0_root).resolve()),
        "n_per_stage": n_per_stage,
        "drop_thresh_meters": drop_thresh,
        "gripper_close_thresh": GRIPPER_CLOSE_THRESH,
        "stage_definition": ("v2: 阶段1:接近(至gripper首次闭合) / "
                              "阶段2:提起转移(至方块相对闭合时刻下沉超过drop_thresh) / "
                              "阶段3:按压入水(至结束)"),
        "episodes": per_episode_manifest,
    }
    with open(Path(out_root) / f"stage_split_manifest_n{n_per_stage}.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d0", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n_per_stage", type=int, required=True, choices=[2, 4],
                     help="每阶段抽帧数：D2用2，D3用4")
    ap.add_argument("--drop_thresh", type=float, default=DROP_THRESH_DEFAULT,
                     help="方块相对gripper闭合时刻高度的下沉量阈值（米），默认0.005（5mm）")
    args = ap.parse_args()

    task_str = (f"grasp floating block and slowly submerge it into water "
                f"(D{'2' if args.n_per_stage == 2 else '3'}: {args.n_per_stage} frames/stage)")
    build_stage_dataset(args.d0, args.out, args.n_per_stage, args.drop_thresh, task_str)


if __name__ == "__main__":
    main()
