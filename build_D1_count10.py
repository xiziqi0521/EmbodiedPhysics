# -*- coding: utf-8 -*-
"""
build_D1_count10.py —— 构造 D1：数据条数-10条
=====================================================================
对应文档《二、每个场景需要准备的数据集》表格 D1 行：

    从 D0 的 20 条里固定抽 10 条（按 episode_index 奇偶抽取，写清楚抽取
    规则以便复现），重新生成 meta/episodes、stats.json。

抽取规则（固定、可复现）：
    取 D0 中 episode_index 为偶数的那 10 条：0,2,4,...,18
    （如果 D0 不是恰好20条，脚本会打印警告并按"偶数索引"规则实际抽取到的条数继续）

每条 episode 的帧、图像、action 完全不变，只是被原样复制到新数据集里，
episode_index 在新数据集中重新从 0 连续编号。

用法：
    python build_D1_count10.py --d0 /path/to/D0 --out /path/to/D1
"""

import argparse
import json
from pathlib import Path

from lerobot_io import LeRobotDataset, LeRobotWriter

TASK_STR_D1 = "grasp floating block and slowly submerge it into water (D1: 10-episode subset)"


def select_episode_indices(all_indices, rule="even"):
    """
    固定抽取规则：偶数 episode_index。
    all_indices: D0 中实际存在的 episode_index 列表（已排序）
    返回被选中的 episode_index 列表，按升序排列。
    """
    if rule == "even":
        selected = [idx for idx in all_indices if idx % 2 == 0]
    elif rule == "first10":
        selected = sorted(all_indices)[:10]
    else:
        raise ValueError(f"未知抽取规则: {rule}")
    return selected


def build_d1(d0_root, out_root, rule="even"):
    ds0 = LeRobotDataset(d0_root)
    all_idx = ds0.all_episode_indices()
    selected = select_episode_indices(all_idx, rule=rule)

    print(f"D0 共 {len(all_idx)} 条 episode: {all_idx}")
    print(f"D1 抽取规则: {rule}  →  选中 {len(selected)} 条: {selected}")
    if len(selected) != 10:
        print(f"  ⚠️ 警告：按规则 '{rule}' 实际选中 {len(selected)} 条，非预期的10条，"
              f"请检查 D0 是否恰好20条 episode（0~19）。")

    writer = LeRobotWriter(out_root, task_str=TASK_STR_D1)
    for src_ep in selected:
        states, actions, meta = ds0.load_episode(src_ep)
        T = len(actions)
        frame_indices = list(range(T))  # 帧本身完全不变，全部保留
        writer.save_episode_from_video_copy(
            ds0, src_ep, states, actions, frame_indices=frame_indices)
        print(f"  已写入 D1 episode {writer.n_episodes - 1} (源自 D0 episode {src_ep}, {T} 帧)")

    writer.print_summary()

    # 记录抽取规则，便于复现和存档
    manifest = {
        "source_dataset": str(Path(d0_root).resolve()),
        "rule": rule,
        "selected_source_episode_indices": selected,
        "n_episodes": len(selected),
    }
    with open(Path(out_root) / "D1_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"抽取规则已记录到 {Path(out_root) / 'D1_manifest.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d0", type=str, required=True, help="D0 数据集根目录")
    ap.add_argument("--out", type=str, required=True, help="D1 输出目录")
    ap.add_argument("--rule", type=str, default="even", choices=["even", "first10"],
                     help="抽取规则：even=偶数episode_index（默认），first10=前10条")
    args = ap.parse_args()
    build_d1(args.d0, args.out, rule=args.rule)


if __name__ == "__main__":
    main()
