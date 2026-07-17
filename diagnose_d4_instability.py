# -*- coding: utf-8 -*-
"""
diagnose_d4_instability.py —— 核查 D4 某条episode是否因MuJoCo数值不稳定警告而留下脏数据
=====================================================================
背景：build_D4_init_noise.py 跑 episode 2（对应D0 ep2）时，MuJoCo打印了：
    WARNING: Nan, Inf or huge value in QACC at DOF 9. The simulation is
    unstable. Time = 0.2140.
Time=0.214s，按 FPS=15 换算约等于第3帧左右（0.214 * 15 ≈ 3.2），说明问题
发生在仿真刚开始不久。虽然最终这条episode保存下来的帧数(110帧)和原始D0
episode 2完全一致，但不能只看"帧数没变"就认为数据没问题——数值不稳定
警告意味着某一步的加速度积分可能已经发散，需要直接检查state数值本身
有没有跳变/爆炸/骤停等不合理现象。

本脚本读取 D4 episode 2 的完整state序列，重点看：
    1. block_z, block_x, block_y 是否有超出物理合理范围的跳变
       （比如某一帧突然变成几十/几百这种明显不合理的大数，或者从一个值
       瞬间跳到另一个值又跳回来）
    2. ee_xyz、关节角是否有类似跳变
    3. 帧与帧之间的差分（速度）是否有异常尖峰，尖峰点对应大约frame 3附近

同时和D0 episode 2、D4其他"干净"的episode（比如episode 0）做对比，
帮助判断这是否是个别几帧的局部异常，还是整条数据已经不可用。

用法：
    python diagnose_d4_instability.py --d0 D:/mujuco/demos_buoyancy \
        --d4 D:/mujuco/demos_D4 --episode 2 --compare_clean 0
"""

import argparse
import numpy as np

from lerobot_io import LeRobotDataset

STATE_NAMES = ["j0", "j1", "j2", "j3", "j4", "j5", "j6", "gripper",
               "block_x", "block_y", "block_z", "ee_x", "ee_y", "ee_z"]


IDX_GRIPPER = 7  # state第7维是gripper，正常范围0~255，不是坐标/角度，检测时需单独排除


def scan_for_anomalies(states, label):
    """
    扫描state序列，标记：
      - 任何NaN/Inf
      - 任何单帧数值超出合理物理范围（block/ee坐标应在大致 -1~2 米量级内，
        关节角应在 -2π~2π 量级内，这里用较宽松的阈值 abs()>10 作为"明显不合理"；
        【重要】gripper维度(index 7)的正常范围是0~255，必须排除在这条检测之外，
        否则任何抓取闭合动作都会被误判为数值异常——这是v1版本的一个bug，
        已在v2修复）
      - 帧间跳变（差分绝对值）里的离群点：超过该维度全程差分标准差的 8倍
        （gripper维度同样排除，因为开合瞬间的跳变是正常指令变化，不是物理发散）
    返回是否发现异常，以及异常帧号列表。
    """
    T = states.shape[0]
    anomalies = []

    nan_mask = ~np.isfinite(states)
    if nan_mask.any():
        bad_frames = sorted(set(np.where(nan_mask)[0].tolist()))
        anomalies.append(("NaN/Inf", bad_frames))

    # 排除gripper维度(index 7)，其余13维（关节角7个 + block_xyz + ee_xyz）用abs()>10判定
    non_gripper_idx = [i for i in range(states.shape[1]) if i != IDX_GRIPPER]
    huge_mask_full = np.abs(states) > 10.0
    huge_mask_full[:, IDX_GRIPPER] = False  # 明确排除gripper那一列，不参与判定
    if huge_mask_full.any():
        bad_frames = sorted(set(np.where(huge_mask_full)[0].tolist()))
        # 定位具体是哪一维、哪个数值触发的，而不是只报帧号——避免像gripper那次一样误判
        trigger_details = []
        first_frame = bad_frames[0]
        for dim_i in range(states.shape[1]):
            if dim_i == IDX_GRIPPER:
                continue
            if huge_mask_full[first_frame, dim_i]:
                trigger_details.append(f"{STATE_NAMES[dim_i]}={states[first_frame, dim_i]:.4f}")
        anomalies.append((
            f"绝对值>10（已排除gripper维度；首个触发帧{first_frame}的具体维度: "
            f"{', '.join(trigger_details)}）",
            bad_frames))

    diffs = np.diff(states, axis=0)  # [T-1, 14]
    diffs_no_gripper = diffs[:, non_gripper_idx]
    diff_std = diffs_no_gripper.std(axis=0, keepdims=True) + 1e-9
    spike_mask = np.abs(diffs_no_gripper) > 8 * diff_std
    if spike_mask.any():
        bad_frames = sorted(set((np.where(spike_mask)[0] + 1).tolist()))  # +1: 对应跳变后的帧
        anomalies.append(("帧间跳变离群点(>8倍该维度标准差，已排除gripper维度)", bad_frames))

    print(f"\n{'='*70}")
    print(f"{label}  (共{T}帧)")
    if not anomalies:
        print("  未发现明显数值异常。")
        return False, []

    all_bad = set()
    for name, frames in anomalies:
        print(f"  ⚠️ {name}: 涉及帧号 {frames}")
        all_bad.update(frames)
    return True, sorted(all_bad)


def print_frame_window(states, center, radius, label):
    T = states.shape[0]
    lo = max(0, center - radius)
    hi = min(T, center + radius + 1)
    print(f"\n  --- {label}: frame [{lo}:{hi}) 详细数值 ---")
    for i in range(lo, hi):
        row = states[i]
        print(f"    frame {i:3d}: " + "  ".join(
            f"{n}={v:.4f}" for n, v in zip(STATE_NAMES, row)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d0", type=str, required=True)
    ap.add_argument("--d4", type=str, required=True)
    ap.add_argument("--episode", type=int, required=True,
                     help="要检查的episode_index（D0和D4里通常一一对应）")
    ap.add_argument("--compare_clean", type=int, default=None,
                     help="可选：另一条'正常'episode的index，用于对比参考")
    ap.add_argument("--warning_time_sec", type=float, default=0.214,
                     help="MuJoCo打印的不稳定警告发生时间（秒），用于估算对应帧号")
    ap.add_argument("--fps", type=float, default=15.0)
    args = ap.parse_args()

    ds0 = LeRobotDataset(args.d0)
    ds4 = LeRobotDataset(args.d4)

    states0, actions0, _ = ds0.load_episode(args.episode)
    states4, actions4, _ = ds4.load_episode(args.episode)

    approx_frame = int(round(args.warning_time_sec * args.fps))
    print(f"MuJoCo警告发生时间 {args.warning_time_sec}s，按fps={args.fps}估算对应约第 {approx_frame} 帧附近")
    print(f"（注意：警告发生时间是仿真内部时间，可能早于第一次RECORD_EVERY记录点，"
          f"实际记录到的异常可能出现在这附近或稍后的第一个记录帧）")

    has_anomaly, bad_frames = scan_for_anomalies(states4, f"D4 episode {args.episode}（重新仿真后）")

    print(f"\n对比：D0 原始 episode {args.episode}（未加噪声，作为基准）")
    scan_for_anomalies(states0, f"D0 episode {args.episode}（原始基准）")

    if args.compare_clean is not None:
        states4_clean, _, _ = ds4.load_episode(args.compare_clean)
        print(f"\n对比：D4 episode {args.compare_clean}（未报警的'干净'样本，作为对比参考）")
        scan_for_anomalies(states4_clean, f"D4 episode {args.compare_clean}（对比参考）")

    # 不管scan有没有抓到，都把警告发生时间附近的帧详细打出来，人工目视确认
    print_frame_window(states4, approx_frame, 5,
                        f"D4 episode {args.episode} 在估算警告帧附近")

    if has_anomaly:
        print(f"\n{'='*70}")
        print(f"结论：D4 episode {args.episode} 检测到数值异常，涉及帧号 {bad_frames}。")
        print("建议：不要直接用于后续训练/指标计算，考虑以下处理方式之一：")
        print("  1. 换个随机seed或调小--sigma，重新生成这一条episode")
        print("  2. 如果异常帧只在开头一两帧且后续已恢复正常，可考虑丢弃开头异常帧")
        print("     （但这样会破坏'完整action序列重新仿真'的一致性，不推荐）")
        print("  3. 直接从D4数据集里剔除这条episode（20条变19条，在manifest里注明）")
    else:
        print(f"\n{'='*70}")
        print(f"结论：D4 episode {args.episode} 的state数值扫描未发现明显异常"
              f"（NaN/Inf/超范围/跳变离群点均未触发阈值）。")
        print("MuJoCo的QACC警告可能只是仿真内部某一子步的瞬时数值问题，")
        print("引擎自身的稳定性处理（如子步长自适应）已经把它\"压\"回了合理范围，")
        print("最终记录到的state序列看起来是正常的。仍建议肉眼过一遍这条episode")
        print("对应的渲染视频（D:/mujuco/demos_D4/videos/.../file-002.mp4），")
        print("确认画面没有肉眼可见的抖动/穿模，再决定是否放心使用。")


if __name__ == "__main__":
    main()
