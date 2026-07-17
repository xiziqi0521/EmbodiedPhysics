# -*- coding: utf-8 -*-
"""
check_block_drift_origin.py —— 核查 block_x 飘移是D0原始数据自带，还是D4/D5重新仿真引入的
=====================================================================
背景：diagnose_d4_instability.py 发现 D4 episode 0 和 episode 2 都在episode
后半段（大约后1/3）出现 block_x 飘到 10~20 这种明显超出水槽范围（TANK_X=0.35
附近，水槽宽度TANK_W=0.28）的现象。这说明方块物理上"飞出"了合理范围，
需要搞清楚：
    (a) D0原始录制数据里，这些episode后半段本来就有类似问题（说明是录制
        阶段的数据质量问题，和D4/D5的重新仿真逻辑无关）
    (b) 还是D0原始数据是干净的，问题是D4/D5重新仿真过程中新引入的
        （比如action回放时坐标系/单位对不上，或者浮力施加逻辑有bug）

核查方法：直接对比同一个episode_index在D0和D4里，block_x的完整时间序列，
重点看：
    - D0原始数据的block_x是否全程保持在合理范围内（基于水槽实际尺寸计算，
      而不是写死一个宽松阈值）
    - 如果D0是干净的，那么D4新增的飘移就出在"重新仿真"这个环节，需要
      进一步查build_D4_init_noise.py的仿真逻辑
    - 如果D0本身在对应时间段已经有block_x的异常大值，那么这是原始录制
      数据的问题，不是D4/D5脚本的bug

本版本相较于最初版本的改进：
    1. 状态向量列索引会先做维度校验（并尽量从metadata里核对字段名），
       避免D0/D4状态向量定义不一致导致取错列、产生误判。
    2. 显式检查D0/D4的帧数是否一致；不一致时给出警告，并且"D4飘移起点
       附近看D0"这类逐帧对比只在两边帧数一致、可信任逐帧对齐时才做，
       否则改用时间比例定位并明确标注这是近似对齐。
    3. 判断阈值改成基于TANK_X/TANK_W动态计算的水槽合理范围，而不是写死
       的|x|>2，这样能抓住"没那么离谱但也确实飞出水槽"的飘移。
    4. "D0和D4是否是同一物理事件"不再只是笼统猜测，而是显式计算两边
       异常起始帧的差值（frame_diff），给出量化依据。
    5. 增加 --tol 参数，允许在阈值基础上放宽/收紧多少（单位：米），
       以及 --strict-index 参数，若字段名校验失败可选择直接报错退出。

用法：
    python check_block_drift_origin.py --d0 D:/mujuco/demos_buoyancy \
        --d4 D:/mujuco/demos_D4 --episodes 0,2
"""

import argparse
import sys

import numpy as np

from lerobot_io import LeRobotDataset

# ---- 水槽几何参数（来自diagnose_d4_instability.py的背景描述） ----
TANK_X = 0.35       # 水槽中心x坐标
TANK_W = 0.28       # 水槽宽度
# 合理范围在水槽中心±(半宽 + 容差)之外都算"飞出"
DEFAULT_TOLERANCE = 0.15  # 额外留一点容差（米），避免卡边缘正常值

# ---- 状态向量里 block 位置对应的列（若metadata里能查到字段名，会优先核对） ----
IDX_BLOCK_X = 8
IDX_BLOCK_Y = 9
IDX_BLOCK_Z = 10
EXPECTED_FIELD_NAMES = {
    IDX_BLOCK_X: ("block_x", "block_pos_x", "obj_x"),
    IDX_BLOCK_Y: ("block_y", "block_pos_y", "obj_y"),
    IDX_BLOCK_Z: ("block_z", "block_pos_z", "obj_z"),
}


def get_bound():
    """根据水槽几何参数计算block_x的合理范围边界"""
    half_w = TANK_W / 2.0
    lo = TANK_X - half_w - DEFAULT_TOLERANCE
    hi = TANK_X + half_w + DEFAULT_TOLERANCE
    return lo, hi


def verify_state_indices(ds, ds_name, strict=False):
    """
    尽量核对 IDX_BLOCK_* 是否真的对应 block 的位置字段。
    如果dataset对象暴露了字段名列表（常见属性名尝试：state_names /
    feature_names / column_names），就做一次名字核对；拿不到就只打印
    提示，不阻塞流程（除非 strict=True）。
    """
    field_names = None
    for attr in ("state_names", "feature_names", "column_names", "state_keys"):
        if hasattr(ds, attr):
            field_names = getattr(ds, attr)
            break

    if field_names is None:
        print(f"  [提示] {ds_name} 未暴露字段名列表，跳过 block_x/y/z 列索引的名称核对，"
              f"直接假定 IDX_BLOCK_X/Y/Z = {IDX_BLOCK_X}/{IDX_BLOCK_Y}/{IDX_BLOCK_Z} 正确。"
              f"若D0/D4状态向量定义不同，这里可能会取错列。")
        return

    mismatches = []
    for idx, candidates in EXPECTED_FIELD_NAMES.items():
        if idx >= len(field_names):
            mismatches.append((idx, "越界", candidates))
            continue
        actual = field_names[idx]
        if actual not in candidates:
            mismatches.append((idx, actual, candidates))

    if mismatches:
        print(f"  ⚠️ [{ds_name}] 状态向量列索引与预期字段名不匹配：")
        for idx, actual, candidates in mismatches:
            print(f"      列{idx}: 实际='{actual}', 预期之一={candidates}")
        if strict:
            print(f"  --strict-index 已开启，因字段名核对失败而中止。")
            sys.exit(1)
        else:
            print(f"     → 继续执行，但结果可能因列索引错位而失真，建议核实后重跑。")
    else:
        print(f"  [{ds_name}] 字段名核对通过，block_x/y/z 列索引正确。")


def find_bad_frames(bx, lo, hi):
    return np.where((bx < lo) | (bx > hi))[0]


def check_episode(ds0, ds4, ep_idx, lo, hi):
    states0, actions0, _ = ds0.load_episode(ep_idx)
    states4, actions4, _ = ds4.load_episode(ep_idx)

    bx0 = states0[:, IDX_BLOCK_X]
    bx4 = states4[:, IDX_BLOCK_X]

    print(f"\n{'='*70}")
    print(f"Episode {ep_idx}")
    print(f"  合理范围（基于 TANK_X={TANK_X}, TANK_W={TANK_W}, 容差={DEFAULT_TOLERANCE}）: "
          f"[{lo:.4f}, {hi:.4f}]")
    print(f"  D0 (原始, {len(bx0)}帧)  block_x 范围: min={bx0.min():.4f}, max={bx0.max():.4f}")
    print(f"  D4 (重仿真, {len(bx4)}帧)  block_x 范围: min={bx4.min():.4f}, max={bx4.max():.4f}")

    frame_count_mismatch = len(bx0) != len(bx4)
    if frame_count_mismatch:
        print(f"  ⚠️ D0与D4帧数不一致（{len(bx0)} vs {len(bx4)}），"
              f"逐帧对齐对比不可靠，下面的'邻近帧'分析仅作近似参考。")

    d0_bad = find_bad_frames(bx0, lo, hi)
    d4_bad = find_bad_frames(bx4, lo, hi)

    if len(d0_bad) > 0:
        print(f"  ⚠️ D0原始数据本身就有 {len(d0_bad)} 帧 block_x 超出合理范围，"
              f"首次出现在帧{d0_bad[0]}, 数值={bx0[d0_bad[0]]:.4f}")
        print(f"     → 说明这是D0录制阶段就存在的问题，与D4/D5的重新仿真逻辑无关")
    else:
        print(f"  ✅ D0原始数据全程 block_x 保持合理范围")

    if len(d4_bad) > 0:
        print(f"  ⚠️ D4重新仿真数据有 {len(d4_bad)} 帧 block_x 超出合理范围，"
              f"首次出现在帧{d4_bad[0]}, 数值={bx4[d4_bad[0]]:.4f}")
        if len(d0_bad) == 0:
            print(f"     → D0是干净的但D4飘了，问题出在重新仿真环节，需要排查"
                  f"build_D4_init_noise.py")
        else:
            frame_diff = abs(int(d0_bad[0]) - int(d4_bad[0]))
            print(f"     → D0和D4均出现异常。异常起始帧相差 {frame_diff} 帧"
                  f"（D0起点={d0_bad[0]}, D4起点={d4_bad[0]}）。")
            if frame_diff <= 3:
                print(f"       起始帧非常接近，很可能是同一物理事件的延续"
                      f"（比如原始录制这条demo后半段机械臂动作本身导致方块脱离正常范围）。")
            else:
                print(f"       起始帧相差较大，不能简单认定是同一事件，"
                      f"建议分别检查两边各自起飞点前后的动作/接触力序列。")
    else:
        print(f"  ✅ D4重新仿真数据全程 block_x 保持合理范围")

    # 打印D0在"飘移开始"附近的原始数值，看是不是也在起飞
    if len(d4_bad) > 0:
        center = int(d4_bad[0])
        if frame_count_mismatch:
            # 帧数不一致时，用时间比例近似定位D0里对应的帧，而不是直接用同一索引
            ratio = center / max(1, len(bx4) - 1)
            center0 = int(round(ratio * (len(bx0) - 1)))
            print(f"\n  [近似对齐] D4帧{center}按时间比例换算到D0约为帧{center0}"
                  f"（帧数不一致，仅供参考）：")
        else:
            center0 = center
            print(f"\n  D0原始数据在 D4飘移起点附近({max(0, center0-3)}~{center0+5}) 的"
                  f"block_x/y/z:")

        lo_i = max(0, center0 - 3)
        hi_i = min(len(bx0), center0 + 6)
        for i in range(lo_i, hi_i):
            print(f"    frame {i:3d}: block_x={states0[i,IDX_BLOCK_X]:.4f}  "
                  f"block_y={states0[i,IDX_BLOCK_Y]:.4f}  block_z={states0[i,IDX_BLOCK_Z]:.4f}")
    else:
        print(f"\n  D4未出现异常，跳过邻近帧对比。")


def main():
    global DEFAULT_TOLERANCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--d0", type=str, required=True)
    ap.add_argument("--d4", type=str, required=True)
    ap.add_argument("--episodes", type=str, default="0,2")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                    help="在水槽半宽基础上额外放宽的容差（米），默认 %(default)s")
    ap.add_argument("--strict-index", action="store_true",
                    help="若字段名核对失败（能核对到的情况下），直接报错退出，"
                         "而不是继续用可能错位的列索引跑下去")
    args = ap.parse_args()

    DEFAULT_TOLERANCE = args.tolerance

    ds0 = LeRobotDataset(args.d0)
    ds4 = LeRobotDataset(args.d4)

    print("正在核对 block_x/y/z 列索引是否与字段名匹配……")
    verify_state_indices(ds0, "D0", strict=args.strict_index)
    verify_state_indices(ds4, "D4", strict=args.strict_index)

    lo, hi = get_bound()

    for ep in [int(x) for x in args.episodes.split(",")]:
        check_episode(ds0, ds4, ep, lo, hi)


if __name__ == "__main__":
    main()
