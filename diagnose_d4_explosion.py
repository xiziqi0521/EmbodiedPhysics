# -*- coding: utf-8 -*-
"""
diagnose_d4_explosion.py —— 定位D4重仿真中block数值爆炸的具体成因
=====================================================================
背景：check_block_drift_origin.py 已确认 D0 全程干净，D4 在某一帧后
block_x 从正常量级（~0.3）在几十帧内飙到 1e4 量级，属于典型的物理引擎
数值发散（而不是简单的坐标系/单位错位）。

本脚本不依赖 apply_buoyancy 的具体实现，只是在重仿真循环里对每一帧
额外记录：
    - block线速度范数 |qvel(block)|
    - block所受外力范数 |xfrc_applied(block)|（如果apply_buoyancy是
      通过xfrc_applied施加力的话；如果是通过qfrc_applied或其他机制，
      这里数值会看不出增长，需要按提示切换记录对象）
    - block位置

通过观察这几列数值随帧数的增长模式，可以判断：
    - 如果 |xfrc_applied| 逐帧线性增长（frame N 大约是 frame 1 的 N 倍）
      → 高度符合"每帧调用 apply_buoyancy 时用 += 而不是 = 施加力，
        导致外力累加、从未清零"这一假设，需要去 apply_buoyancy 里改成
        赋值或每帧先清零再赋值
    - 如果 |xfrc_applied| 本身正常（比如稳定在浮力量级 ~ 0.1~1 N），
      但 |qvel| 却异常增长 → 更可能是接触穿透/约束求解发散
      （比如扰动后block初始位置与水槽壁/其他物体有微小重叠，
      constraint solver为消除穿透生成了巨大反作用力）
    - 如果两者都从一开始就正常，只是在某一帧突然跳变（阶跃式而非渐进）
      → 更可能是某个索引越界/赋值错误（比如qpos偏移量算错，把别的
        物体状态写进了block的位置分量）

用法：
    python diagnose_d4_explosion.py --d0 D:/mujuco/demos_buoyancy --episode 0 --sigma 0.006 --seed 42
"""

import argparse

import numpy as np
import mujoco

from lerobot_io import LeRobotDataset
import teleop_buoyancy_lerobot as sim


def diagnose(d0_root, ep_idx, sigma, seed, max_frames_to_print):
    ds0 = LeRobotDataset(d0_root)
    _, orig_actions, _ = ds0.load_episode(ep_idx)

    rng = np.random.default_rng(seed)
    m, d, water_surface_z = sim.build_model()
    hand_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    block_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "block_body")

    mujoco.mj_resetData(m, d)
    home_q = np.array([0.0, 0.5, 0.0, -2.0, 0.0, 2.5, 0.785])
    d.qpos[:7] = home_q.copy()
    d.ctrl[:7] = home_q.copy()
    d.ctrl[7] = sim.GRIPPER_OPEN

    float_depth = (sim.BLOCK_DENSITY / sim.WATER_DENSITY) * 2 * sim.BALL_RADIUS
    block_z = water_surface_z - float_depth + sim.BALL_RADIUS

    noise_xy = rng.normal(0.0, sigma, size=2)
    block_x = sim.TANK_X + noise_xy[0]
    block_y = 0.0 + noise_xy[1]

    block_jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "block_joint")
    jnt_qposadr = m.jnt_qposadr[block_jnt_id]
    jnt_dofadr = m.jnt_dofadr[block_jnt_id]
    print(f"block_joint id={block_jnt_id}, qposadr={jnt_qposadr}, dofadr={jnt_dofadr}, "
          f"jnt_type={m.jnt_type[block_jnt_id]} (0=free,1=ball,2=slide,3=hinge)")
    print(f"扰动后初始位置: x={block_x:.4f}, y={block_y:.4f}, z={block_z:.4f}, "
          f"water_surface_z={water_surface_z:.4f}\n")

    d.qpos[jnt_qposadr:jnt_qposadr + 3] = [block_x, block_y, block_z]
    d.qpos[jnt_qposadr + 3] = 1.0
    d.qpos[jnt_qposadr + 4:jnt_qposadr + 7] = 0.0
    mujoco.mj_forward(m, d)

    print(f"{'frame':>5} | {'|qvel_block|':>14} | {'|xfrc_applied|':>15} | "
          f"{'|qfrc_applied|':>15} | {'block_xyz':>28} | {'ncon(接触点数)':>14}")
    print("-" * 100)

    prev_xfrc_norm = None
    growth_ratios = []

    for i, ctrl in enumerate(orig_actions):
        # 【修复验证】清零上一帧遗留的xfrc_applied，避免浮力逐帧累加
        d.xfrc_applied[block_bid] = 0.0
        sim.apply_buoyancy(m, d, block_bid, water_surface_z)
        d.ctrl[:7] = ctrl[:7]
        d.ctrl[7] = ctrl[7]

        # 记录施加力（在mj_step之前，即apply_buoyancy刚施加完之后的状态）
        xfrc_norm = float(np.linalg.norm(d.xfrc_applied[block_bid]))
        qfrc_norm = float(np.linalg.norm(d.qfrc_applied[jnt_dofadr:jnt_dofadr + 3]))

        mujoco.mj_step(m, d)

        qvel_norm = float(np.linalg.norm(d.qvel[jnt_dofadr:jnt_dofadr + 3]))
        block_pos = d.qpos[jnt_qposadr:jnt_qposadr + 3]
        ncon = d.ncon

        if i < max_frames_to_print or qvel_norm > 10 or xfrc_norm > 100:
            print(f"{i:5d} | {qvel_norm:14.4f} | {xfrc_norm:15.4f} | "
                  f"{qfrc_norm:15.4f} | "
                  f"({block_pos[0]:8.4f},{block_pos[1]:8.4f},{block_pos[2]:8.4f}) | "
                  f"{ncon:14d}")

        if prev_xfrc_norm is not None and prev_xfrc_norm > 1e-9:
            growth_ratios.append(xfrc_norm / prev_xfrc_norm)
        prev_xfrc_norm = xfrc_norm

        # 一旦速度或位置已经明显发散，再打印几帧就退出，没必要跑完全程
        if qvel_norm > 1e4 or abs(block_pos[0]) > 100:
            print(f"\n  数值已明显发散（frame {i}），提前终止诊断循环。")
            break

    if growth_ratios:
        ratios = np.array(growth_ratios[-30:]) if len(growth_ratios) > 30 else np.array(growth_ratios)
        ratios = ratios[np.isfinite(ratios)]
        if len(ratios) > 0:
            print(f"\n  最近若干帧 |xfrc_applied| 逐帧增长比例的中位数 ≈ {np.median(ratios):.3f}")
            print(f"  （若持续 > 1 且相对稳定，强烈提示是外力逐帧累加未清零；"
                  f"若在1附近波动后突然跳变，更像是接触/约束问题）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d0", type=str, required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=0.006)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-frames-to-print", type=int, default=10,
                     help="正常情况下打印开头多少帧（异常帧无论如何都会打印）")
    args = ap.parse_args()
    diagnose(args.d0, args.episode, args.sigma, args.seed, args.max_frames_to_print)


if __name__ == "__main__":
    main()
