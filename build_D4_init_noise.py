# -*- coding: utf-8 -*-
"""
build_D4_init_noise.py —— 构造 D4：初始状态误差-小噪声
=====================================================================
【需在装有 MuJoCo 的本地环境运行，依赖 teleop_buoyancy_lerobot.py】

对应文档《二、每个场景需要准备的数据集》表格 D4 行：

    只对每条 demo 的第一帧 state（如 ball_xyz、ee_xyz）加一次小幅高斯扰动，
    之后所有帧按物理引擎重新演算，不能只改标签不重新仿真。

── 实现思路 ──────────────────────────────────────────────────────────────
D0 的每条 episode 里，action 序列本质是"末端执行器目标位置轨迹 + 夹爪开合指令"
（teleop_buoyancy_lerobot.py 里 ik_step 每步用 d.ctrl[:7] 跟踪 ee_target，
d.ctrl[7] 是夹爪指令）。要做到"重新演算"而不是"只改标签"，本脚本：

    1. 重建与录制时完全相同的 MuJoCo 场景（build_model）
    2. 把方块（block）初始位置在 D0 基础上加一次小幅高斯扰动
       （只扰动 x,y，z 由浮力平衡自动决定，避免破坏漂浮物理约束）
    3. 把原 D0 episode 的 action 序列（ctrl 指令）逐帧原样回放，
       每步都调用 apply_buoyancy 施加浮力，让 mujoco 重新积分整条轨迹
    4. 重新渲染三路相机画面，记录新的 state 序列
    5. 新 episode 长度可能与原 episode 不同（比如球初始位置变化导致抓取
       时机略有偏差），属于"重新仿真"的正常结果，如实记录，不做截断/补齐

── 扰动设定 ──────────────────────────────────────────────────────────────
    对方块初始 (x, y) 位置加 N(0, sigma^2)，sigma 默认 0.006 m（6mm，
    小幅扰动，量级约为球半径0.025m的1/4，不会导致抓取完全失败）
    可用 --sigma 命令行参数调整并记录进 manifest 以便复现

── 已知教训（对应D2/D3调试过程）────────────────────────────────────────
本脚本不依赖任何"绝对水面高度"之类的判定阈值来切分/判断，只是原样回放
action序列，所以不会重蹈D2/D3最初版本的覆辙（阈值和实际漂浮状态不吻合）。
但要注意：如果扰动导致抓取失败（方块被推出可及范围），新episode可能会
明显变长或变短，脚本会打印警告，需要人工确认是否要调小sigma重跑。

── 【本版本修复】xfrc_applied 累加未清零导致数值发散 ─────────────────────
diagnose_d4_explosion.py 诊断确认：block受到的 |xfrc_applied| 逐帧增量
趋于一个接近常数的值（约1.4，即真实浮力大小），说明 apply_buoyancy
每帧施加的浮力是在上一帧遗留力的基础上累加的，而 mj_step 并不会自动
清零 xfrc_applied。几十帧后累积力变得极大，推动/顶穿接触面，约束求解器
随之发散，block位置飙到 1e4 量级。
本版本在每帧调用 apply_buoyancy 之前，显式将该帧对应的 xfrc_applied
清零，确保浮力是"当帧重新计算、不带历史残留"地施加，从根源上修复发散。

用法（本地 Windows + MuJoCo 环境）：
    python build_D4_init_noise.py --d0 D:/mujuco/demos_buoyancy --out D:/mujuco/demos_D4 --sigma 0.006
"""

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco

from lerobot_io import LeRobotDataset, LeRobotWriter
import teleop_buoyancy_lerobot as sim

TASK_STR_D4 = "grasp floating block and slowly submerge it into water (D4: initial-state small noise)"


def rerun_episode_with_init_noise(m, d, water_surface_z, hand_bid, block_bid,
                                   renderers, cam_ids, orig_actions, rng, sigma):
    """
    在给定 mujoco model/data 上，对方块初始位置加噪声后，
    按 orig_actions 逐帧回放并重新仿真，返回 (states, actions, frames_dict, init_xyz)
    """
    rf, rs, rt = renderers
    id_front, id_side, id_top = cam_ids

    mujoco.mj_resetData(m, d)
    home_q = np.array([0.0, 0.5, 0.0, -2.0, 0.0, 2.5, 0.785])
    d.qpos[:7] = home_q.copy()
    d.ctrl[:7] = home_q.copy()
    d.ctrl[7] = sim.GRIPPER_OPEN

    float_depth = (sim.BLOCK_DENSITY / sim.WATER_DENSITY) * 2 * sim.BALL_RADIUS
    block_z = water_surface_z - float_depth + sim.BALL_RADIUS

    # 初始状态小幅高斯扰动：只扰动 x,y（z 由浮力平衡决定，避免破坏物理合理性）
    noise_xy = rng.normal(0.0, sigma, size=2)
    block_x = sim.TANK_X + noise_xy[0]
    block_y = 0.0 + noise_xy[1]

    block_jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "block_joint")
    jnt_qposadr = m.jnt_qposadr[block_jnt_id]
    d.qpos[jnt_qposadr:jnt_qposadr + 3] = [block_x, block_y, block_z]
    d.qpos[jnt_qposadr + 3] = 1.0
    d.qpos[jnt_qposadr + 4:jnt_qposadr + 7] = 0.0
    mujoco.mj_forward(m, d)

    states, actions, frames = [], [], {"cam_front": [], "cam_side": [], "cam_top": []}

    def render(renderer, cam_id):
        renderer.update_scene(d, camera=cam_id)
        return renderer.render().copy()

    for ctrl in orig_actions:
        # 关键修复：xfrc_applied 不会被 mj_step 自动清零。如果 apply_buoyancy
        # 内部是用 += 施加浮力，每帧调用一次就会导致外力在上一帧的基础上
        # 继续叠加、从未清零。diagnose_d4_explosion.py 的诊断结果显示
        # |xfrc_applied| 的逐帧增量趋于一个接近常数的值（约等于真实浮力大小），
        # 这正是"力累加未清零"的典型特征：第N帧的力 ≈ N × 单帧浮力。
        # 这里显式先清零该帧的 xfrc_applied，确保 apply_buoyancy 施加的是
        # "当帧应有的浮力"，而不是历史力的叠加，从根源上避免发散。
        d.xfrc_applied[block_bid] = 0.0
        sim.apply_buoyancy(m, d, block_bid, water_surface_z)
        d.ctrl[:7] = ctrl[:7]
        d.ctrl[7] = ctrl[7]
        mujoco.mj_step(m, d)

        block_pos = d.xpos[block_bid]
        ee_pos = d.xpos[hand_bid]
        state = (d.qpos[:7].tolist() + [float(d.qpos[7])] +
                  block_pos.tolist() + ee_pos.tolist())
        states.append(state)
        actions.append(d.ctrl[:8].tolist())
        frames["cam_front"].append(render(rf, id_front))
        frames["cam_side"].append(render(rs, id_side))
        frames["cam_top"].append(render(rt, id_top))

    return (np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            frames, (block_x, block_y, block_z))


def build_d4(d0_root, out_root, sigma, seed):
    ds0 = LeRobotDataset(d0_root)
    all_idx = ds0.all_episode_indices()
    rng = np.random.default_rng(seed)

    m, d, water_surface_z = sim.build_model()
    hand_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    block_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "block_body")
    id_front = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam_front")
    id_side = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam_side")
    id_top = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam_top")
    rf = mujoco.Renderer(m, sim.IMG_SIZE, sim.IMG_SIZE)
    rs = mujoco.Renderer(m, sim.IMG_SIZE, sim.IMG_SIZE)
    rt = mujoco.Renderer(m, sim.IMG_SIZE, sim.IMG_SIZE)

    writer = LeRobotWriter(out_root, task_str=TASK_STR_D4)
    manifest_eps = []

    try:
        for src_ep in all_idx:
            _, orig_actions, meta = ds0.load_episode(src_ep)
            states, actions, frames, init_xyz = rerun_episode_with_init_noise(
                m, d, water_surface_z, hand_bid, block_bid,
                (rf, rs, rt), (id_front, id_side, id_top),
                orig_actions, rng, sigma)

            T = writer.save_episode_with_frames(frames, states, actions)
            len_diff = T - len(orig_actions)
            print(f"  D4 episode {writer.n_episodes - 1} 源自 D0 ep{src_ep}: "
                  f"重新仿真 {T} 帧 (原{len(orig_actions)}帧, 差{len_diff:+d}), "
                  f"初始位置扰动后=({init_xyz[0]:.4f},{init_xyz[1]:.4f},{init_xyz[2]:.4f})")
            if abs(len_diff) > 0.2 * len(orig_actions):
                print(f"    ⚠️ 帧数差异较大(>20%)，建议检查该episode是否因初始位置偏移导致"
                      f"抓取时机/成功率明显变化")

            manifest_eps.append({
                "source_episode_index": int(src_ep),
                "orig_length": int(len(orig_actions)),
                "resim_length": int(T),
                "init_block_xyz_after_noise": [float(x) for x in init_xyz],
            })
    finally:
        rf.close(); rs.close(); rt.close()

    writer.print_summary()

    manifest = {
        "source_dataset": str(Path(d0_root).resolve()),
        "sigma_meters": sigma,
        "noise_dims": "block initial (x, y); z determined by buoyancy equilibrium",
        "seed": seed,
        "note": "整条action序列在噪声化初始状态下重新用MuJoCo物理引擎演算，而非仅修改标签",
        "episodes": manifest_eps,
    }
    with open(Path(out_root) / "D4_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"扰动参数已记录到 {Path(out_root) / 'D4_manifest.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d0", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--sigma", type=float, default=0.006, help="初始位置高斯扰动标准差（米）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build_d4(args.d0, args.out, args.sigma, args.seed)


if __name__ == "__main__":
    main()
