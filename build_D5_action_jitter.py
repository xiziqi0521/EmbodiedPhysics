# -*- coding: utf-8 -*-
"""
build_D5_action_jitter.py —— 构造 D5：轨迹平滑度扰动
=====================================================================
【需在装有 MuJoCo 的本地环境运行，依赖 teleop_buoyancy_lerobot.py】

对应文档《二、每个场景需要准备的数据集》表格 D5 行 & 《三、为什么要新增D5》：

    保持帧密度、数据条数都不变，只对整条 action 序列加低幅高频扰动，
    让轨迹本身运动不平滑，需重新仿真或至少保证图像-状态一致。
    目的：制造一个"轨迹本身真实变不平滑"的对照组，且不改变帧数/条数，
    这样才能和 D1(条数)、D2/D3(帧密度) 的影响干净地分开。

── 扰动设计 ──────────────────────────────────────────────────────────────
只扰动关节控制量 ctrl[:7]（不扰动 gripper 维度 ctrl[7]，避免误触发抓取/
释放导致任务失败、帧数发生质变）：

    action_noisy[t, :7] = action_orig[t, :7] + A * sin(2*pi*f*t/FPS + phi_j)

    - "低幅"：幅度 A（弧度），默认 0.01 rad，相对关节活动范围（约几个rad）
      量级很小，不足以让机械臂偏离原轨迹的抓取/沉入意图
    - "高频"：频率 f，默认 6 Hz，明显高于示教动作本身的运动频率
      （示教节奏是"缓慢下压"，基频远低于1Hz），可以在SPARC/FFT谱上
      清楚地体现为高频成分增多、Jerk增大
    - 每个关节使用不同的随机相位 phi_j，避免7个关节同相位抖动导致
      整体只是刚体平移（那样反而不会破坏平滑度，会被浮力/接触抵消）
    - 用固定 seed 保证可复现

因为是在 ctrl 层面加扰动后重新仿真（而不是直接在录制好的 state 后处理），
帧数由物理引擎实际演化结果决定，不强制与源 episode 等长；若因为扰动导致
提前脱手等异常，脚本会打印警告，人工确认是否需要调低幅度重录。

── 【本版本修复】沿用 D4 排查确认的 xfrc_applied 累加未清零 bug ──────────
build_D4_init_noise.py 早期版本曾出现 block 位置在几十帧后数值发散到
1e4 量级的问题，diagnose_d4_explosion.py 诊断确认根因是：apply_buoyancy
每帧施加的浮力是在上一帧遗留的 xfrc_applied 基础上累加的，而 mj_step
不会自动清零该字段。本脚本重新仿真循环与 D4 是同一模式（每帧调用
apply_buoyancy 后 mj_step），存在完全相同的隐患，因此同步补上清零逻辑：
每帧调用 apply_buoyancy 之前，显式清零该帧对应的 xfrc_applied。

用法（本地 Windows + MuJoCo 环境）：
    python build_D5_action_jitter.py --d0 D:/mujuco/demos_buoyancy --out D:/mujuco/demos_D5 \
        --amp 0.01 --freq 6.0 --seed 123
"""

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco

from lerobot_io import LeRobotDataset, LeRobotWriter
import teleop_buoyancy_lerobot as sim

TASK_STR_D5 = "grasp floating block and slowly submerge it into water (D5: action jitter, low-amp high-freq)"


def jitter_actions(orig_actions, amp, freq, fps, rng):
    """
    对 orig_actions[:, :7]（关节ctrl）加低幅高频正弦扰动，gripper维度(index 7)不动。
    返回同形状的新 action 数组。
    """
    T = len(orig_actions)
    t = np.arange(T, dtype=np.float32) / fps
    phases = rng.uniform(0, 2 * np.pi, size=7)
    noisy = orig_actions.copy()
    for j in range(7):
        noisy[:, j] = orig_actions[:, j] + amp * np.sin(2 * np.pi * freq * t + phases[j])
    return noisy


def rerun_episode_with_action_jitter(m, d, water_surface_z, hand_bid, block_bid,
                                      renderers, cam_ids, noisy_actions):
    """在原始初始状态（不加初始噪声）下，按 noisy_actions 回放并重新仿真"""
    rf, rs, rt = renderers
    id_front, id_side, id_top = cam_ids

    mujoco.mj_resetData(m, d)
    home_q = np.array([0.0, 0.5, 0.0, -2.0, 0.0, 2.5, 0.785])
    d.qpos[:7] = home_q.copy()
    d.ctrl[:7] = home_q.copy()
    d.ctrl[7] = sim.GRIPPER_OPEN

    float_depth = (sim.BLOCK_DENSITY / sim.WATER_DENSITY) * 2 * sim.BALL_RADIUS
    block_z = water_surface_z - float_depth + sim.BALL_RADIUS
    block_jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "block_joint")
    jnt_qposadr = m.jnt_qposadr[block_jnt_id]
    d.qpos[jnt_qposadr:jnt_qposadr + 3] = [sim.TANK_X, 0.0, block_z]
    d.qpos[jnt_qposadr + 3] = 1.0
    d.qpos[jnt_qposadr + 4:jnt_qposadr + 7] = 0.0
    mujoco.mj_forward(m, d)

    states, actions = [], []
    frames = {"cam_front": [], "cam_side": [], "cam_top": []}

    def render(renderer, cam_id):
        renderer.update_scene(d, camera=cam_id)
        return renderer.render().copy()

    for ctrl in noisy_actions:
        # 【修复】与 D4 相同的隐患：xfrc_applied 不会被 mj_step 自动清零，
        # apply_buoyancy 每帧调用若在上一帧遗留力基础上累加，几十帧后就会
        # 导致 block 数值发散（参见 build_D4_init_noise.py 的修复说明）。
        # 这里显式清零，确保浮力是"当帧重新计算"的，不带历史残留。
        d.xfrc_applied[block_bid] = 0.0
        sim.apply_buoyancy(m, d, block_bid, water_surface_z)
        d.ctrl[:7] = np.clip(ctrl[:7], m.jnt_range[:7, 0], m.jnt_range[:7, 1])
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
            frames)


def build_d5(d0_root, out_root, amp, freq, seed):
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

    writer = LeRobotWriter(out_root, task_str=TASK_STR_D5)
    manifest_eps = []

    try:
        for src_ep in all_idx:
            _, orig_actions, meta = ds0.load_episode(src_ep)
            noisy_actions = jitter_actions(orig_actions, amp, freq, sim.FPS, rng)
            states, actions, frames = rerun_episode_with_action_jitter(
                m, d, water_surface_z, hand_bid, block_bid,
                (rf, rs, rt), (id_front, id_side, id_top), noisy_actions)

            T = writer.save_episode_with_frames(frames, states, actions)
            len_diff = T - len(orig_actions)
            print(f"  D5 episode {writer.n_episodes - 1} 源自 D0 ep{src_ep}: "
                  f"重新仿真 {T} 帧 (原{len(orig_actions)}帧, 差{len_diff:+d})")
            if abs(len_diff) > 0.2 * len(orig_actions):
                print(f"    ⚠️ 帧数差异较大(>20%)，建议检查该episode是否因扰动导致任务提前失败/脱手")

            manifest_eps.append({
                "source_episode_index": int(src_ep),
                "orig_length": int(len(orig_actions)),
                "resim_length": int(T),
            })
    finally:
        rf.close(); rs.close(); rt.close()

    writer.print_summary()

    manifest = {
        "source_dataset": str(Path(d0_root).resolve()),
        "amplitude_rad": amp,
        "frequency_hz": freq,
        "perturbed_dims": "joint ctrl[:7] only, gripper untouched",
        "seed": seed,
        "note": "整条action序列加正弦低幅高频扰动后重新用MuJoCo物理引擎演算，帧密度/条数与D0保持一致（除非扰动导致任务异常终止）",
        "episodes": manifest_eps,
    }
    with open(Path(out_root) / "D5_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"扰动参数已记录到 {Path(out_root) / 'D5_manifest.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d0", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--amp", type=float, default=0.01, help="关节角扰动幅度（弧度）")
    ap.add_argument("--freq", type=float, default=6.0, help="扰动频率（Hz），需明显高于示教动作基频")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    build_d5(args.d0, args.out, args.amp, args.freq, args.seed)


if __name__ == "__main__":
    main()