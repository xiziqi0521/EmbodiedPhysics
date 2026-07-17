"""
eval_v4.py — 闭环评估脚本
统计提升成功率、落地成功率、时间误差

服务器运行: python eval_v4.py
"""
import os
os.environ.pop('MUJOCO_GL', None)
import json, numpy as np, mujoco, torch
import xml.etree.ElementTree as ET
from pathlib import Path
import pyarrow.parquet as pq

# ── 配置 ──────────────────────────────────────────────────────────────────────
FRANKA_DIR  = "/root/mujoco_menagerie/franka_emika_panda"
MODEL_DIR   = Path("/root/autodl-tmp/train_demos_v4/checkpoints/last/pretrained_model")
DATA_DIR    = Path("/root/demos_lerobot")
BLOCK_SIZE  = 0.025
BLOCK_MASS  = 0.10
GRIPPER_OPEN = 0.040
DEVICE      = "cuda"
SUBSAMPLE   = 8
N_EP        = 10      # 评估 episode 数量

GRASP_WAIT           = 150
FORCE_RELEASE_HEIGHT = 0.30
FORCE_RELEASE_STEPS  = 500
SMOOTH_ALPHA         = 0.7
BLOCK_POS            = np.array([0.409, 0.006, BLOCK_SIZE])

# ── 加载归一化参数 ─────────────────────────────────────────────────────────────
stats   = json.load(open(str(DATA_DIR / "meta" / "stats.json")))
OBS_MIN = np.array(stats["observation.state"]["min"], dtype=np.float32)
OBS_MAX = np.array(stats["observation.state"]["max"], dtype=np.float32)
ACT_MIN = np.array(stats["action"]["min"], dtype=np.float32)
ACT_MAX = np.array(stats["action"]["max"], dtype=np.float32)


def build_scene(block_pos):
    panda_tree = ET.parse(f"{FRANKA_DIR}/panda.xml")
    pr = panda_tree.getroot()
    for kf in pr.findall("keyframe"): pr.remove(kf)
    default_el = pr.find("default")
    if default_el is not None:
        inner  = default_el.find("default")
        target = inner if inner is not None else default_el
        ge = ET.SubElement(target, "geom")
    else:
        de = ET.SubElement(pr, "default")
        ge = ET.SubElement(de, "geom")
    ge.set("friction", "2.5 0.5 0.5")
    ge.set("condim",   "4")
    panda_tree.write(f"{FRANKA_DIR}/_ip.xml", encoding="unicode")

    scene_tree = ET.parse(f"{FRANKA_DIR}/scene.xml")
    sr = scene_tree.getroot()
    for inc in sr.findall("include"): inc.set("file", "_ip.xml")
    for kf  in sr.findall("keyframe"): sr.remove(kf)
    wb = sr.find("worldbody")
    bx, by, bz = block_pos
    wb.append(ET.fromstring(f"""
    <body name="block" pos="{bx:.4f} {by:.4f} {bz:.4f}">
      <joint name="block_free" type="free"/>
      <geom name="block_geom" type="box"
            size="{BLOCK_SIZE:.4f} {BLOCK_SIZE:.4f} {BLOCK_SIZE:.4f}"
            mass="{BLOCK_MASS:.4f}" friction="2.0 0.5 0.5" condim="4"
            solimp="0.99 0.999 0.001" solref="0.004 1" rgba="0.15 0.50 0.90 1"/>
    </body>"""))
    scene_tree.write(f"{FRANKA_DIR}/_is.xml", encoding="unicode")
    m = mujoco.MjModel.from_xml_path(f"{FRANKA_DIR}/_is.xml")
    return m, mujoco.MjData(m)


# ── 加载模型 ───────────────────────────────────────────────────────────────────
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
policy = DiffusionPolicy.from_pretrained(str(MODEL_DIR))
policy = policy.to(DEVICE).eval()
print("模型加载成功\n")

tbl        = pq.read_table(str(DATA_DIR / "data" / "chunk-000" / "file-000.parquet"))
rows       = tbl.to_pydict()
obs_all    = np.array(rows["observation.state"])
ep_indices = np.array(rows["episode_index"])

lift_ok     = 0
land_ok     = 0
ff_ok       = 0   # 自由落体验证通过（误差<20%）
time_errors = []

for ep_id in range(N_EP):
    policy.reset()
    src_ep = ep_id % 50
    start  = int(np.where(ep_indices == src_ep)[0][0])
    obs0   = obs_all[start]
    block_pos = BLOCK_POS.copy()

    m, d = build_scene(block_pos)
    blk_jid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "block_free")
    qpos_adr = m.jnt_qposadr[blk_jid]
    qvel_adr = m.jnt_dofadr[blk_jid]

    d.qpos[:7]  = obs0[:7];  d.ctrl[:7]  = obs0[:7]
    d.ctrl[7:9] = obs0[7] * GRIPPER_OPEN
    d.qpos[7:9] = obs0[7] * GRIPPER_OPEN
    d.qpos[qpos_adr:qpos_adr+3] = block_pos
    d.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    released    = False
    landed      = False
    lifted      = False
    prev_grip   = obs0[7] * GRIPPER_OPEN
    release_z   = 0
    release_t   = 0
    grasp_wait  = 0
    is_grasping = False
    grasped     = False
    high_steps  = 0
    force_released = False
    smoothed_q  = obs0[:7].copy()

    for step in range(12000):
        blk_z  = float(d.qpos[qpos_adr+2])
        blk_vz = float(d.qvel[qvel_adr+2])
        grip   = float(np.mean(d.qpos[7:9]))

        if blk_z > 0.1 and not lifted:
            lifted = True
            lift_ok += 1

        if blk_z > FORCE_RELEASE_HEIGHT:
            high_steps += 1
        else:
            if not force_released:
                high_steps = 0

        if high_steps >= FORCE_RELEASE_STEPS and not force_released:
            force_released = True
            release_z = blk_z
            release_t = float(d.time)

        if blk_z > 0.08 and grip > 0.02 and prev_grip < 0.01 and not released:
            released  = True
            release_z = blk_z
            release_t = float(d.time)

        if (released or force_released) and not landed \
                and blk_z < BLOCK_SIZE + 0.015 and abs(blk_vz) < 2.0:
            ok = True
            if force_released and not released:
                ok = (grip > 0.01)
            if ok:
                landed = True
                land_ok += 1
                t_a = float(d.time) - release_t
                t_t = np.sqrt(2 * max(release_z - BLOCK_SIZE, 0) / 9.81)
                err = abs(t_a - t_t) / max(t_t, 1e-6) * 100
                time_errors.append(err)
                if err < 20:
                    ff_ok += 1
                icon = "✅" if err < 20 else "⚠️"
                print(f"Ep{ep_id+1:2d}: {icon} "
                      f"提升={'✅' if lifted else '❌'} "
                      f"落地=✅ "
                      f"h={release_z:.3f}m "
                      f"理论={t_t:.3f}s 实测={t_a:.3f}s 误差={err:.1f}%")
                break

        prev_grip = grip

        if step % SUBSAMPLE == 0:
            arm_q  = d.qpos[:7].astype(np.float32)
            grip_v = np.float32(grip / GRIPPER_OPEN)
            blk_p  = d.qpos[qpos_adr:qpos_adr+3].astype(np.float32)
            blk_v  = d.qvel[qvel_adr:qvel_adr+3].astype(np.float32)
            obs    = np.concatenate([arm_q, [grip_v], blk_p, blk_v])
            obs_norm = np.clip(
                2 * (obs - OBS_MIN) / (OBS_MAX - OBS_MIN + 1e-8) - 1,
                -1, 1
            ).astype(np.float32)
            obs_t = torch.from_numpy(obs_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
            batch = {"observation.state": obs_t, "observation.environment_state": obs_t}
            with torch.no_grad():
                action = policy.select_action(batch).squeeze().cpu().numpy()
            act_raw  = (action + 1) / 2 * (ACT_MAX - ACT_MIN) + ACT_MIN
            act_grip = float(np.clip(act_raw[7], 0, 1))
            smoothed_q = SMOOTH_ALPHA * act_raw[:7] + (1 - SMOOTH_ALPHA) * smoothed_q

            if force_released:
                d.ctrl[7]  = 255.0
                d.ctrl[:7] = np.clip(smoothed_q, m.jnt_range[:7, 0], m.jnt_range[:7, 1])
            elif act_grip < 0.01 and not is_grasping and not grasped and blk_z < 0.1:
                is_grasping = True
                grasp_wait  = 0
                smoothed_q  = d.qpos[:7].copy()
                d.ctrl[:7]  = d.qpos[:7]
                d.ctrl[7]   = 0.0
            elif is_grasping and grasp_wait < GRASP_WAIT:
                d.ctrl[:7]  = d.qpos[:7]
                d.ctrl[7]   = 0.0
                smoothed_q  = d.qpos[:7].copy()
                grasp_wait += 1
                if grasp_wait == GRASP_WAIT:
                    grasped = True
            else:
                d.ctrl[:7] = np.clip(smoothed_q, m.jnt_range[:7, 0], m.jnt_range[:7, 1])
                d.ctrl[7]  = act_grip * 255.0

        mujoco.mj_step(m, d)

    if not landed:
        print(f"Ep{ep_id+1:2d}: ❌ 提升={'✅' if lifted else '❌'} 落地=❌")

# ── 评估结果汇总 ──────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"评估 episode 数: {N_EP}")
print(f"提升成功率:       {lift_ok}/{N_EP} ({lift_ok/N_EP*100:.0f}%)")
print(f"落地成功率:       {land_ok}/{N_EP} ({land_ok/N_EP*100:.0f}%)")
if time_errors:
    print(f"平均时间误差:     {np.mean(time_errors):.1f}%")
    print(f"最大时间误差:     {np.max(time_errors):.1f}%")
    print(f"最小时间误差:     {np.min(time_errors):.1f}%")
    print(f"自由落体验证通过: {ff_ok}/{len(time_errors)} (误差<20%)")
print(f"{'='*50}")
