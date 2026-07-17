"""
eval_openloop_v4.py — 开环评估脚本
将训练数据的 obs 直接喂给模型，比较预测 action 和真实 action 的差距

指标：
  MAE_q    — 关节角预测平均绝对误差（rad）
  MAE_grip — 夹爪预测平均绝对误差
  MSE      — 均方误差

服务器运行: python eval_openloop_v4.py
"""
import json
import numpy as np
import torch
from pathlib import Path
import pyarrow.parquet as pq

# ── 配置 ──────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("/root/demos_lerobot")
MODEL_DIR = Path("/root/autodl-tmp/train_demos_v4/checkpoints/last/pretrained_model")
DEVICE    = "cuda"
N_EP      = 10   # 评估 episode 数量

# ── 加载归一化参数 ─────────────────────────────────────────────────────────────
stats   = json.load(open(str(DATA_DIR / "meta" / "stats.json")))
OBS_MIN = np.array(stats["observation.state"]["min"], dtype=np.float32)
OBS_MAX = np.array(stats["observation.state"]["max"], dtype=np.float32)
ACT_MIN = np.array(stats["action"]["min"], dtype=np.float32)
ACT_MAX = np.array(stats["action"]["max"], dtype=np.float32)

# ── 加载模型 ───────────────────────────────────────────────────────────────────
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
policy = DiffusionPolicy.from_pretrained(str(MODEL_DIR))
policy = policy.to(DEVICE).eval()
print("模型加载成功\n")

# ── 加载数据 ───────────────────────────────────────────────────────────────────
tbl        = pq.read_table(str(DATA_DIR / "data" / "chunk-000" / "file-000.parquet"))
rows       = tbl.to_pydict()
obs_all    = np.array(rows["observation.state"])
acts_all   = np.array(rows["action"])
ep_indices = np.array(rows["episode_index"])

# ── 开环评估循环 ───────────────────────────────────────────────────────────────
all_mae_q    = []
all_mae_grip = []
all_mse      = []

for ep_id in range(N_EP):
    policy.reset()
    mask   = ep_indices == ep_id
    obs_ep = obs_all[mask]
    act_ep = acts_all[mask]
    T      = len(obs_ep)

    mae_q_list    = []
    mae_grip_list = []
    mse_list      = []

    for t in range(T):
        obs    = obs_ep[t].astype(np.float32)
        act_gt = act_ep[t].astype(np.float32)

        # 归一化 obs
        obs_norm = np.clip(
            2 * (obs - OBS_MIN) / (OBS_MAX - OBS_MIN + 1e-8) - 1,
            -1, 1
        ).astype(np.float32)

        # 模型推理
        obs_t = torch.from_numpy(obs_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = policy.select_action({
                "observation.state": obs_t,
                "observation.environment_state": obs_t,
            }).squeeze().cpu().numpy()

        # 反归一化预测 action
        act_pred = (action + 1) / 2 * (ACT_MAX - ACT_MIN) + ACT_MIN

        # 反归一化真实 action（训练数据已归一化存储）
        act_gt_raw = act_gt * (ACT_MAX - ACT_MIN) + ACT_MIN

        # 计算误差
        mae_q    = float(np.mean(np.abs(act_pred[:7] - act_gt_raw[:7])))
        mae_grip = float(np.abs(act_pred[7] - act_gt_raw[7]))
        mse      = float(np.mean((act_pred - act_gt_raw) ** 2))

        mae_q_list.append(mae_q)
        mae_grip_list.append(mae_grip)
        mse_list.append(mse)

    ep_mae_q    = float(np.mean(mae_q_list))
    ep_mae_grip = float(np.mean(mae_grip_list))
    ep_mse      = float(np.mean(mse_list))

    all_mae_q.append(ep_mae_q)
    all_mae_grip.append(ep_mae_grip)
    all_mse.append(ep_mse)

    print(f"Ep{ep_id+1:2d}: "
          f"MAE_q={ep_mae_q:.4f} rad  "
          f"MAE_grip={ep_mae_grip:.3f}  "
          f"MSE={ep_mse:.4f}  "
          f"T={T}")

# ── 汇总 ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"评估 episode 数:    {N_EP}")
print(f"平均 MAE (关节角): {np.mean(all_mae_q):.4f} rad")
print(f"平均 MAE (夹爪):   {np.mean(all_mae_grip):.3f}")
print(f"平均 MSE:          {np.mean(all_mse):.4f}")
print(f"{'='*50}")
