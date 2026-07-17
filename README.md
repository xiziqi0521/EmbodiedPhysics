# 数据质量层评估 —— 浮力实验（07号场景）代码

对应《数据质量层评估——实验操作说明》V2 文档，围绕已录制好的 D0（20条原始
demo，lerobot v3.0 格式）构造 D1~D5，计算平滑度候选指标并做相关性分析。

## 环境要求

- **需要 MuJoCo 的脚本**（在你录制 D0 用的同一台 Windows 机器上跑）：
  `build_D4_init_noise.py`、`build_D5_action_jitter.py`
  （它们 `import teleop_buoyancy_lerobot as sim`，复用你原脚本里的
  `build_model` / `apply_buoyancy` / 浮力常量，保证物理一致性）
- **不需要 MuJoCo 的脚本**（纯数据后处理/数值分析，可在任意有 python 环境的机
  器上跑，只需要 `pandas pyarrow scipy imageio[ffmpeg] tabulate`）：
  `build_D1_count10.py`、`build_D2_D3_stage_frames.py`、
  `compute_smoothness_metrics.py`、`correlation_analysis.py`、
  `summarize_report.py`

所有脚本共用 `lerobot_io.py`（数据集读写工具），必须放在同一目录下。

## 完整运行顺序

```bash
# 假设 D0 = D:\mujuco\demos_buoyancy，已有20条episode

# ── 第一步：构造 D1~D5（对应文档第二节）───────────────────────────────
python build_D1_count10.py --d0 D:/mujuco/demos_buoyancy --out D:/mujuco/demos_D1

python build_D2_D3_stage_frames.py --d0 D:/mujuco/demos_buoyancy --out D:/mujuco/demos_D2 --n_per_stage 2
python build_D2_D3_stage_frames.py --d0 D:/mujuco/demos_buoyancy --out D:/mujuco/demos_D3 --n_per_stage 4

# D4/D5 需要重新仿真，必须在装有MuJoCo的环境跑
python build_D4_init_noise.py --d0 D:/mujuco/demos_buoyancy --out D:/mujuco/demos_D4 --sigma 0.006
python build_D5_action_jitter.py --d0 D:/mujuco/demos_buoyancy --out D:/mujuco/demos_D5 --amp 0.01 --freq 6.0

# ── 第二步：训练 + 评估（对应文档第五节，需你已有的DP训练/评估脚本）──────
# 每组数据集(D0~D5)各训练一次DP模型，跑闭环评估，把结果整理成 results.csv：
#   dataset,S_exec,S_final
#   D0,0.85,78.3
#   D1,0.70,65.1
#   ...
# 这一步文档明确说"暂不加随机种子"，且不在本代码包范围内（依赖你现有的
# 训练/评估pipeline），需要你手工跑完6组后填写 results.csv

# ── 第三步：平滑度候选指标（对应文档第四节，只在D0/D4/D5上算）──────────
python compute_smoothness_metrics.py \
    --d0 D:/mujuco/demos_buoyancy --d4 D:/mujuco/demos_D4 --d5 D:/mujuco/demos_D5 \
    --out smoothness_metrics.csv --fps 15
# 产出: smoothness_metrics.csv（逐episode） + smoothness_metrics_summary.csv（组内均值）

# ── 第四步：相关性分析（对应文档第六节）─────────────────────────────
python correlation_analysis.py \
    --results results.csv \
    --smoothness smoothness_metrics_summary.csv \
    --out correlation_report.json

# ── 第五步：汇总报告（对应文档第七节产出物清单）─────────────────────
python summarize_report.py --scene_name scene_07_buoyancy \
    --d1_manifest D:/mujuco/demos_D1/D1_manifest.json \
    --d2_manifest D:/mujuco/demos_D2/stage_split_manifest_n2.json \
    --d3_manifest D:/mujuco/demos_D3/stage_split_manifest_n4.json \
    --d4_manifest D:/mujuco/demos_D4/D4_manifest.json \
    --d5_manifest D:/mujuco/demos_D5/D5_manifest.json \
    --results results.csv \
    --smoothness_summary smoothness_metrics_summary.csv \
    --correlation_report correlation_report.json \
    --out report.md
```

## 各数据集构造规则对照文档

| 编号 | 文档要求 | 本代码实现 |
|---|---|---|
| D1 | 固定抽10条，写清规则以便复现 | 按 `episode_index` 偶数抽取（`--rule even`，可选 `first10`），规则写入 `D1_manifest.json` |
| D2/D3 | 先定阶段切分规则，阶段内均匀抽帧 | 按 gripper闭合时刻 / 方块触水时刻 切3阶段，阶段内 `np.linspace` 均匀抽帧，规则写入 `stage_split_manifest_n{2,4}.json` |
| D4 | 只扰动首帧state，之后重新仿真 | 对方块初始(x,y)加高斯扰动（z由浮力平衡决定），原action序列在新初始状态下用MuJoCo重新逐步积分 |
| D5 | 整条action加低幅高频扰动，重新仿真 | 对`ctrl[:7]`（不含gripper）叠加正弦扰动（默认0.01rad, 6Hz），重新用MuJoCo逐步积分 |

## 已知的人工决策点（文档中留给你判断、代码里已给出默认值，可按需调整）

1. **D1抽取规则**：默认按episode_index偶数抽取，可改 `--rule first10`
2. **D2/D3阶段切分规则**：默认按"gripper首次闭合"和"方块首次触水"切3段，
   这是我基于任务物理特征给出的定义，如果你们已有其他约定的阶段划分标准，
   需要替换 `build_D2_D3_stage_frames.py` 里的 `split_stages()` 函数
3. **D4噪声幅度**：默认 sigma=6mm，量级约为球半径(25mm)的1/4，可用`--sigma`调整
4. **D5扰动幅度/频率**：默认 amp=0.01rad, freq=6Hz，可用`--amp --freq`调整；
   建议先跑1~2条看重新仿真后是否仍能完成抓取任务，再决定是否需要调参重录全部20条

## 本次未包含的部分

文档第五节"训练与评估流程"依赖你现有的DP训练pipeline和闭环评估脚本，
不在这批代码范围内；`results.csv` 需要你在跑完6组训练+评估后手工整理。
