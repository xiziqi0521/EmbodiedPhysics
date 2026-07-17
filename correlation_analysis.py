# -*- coding: utf-8 -*-
"""
correlation_analysis.py —— 相关性分析
=====================================================================
对应文档《六、相关性分析方法》：

    - 数据条数维度：D0（20条）vs D1（10条），2个点，看条数变化对 S_exec 的影响趋势
    - 阶段完整性维度：D0（满帧）vs D2（2帧）vs D3（4帧），3个点
    - 初始状态误差维度：D0（无噪声）vs D4（小噪声），2个点
    - 平滑度维度：把第四节算出的4个候选指标值（D0/D4/D5 共3个点）分别与
      S_exec 做 Spearman 相关系数计算，选相关性最强、方向也符合直觉
      （平滑度越差、成功率越低）的指标作为最终 S_smooth 定义

── 输入 ──────────────────────────────────────────────────────────────────
1. results.csv —— 六组数据集的训练+评估结果，至少包含列：
       dataset, S_exec, S_final
   （对应文档第五节"训练与评估流程"产出，需要手工/评估脚本跑完6组后整理）
   示例：
       dataset,S_exec,S_final
       D0,0.85,78.3
       D1,0.70,65.1
       D2,0.60,55.0
       D3,0.75,70.2
       D4,0.80,74.0
       D5,0.55,50.0

2. smoothness_metrics_summary.csv —— compute_smoothness_metrics.py 的输出
   （D0/D4/D5 × 4个候选指标的组内均值）

── 输出 ──────────────────────────────────────────────────────────────────
- 数据条数/阶段完整性/初始状态误差三个维度的趋势表
- 平滑度维度：4个候选指标分别与 S_exec 的 Spearman 相关系数 + p值，
  并自动挑选"相关性最强、方向符合直觉（指标越差、S_exec越低 → 负相关或
  正相关，取决于指标定义方向）"的指标作为推荐的 S_smooth 定义

用法：
    python correlation_analysis.py --results results.csv \
        --smoothness smoothness_metrics_summary.csv --out correlation_report.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# 每个候选指标"数值越大代表越不平滑"吗？用于判断预期相关方向。
# path_efficiency_ratio: 越大越不平滑 (>1, 1最优)
# jerk_mean_square:      越大越不平滑
# sparc:                 越接近0越平滑，越负越不平滑 -> 数值越大(越接近0)越平滑，即"数值越大越平滑"
# velocity_local_extrema_count: 越大越不平滑
METRIC_DIRECTION = {
    "path_efficiency_ratio": "higher_is_worse",
    "jerk_mean_square": "higher_is_worse",
    "sparc": "higher_is_better",  # 越接近0（越大）越平滑
    "velocity_local_extrema_count": "higher_is_worse",
}


def dimension_trend(results_df, dataset_names, label):
    """打印/返回某个维度上 S_exec、S_final 随数据集变化的简单趋势表"""
    sub = results_df[results_df["dataset"].isin(dataset_names)].copy()
    sub["dataset"] = pd.Categorical(sub["dataset"], categories=dataset_names, ordered=True)
    sub = sub.sort_values("dataset")
    print(f"\n【{label}】")
    print(sub[["dataset", "S_exec", "S_final"]].to_string(index=False))
    return sub[["dataset", "S_exec", "S_final"]].to_dict("records")


def smoothness_correlation(results_df, smooth_df):
    """
    D0/D4/D5 三个点，4个候选指标分别与 S_exec 做 Spearman 相关性分析。
    返回每个指标的 (rho, p_value, 判定是否符合直觉方向)，并挑选最优指标。
    """
    merged = smooth_df.merge(
        results_df[results_df["dataset"].isin(["D0", "D4", "D5"])][["dataset", "S_exec"]],
        on="dataset", how="inner")
    merged = merged.sort_values("dataset")
    print("\n【平滑度候选指标 vs S_exec（D0/D4/D5）】")
    print(merged.to_string(index=False))

    metric_cols = ["path_efficiency_ratio", "jerk_mean_square", "sparc",
                   "velocity_local_extrema_count"]
    results = {}
    for col in metric_cols:
        x = merged[col].values
        y = merged["S_exec"].values
        if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
            rho, p = np.nan, np.nan
        else:
            rho, p = spearmanr(x, y)

        direction = METRIC_DIRECTION[col]
        # 符合直觉：指标变差 -> S_exec 变低
        # higher_is_worse 的指标：期望与 S_exec 负相关 (rho < 0)
        # higher_is_better (sparc) 的指标：期望与 S_exec 正相关 (rho > 0)
        if np.isnan(rho):
            matches_intuition = False
        elif direction == "higher_is_worse":
            matches_intuition = rho < 0
        else:
            matches_intuition = rho > 0

        results[col] = {
            "spearman_rho": None if np.isnan(rho) else float(rho),
            "p_value": None if np.isnan(p) else float(p),
            "direction_definition": direction,
            "matches_intuition": bool(matches_intuition),
        }

    # 挑选：先筛选出方向符合直觉的候选，再按 |rho| 从大到小选最强的
    candidates = {k: v for k, v in results.items() if v["matches_intuition"]}
    if candidates:
        best = max(candidates.items(), key=lambda kv: abs(kv[1]["spearman_rho"]))
    else:
        # 若没有任何指标方向符合直觉（可能因为只有3个点、噪声较大），
        # 退化为在全部指标里选 |rho| 最大的，但明确标注未通过方向检验
        valid = {k: v for k, v in results.items() if v["spearman_rho"] is not None}
        best = max(valid.items(), key=lambda kv: abs(kv[1]["spearman_rho"])) if valid else (None, None)

    print("\n候选指标相关性结果：")
    for k, v in results.items():
        print(f"  {k}: rho={v['spearman_rho']}, p={v['p_value']}, "
              f"方向={v['direction_definition']}, 符合直觉={v['matches_intuition']}")
    if best[0]:
        print(f"\n推荐 S_smooth 指标: {best[0]}  (rho={best[1]['spearman_rho']:.4f}, "
              f"符合直觉={best[1]['matches_intuition']})")
        print("⚠️ 注意：仅3个数据点，样本量偏少，此结论仅作初步参考（详见文档第六节备注）。")

    return results, best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, required=True,
                     help="六组数据集的训练+评估结果 CSV，需含列: dataset,S_exec,S_final")
    ap.add_argument("--smoothness", type=str, required=True,
                     help="compute_smoothness_metrics.py 输出的 *_summary.csv")
    ap.add_argument("--out", type=str, default="correlation_report.json")
    args = ap.parse_args()

    results_df = pd.read_csv(args.results)
    smooth_df = pd.read_csv(args.smoothness)

    required_cols = {"dataset", "S_exec", "S_final"}
    missing = required_cols - set(results_df.columns)
    if missing:
        raise ValueError(f"results.csv 缺少必需列: {missing}")

    report = {}

    # 数据条数维度: D0 vs D1
    report["count_dimension"] = dimension_trend(results_df, ["D0", "D1"], "数据条数维度 D0(20条) vs D1(10条)")

    # 阶段完整性维度: D0 vs D2 vs D3
    report["stage_completeness_dimension"] = dimension_trend(
        results_df, ["D0", "D2", "D3"], "阶段完整性维度 D0(满帧) vs D2(2帧) vs D3(4帧)")

    # 初始状态误差维度: D0 vs D4
    report["init_noise_dimension"] = dimension_trend(
        results_df, ["D0", "D4"], "初始状态误差维度 D0(无噪声) vs D4(小噪声)")

    # 平滑度维度
    smooth_results, best_metric = smoothness_correlation(results_df, smooth_df)
    report["smoothness_dimension"] = {
        "per_metric": smooth_results,
        "recommended_S_smooth_metric": best_metric,
        "sample_size_warning": "仅3个数据点(D0/D4/D5)，样本量偏少，结论仅作初步参考；"
                                "如需更可信结果可增加噪声/扰动档位增加数据点。",
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n完整相关性分析报告已保存到: {args.out}")


if __name__ == "__main__":
    main()
