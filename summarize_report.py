# -*- coding: utf-8 -*-
"""
summarize_report.py —— 产出物汇总
=====================================================================
对应文档《七、产出物清单》，把前面各脚本的输出整合成一份 Markdown 报告：

    - 每个场景的 6 组数据集说明（构造方式、条数、抽帧/噪声参数）
    - 每个场景的平滑度候选指标数值表（D0/D4/D5 × 4个指标）
    - 每个场景的模型表现结果表（6组的 S_exec、S_final）
    - 相关系数结果表 + 最终选定的 S_smooth 定义和理由

用法：
    python summarize_report.py \
        --scene_name scene_07_buoyancy \
        --d1_manifest D1/D1_manifest.json \
        --d2_manifest D2/stage_split_manifest_n2.json \
        --d3_manifest D3/stage_split_manifest_n4.json \
        --d4_manifest D4/D4_manifest.json \
        --d5_manifest D5/D5_manifest.json \
        --results results.csv \
        --smoothness_summary smoothness_metrics_summary.csv \
        --correlation_report correlation_report.json \
        --out report.md
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(path):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r") as f:
        return json.load(f)


def build_dataset_construction_table(d1, d2, d3, d4, d5):
    rows = []
    rows.append(["D0", "基线", "现有原始demo，不做任何处理", "-", "原始数据"])
    if d1:
        rows.append(["D1", "数据条数-10条",
                     f"抽取规则: {d1.get('rule')}", f"{d1.get('n_episodes')} 条",
                     f"源: {d1.get('selected_source_episode_indices')}"])
    if d2:
        rows.append(["D2", "阶段完整性-2帧",
                     f"每阶段抽 {d2.get('n_per_stage')} 帧", "-",
                     d2.get("stage_definition", "")])
    if d3:
        rows.append(["D3", "阶段完整性-4帧",
                     f"每阶段抽 {d3.get('n_per_stage')} 帧", "-",
                     d3.get("stage_definition", "")])
    if d4:
        rows.append(["D4", "初始状态误差-小噪声",
                     f"初始位置高斯扰动 sigma={d4.get('sigma_meters')}m", "-",
                     d4.get("noise_dims", "")])
    if d5:
        rows.append(["D5", "轨迹平滑度扰动",
                     f"amp={d5.get('amplitude_rad')}rad, freq={d5.get('frequency_hz')}Hz",
                     "-", d5.get("perturbed_dims", "")])
    return pd.DataFrame(rows, columns=["编号", "名称", "构造方式", "规模", "备注"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_name", type=str, required=True)
    ap.add_argument("--d1_manifest", type=str, default=None)
    ap.add_argument("--d2_manifest", type=str, default=None)
    ap.add_argument("--d3_manifest", type=str, default=None)
    ap.add_argument("--d4_manifest", type=str, default=None)
    ap.add_argument("--d5_manifest", type=str, default=None)
    ap.add_argument("--results", type=str, required=True)
    ap.add_argument("--smoothness_summary", type=str, required=True)
    ap.add_argument("--correlation_report", type=str, required=True)
    ap.add_argument("--out", type=str, default="report.md")
    args = ap.parse_args()

    d1 = load_json(args.d1_manifest)
    d2 = load_json(args.d2_manifest)
    d3 = load_json(args.d3_manifest)
    d4 = load_json(args.d4_manifest)
    d5 = load_json(args.d5_manifest)
    results_df = pd.read_csv(args.results)
    smooth_df = pd.read_csv(args.smoothness_summary)
    corr = load_json(args.correlation_report)

    construction_table = build_dataset_construction_table(d1, d2, d3, d4, d5)

    lines = []
    lines.append(f"# 数据质量层评估报告 —— {args.scene_name}\n")

    lines.append("## 一、6组数据集构造说明\n")
    lines.append(construction_table.to_markdown(index=False))
    lines.append("")

    lines.append("## 二、平滑度候选指标数值表（D0/D4/D5 × 4个指标）\n")
    lines.append(smooth_df.to_markdown(index=False))
    lines.append("")

    lines.append("## 三、模型表现结果表（6组 S_exec / S_final）\n")
    lines.append(results_df.to_markdown(index=False))
    lines.append("")

    lines.append("## 四、相关性分析结果\n")
    if corr:
        lines.append("### 4.1 数据条数维度 (D0 vs D1)\n")
        lines.append(pd.DataFrame(corr["count_dimension"]).to_markdown(index=False))
        lines.append("")
        lines.append("### 4.2 阶段完整性维度 (D0 vs D2 vs D3)\n")
        lines.append(pd.DataFrame(corr["stage_completeness_dimension"]).to_markdown(index=False))
        lines.append("")
        lines.append("### 4.3 初始状态误差维度 (D0 vs D4)\n")
        lines.append(pd.DataFrame(corr["init_noise_dimension"]).to_markdown(index=False))
        lines.append("")
        lines.append("### 4.4 平滑度维度：候选指标 Spearman 相关系数\n")
        sm = corr["smoothness_dimension"]
        metric_rows = []
        for k, v in sm["per_metric"].items():
            metric_rows.append({
                "指标": k,
                "spearman_rho": v["spearman_rho"],
                "p_value": v["p_value"],
                "方向定义": v["direction_definition"],
                "符合直觉": v["matches_intuition"],
            })
        lines.append(pd.DataFrame(metric_rows).to_markdown(index=False))
        lines.append("")
        lines.append(f"**推荐 S_smooth 定义**: `{sm['recommended_S_smooth_metric']}`\n")
        lines.append(f"> {sm['sample_size_warning']}\n")

    lines.append("## 五、产出物索引\n")
    lines.append("- 数据集构造 manifest: D1_manifest.json / stage_split_manifest_n2.json / "
                  "stage_split_manifest_n4.json / D4_manifest.json / D5_manifest.json")
    lines.append("- 平滑度逐episode指标: smoothness_metrics.csv")
    lines.append("- 平滑度组内均值: smoothness_metrics_summary.csv")
    lines.append("- 模型表现结果: results.csv（需按文档第五节流程手工训练+评估后填写）")
    lines.append("- 相关性分析完整报告: correlation_report.json")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已生成: {out_path}")


if __name__ == "__main__":
    main()
