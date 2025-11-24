# compare_experiments_lib.py
# -*- coding: utf-8 -*-

"""
对两张实验结果表进行对比分析（不用命令行）：

假设两张表是类似结构：
    dataset, method, projection, intersections, ..., bbox_norm,
    ari, nmi, accuracy, macro_f1

例如：
- 表 A：baseline（未加权）
- 表 B：new（加权）

主要入口函数：
- analyze_experiment_tables(table_a, table_b, metrics=None, out_prefix=None)

返回一个 dict:
{
    "merged": merged_df,         # 对齐后的详细表（含 *_A, *_B, *_diff, *_rel_diff）
    "overall": overall_df,       # 全局平均提升
    "by_dataset": by_dataset_df, # 按数据集分组
    "by_method": by_method_df,   # 按聚类方法分组
}
"""

import os
import numpy as np
import pandas as pd
from typing import List, Union, Dict, Optional

# 默认要分析的指标列
DEFAULT_METRICS = ["ari", "nmi", "accuracy", "macro_f1"]


def _to_dataframe(table: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """
    辅助函数：如果传进来是路径字符串，就 read_csv；
    如果本来就是 DataFrame，就直接返回。
    """
    if isinstance(table, str):
        df = pd.read_csv(table)
        print(f"[load] {table}: shape={df.shape}")
        return df
    elif isinstance(table, pd.DataFrame):
        print(f"[load] DataFrame: shape={table.shape}")
        return table.copy()
    else:
        raise TypeError("table_a / table_b 既可以是 CSV 路径(str)，也可以是 pandas.DataFrame")


def _align_tables(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metrics: Optional[List[str]] = None
):
    """
    对齐两张表格：
    - metrics: 要比较的指标列（两表中都必须存在）
    - 其它公共列作为“实验条件 key”

    返回：
    - merged: DataFrame，带有 metric_A, metric_B, metric_diff, metric_rel_diff
    - key_cols: 用于对齐的列名列表
    - metrics: 实际参与比较的指标名列表
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    # 只保留两边都存在的指标
    metrics = [m for m in metrics if (m in df_a.columns and m in df_b.columns)]
    if not metrics:
        raise ValueError("两张表没有共同的指标列，请检查列名（ari/nmi/accuracy/macro_f1 等）。")

    common_cols = sorted(list(set(df_a.columns) & set(df_b.columns)))
    key_cols = [c for c in common_cols if c not in metrics]

    print(f"[align] 使用指标列: {metrics}")
    print(f"[align] 使用 key 列对齐: {key_cols}")

    merged = df_a.merge(
        df_b,
        on=key_cols,
        suffixes=("_A", "_B"),
        how="inner",
    )
    print(f"[align] A: {len(df_a)}, B: {len(df_b)}, merged (inner): {len(merged)}")

    if len(merged) == 0:
        raise ValueError("两张表 inner merge 为空，请确认 key 列一致（dataset/method/特征列/bbox_norm 等）。")

    # 计算差值和相对提升
    for m in metrics:
        col_a = f"{m}_A"
        col_b = f"{m}_B"
        merged[f"{m}_diff"] = merged[col_b] - merged[col_a]

        # 相对提升：对 A=0 的情况保护
        denom = merged[col_a].replace(0, np.nan).astype(float)
        merged[f"{m}_rel_diff"] = merged[f"{m}_diff"] / denom

    return merged, key_cols, metrics


def _summarize_overall(merged: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """
    全局统计：每个指标在所有组合上的平均值 / 平均提升 / win-tie-lose。
    """
    print("\n========== Overall Summary ==========")
    n = len(merged)
    print(f"[overall] 对齐后的实验组合数 = {n}")

    rows = []

    for m in metrics:
        col_a = f"{m}_A"
        col_b = f"{m}_B"
        col_diff = f"{m}_diff"

        mean_a = merged[col_a].mean()
        mean_b = merged[col_b].mean()
        mean_diff = merged[col_diff].mean()

        if mean_a != 0:
            rel_impr = (mean_b - mean_a) / abs(mean_a)
        else:
            rel_impr = np.nan

        improved = (merged[col_diff] > 0).sum()
        tie = (merged[col_diff] == 0).sum()
        worse = (merged[col_diff] < 0).sum()

        print(f"\n[metric: {m}]")
        print(f"  baseline 平均值 (A) = {mean_a:.6f}")
        print(f"  new      平均值 (B) = {mean_b:.6f}")
        print(f"  平均绝对提升 B-A   = {mean_diff:.6f}")
        print(f"  平均相对提升 (B-A)/|A| ≈ {rel_impr:.2%}")
        print(f"  win / tie / lose   = {improved} / {tie} / {worse}")

        rows.append({
            "metric": m,
            "mean_A": mean_a,
            "mean_B": mean_b,
            "mean_diff": mean_diff,
            "mean_rel_diff": rel_impr,
            "win": improved,
            "tie": tie,
            "lose": worse,
            "total": n,
        })

    return pd.DataFrame(rows)


def _summarize_by_group(
    merged: pd.DataFrame,
    metrics: List[str],
    group_col: str,
    title: str
) -> Optional[pd.DataFrame]:
    """
    按某一列（例如 dataset 或 method）分组统计平均提升。
    """
    print(f"\n========== Summary by {group_col} ({title}) ==========")

    if group_col not in merged.columns:
        print(f"[warn] merged 中没有列 {group_col}，跳过该分组统计。")
        return None

    rows = []
    groups = merged[group_col].unique()

    for g in sorted(groups):
        sub = merged[merged[group_col] == g]
        print(f"\n[{group_col} = {g}]  组合数 = {len(sub)}")

        row = {group_col: g, "n": len(sub)}

        for m in metrics:
            col_a = f"{m}_A"
            col_b = f"{m}_B"
            col_diff = f"{m}_diff"

            mean_a = sub[col_a].mean()
            mean_b = sub[col_b].mean()
            mean_diff = sub[col_diff].mean()

            if mean_a != 0:
                rel_impr = (mean_b - mean_a) / abs(mean_a)
            else:
                rel_impr = np.nan

            print(
                f"  [metric {m}] "
                f"A_mean={mean_a:.6f}, "
                f"B_mean={mean_b:.6f}, "
                f"diff={mean_diff:.6f}, "
                f"rel≈{rel_impr:.2%}"
            )

            row[f"{m}_A_mean"] = mean_a
            row[f"{m}_B_mean"] = mean_b
            row[f"{m}_diff_mean"] = mean_diff
            row[f"{m}_rel_diff_mean"] = rel_impr

        rows.append(row)

    return pd.DataFrame(rows)


def analyze_experiment_tables(
    table_a: Union[str, pd.DataFrame],
    table_b: Union[str, pd.DataFrame],
    metrics: Optional[List[str]] = None,
    out_prefix: Optional[str] = None,
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    主入口函数（无需命令行）：

    参数：
    - table_a: baseline 表，可以是 CSV 路径或 DataFrame
    - table_b: new 表，可以是 CSV 路径或 DataFrame
    - metrics: 指标列表（默认为 ["ari","nmi","accuracy","macro_f1"]）
    - out_prefix: 若不为 None，则把若干结果 DataFrame 保存为 CSV

    返回：
    {
        "merged": merged_df,
        "overall": overall_df,
        "by_dataset": by_dataset_df 或 None,
        "by_method": by_method_df 或 None,
    }
    """
    df_a = _to_dataframe(table_a)
    df_b = _to_dataframe(table_b)

    merged, key_cols, metrics_used = _align_tables(df_a, df_b, metrics=metrics)

    overall_df = _summarize_overall(merged, metrics_used)
    by_dataset_df = _summarize_by_group(
        merged, metrics_used, group_col="dataset", title="dataset"
    )
    by_method_df = _summarize_by_group(
        merged, metrics_used, group_col="method", title="method"
    )

    # 如需保存 CSV
    if out_prefix is not None:
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

        merged.to_csv(out_prefix + "_merged_with_diff.csv", index=False)
        overall_df.to_csv(out_prefix + "_summary_overall.csv", index=False)

        if by_dataset_df is not None:
            by_dataset_df.to_csv(out_prefix + "_summary_by_dataset.csv", index=False)
        if by_method_df is not None:
            by_method_df.to_csv(out_prefix + "_summary_by_method.csv", index=False)

        print(f"\n[save] 已保存分析结果到前缀: {out_prefix}_*.csv")

    return {
        "merged": merged,
        "overall": overall_df,
        "by_dataset": by_dataset_df,
        "by_method": by_method_df,
    }

res = analyze_experiment_tables(
    "results/n0s2notst.csv",                                                                                                                               
    "results/n0s2notst_weight.csv",
    metrics=["ari", "nmi", "accuracy", "macro_f1"],  # 可省略，用默认
    out_prefix="results/compare_weighted_vs_unweighted"  # 若不想保存 CSV，可以设为 None
)

merged = res["merged"]
overall = res["overall"]
by_dataset = res["by_dataset"]
by_method = res["by_method"]