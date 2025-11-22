# grid_experiments.py
import os
import csv
import numpy as np
from itertools import combinations

from digit_dataset import load_dataset_from_disk
from features import build_features
from evaluation import evaluate_clustering_full
from clustering_methods import CLUSTERING_METHODS

from sklearn.preprocessing import StandardScaler  # 可选：特征标准化


# =============== 1. 全局实验配置 ===============

# 你现有的数据集目录，可以按自己实际情况修改
DATASET_CONFIGS = [
    {
        "name": "noise0_shift2",
        "dir": "data/digits/noise0_shift2",
    },
]

# 所有可用特征名称（要和 features.py -> make_feature_fns() 里的 key 对应）
ALL_FEATURE_NAMES = [
    "projection",
    "intersections",
    "zoning_4x4",
    "zoning_8x8",
    "global",
    "grad_hist",
    "skeleton",
]

# 是否自动生成所有特征组合
USE_AUTO_FEATURE_CONFIGS = True

# 如果自动生成组合：在这些特征里做幂集
FEATURES_FOR_AUTO_COMBO = [
    "projection",
    "intersections",
    "zoning_4x4",
    "zoning_8x8",
    "global",
    "grad_hist",
    "skeleton",
]

# 限制自动组合的最大特征数（None 表示不限制，即 1~len(...) 的所有子集）
MAX_FEATURE_COMBO_SIZE = 2  # 例如设为 3，则只生成 1/2/3 个特征的组合

# 是否对每个特征组合都同时跑“预处理=True/False”
# PREPROCESS_OPTIONS = [False, True]   # 你也可以改成 [True] 或 [False]
PREPROCESS_OPTIONS = [False]   # 你也可以改成 [True] 或 [False]

# 如果不用自动生成，而是手动配置，就用这个列表（和之前类似）
FEATURE_CONFIGS_MANUAL = [
    {
        "name": "proj+zone+global_no_bbox",
        "features": ["projection", "zoning_4x4", "global"],
        "use_bbox_norm": False,
    },
    {
        "name": "proj+zone+grad+skel_bbox",
        "features": ["projection", "zoning_4x4", "grad_hist", "skeleton"],
        "use_bbox_norm": True,
    },
]

# 聚类方法（来自 clustering_methods.py）
METHODS_TO_RUN = ["maxmin"]  # 你可以加 "agglomerative"

DIGITS = list(range(10))
IMAGE_SIZE = 32
RANDOM_STATE = 0

# 是否对特征做 StandardScaler 标准化
USE_SCALER = False

RESULTS_DIR = "results"
SUMMARY_CSV = os.path.join(RESULTS_DIR, "experiment_summary.csv")


# =============== 2. 生成特征组合配置 ===============

def generate_auto_feature_configs():
    """
    自动生成所有特征组合：
    - 从 FEATURES_FOR_AUTO_COMBO 中生成所有非空子集
    - 每个子集都搭配 PREPROCESS_OPTIONS 中的每一个选项
    """
    configs = []
    feat_names = FEATURES_FOR_AUTO_COMBO

    max_k = len(feat_names)
    if MAX_FEATURE_COMBO_SIZE is not None:
        max_k = min(max_k, MAX_FEATURE_COMBO_SIZE)

    for k in range(1, max_k + 1):
        for subset in combinations(feat_names, k):
            subset = list(subset)
            subset_name = "+".join(subset)
            for use_bbox in PREPROCESS_OPTIONS:
                conf = {
                    "name": f"auto_{subset_name}_{'bbox' if use_bbox else 'no_bbox'}",
                    "features": subset,
                    "use_bbox_norm": use_bbox,
                }
                configs.append(conf)

    return configs


def get_feature_configs():
    """
    根据 USE_AUTO_FEATURE_CONFIGS 决定用自动生成的组合还是手动配置。
    """
    if USE_AUTO_FEATURE_CONFIGS:
        confs = generate_auto_feature_configs()
        print(f"[config] 自动生成了 {len(confs)} 组特征组合。")
        return confs
    else:
        print(f"[config] 使用手动配置的 {len(FEATURE_CONFIGS_MANUAL)} 组特征组合。")
        return FEATURE_CONFIGS_MANUAL


# =============== 3. 单次实验：一个 dataset × 一个 feature × 一个 method ===============

def run_single_experiment(
    dataset_name,
    dataset_dir,
    feature_conf,
    method_name,
    all_feature_names
):
    """
    在某个 数据集 × 特征组合 × 预处理选项 × 聚类方法 上跑一次实验。
    返回一个 summary dict，用于写入 CSV。
    同时在屏幕上打印整体 & per-class 指标。
    """
    features_to_use = feature_conf["features"]
    use_bbox_norm = feature_conf["use_bbox_norm"]
    feature_set_name = feature_conf["name"]

    print("\n" + "=" * 70)
    print(f"Dataset = {dataset_name}, FeatureSet = {feature_set_name}, "
          f"Method = {method_name}, BBoxNorm = {use_bbox_norm}")
    print("=" * 70)

    # 1. 读取数据
    if not os.path.exists(dataset_dir) or not any(os.scandir(dataset_dir)):
        print(f"[WARN] 数据集目录 {dataset_dir} 不存在或为空，跳过该实验。")
        return None

    images, labels = load_dataset_from_disk(
        dataset_dir=dataset_dir,
        digits=DIGITS,
        image_size=IMAGE_SIZE
    )
    n_samples = len(labels)
    print(f"[data] images: {images.shape}, labels: {labels.shape}")

    # 2. 特征提取
    X = build_features(
        images,
        feature_names=features_to_use,
        use_bbox_norm=use_bbox_norm,
        out_size=IMAGE_SIZE
    )

    # 3. 可选：特征标准化
    if USE_SCALER:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print("[feat] 使用 StandardScaler 对特征做标准化。")

    # 4. 聚类
    if method_name not in CLUSTERING_METHODS:
        print(f"[WARN] 未知聚类方法 {method_name}，跳过。")
        return None

    cluster_fn = CLUSTERING_METHODS[method_name]

    if "random_state" in cluster_fn.__code__.co_varnames:
        cluster_labels, model = cluster_fn(
            X,
            n_clusters=len(DIGITS),
            random_state=RANDOM_STATE
        )
    else:
        cluster_labels, model = cluster_fn(
            X,
            n_clusters=len(DIGITS)
        )

    # 5. 评价
    eval_res = evaluate_clustering_full(labels, cluster_labels, DIGITS)

    ari = eval_res["ari"]
    nmi = eval_res["nmi"]
    acc = eval_res["accuracy"]
    f1_list = [eval_res["per_class"][d]["f1"] for d in DIGITS]
    macro_f1 = float(np.mean(f1_list))

    print(f"[result] ARI = {ari:.4f}, NMI = {nmi:.4f}, "
          f"Acc = {acc:.4f}, Macro-F1 = {macro_f1:.4f}")

    # 6. 打印每类详细指标（含真实数 + 预测数）
    print(f"[per-class metrics] (数字: P, R, F1, 真实数, 预测数)")
    for d in DIGITS:
        m = eval_res["per_class"][d]
        print(
            f"  数字 {d}: "
            f"P={m['precision']:.3f}, "
            f"R={m['recall']:.3f}, "
            f"F1={m['f1']:.3f}, "
            f"真实数={m['support']}, "
            f"预测数={m['pred_count']}"
        )

    # 7. 组装 summary 行（用于 CSV）
    summary = {
        "dataset": dataset_name,
        "method": method_name,
        "bbox_norm": "Y" if use_bbox_norm else "N",
        "ari": ari,
        "nmi": nmi,
        "accuracy": acc,
        "macro_f1": macro_f1,
    }

    # 对每个特征，记录 Y/N，列名直接用特征名
    for feat in all_feature_names:
        summary[feat] = "Y" if feat in features_to_use else "N"

    return summary


# =============== 4. 主函数：遍历所有组合，统一跑完并写 CSV ===============

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 生成/选择特征组合配置
    feature_configs = get_feature_configs()

    all_summaries = []

    for ds_conf in DATASET_CONFIGS:
        ds_name = ds_conf["name"]
        ds_dir = ds_conf["dir"]

        for feat_conf in feature_configs:
            for method_name in METHODS_TO_RUN:
                summary = run_single_experiment(
                    dataset_name=ds_name,
                    dataset_dir=ds_dir,
                    feature_conf=feat_conf,
                    method_name=method_name,
                    all_feature_names=ALL_FEATURE_NAMES
                )
                if summary is not None:
                    all_summaries.append(summary)

    if not all_summaries:
        print("[main] 没有成功运行的实验，检查配置。")
        return

    fieldnames = (
        ["dataset", "method"] +
        ALL_FEATURE_NAMES +
        ["bbox_norm", "ari", "nmi", "accuracy", "macro_f1"]
    )

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in all_summaries:
            writer.writerow(s)

    print(f"\n[main] 所有实验已完成，汇总结果已写入 {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
