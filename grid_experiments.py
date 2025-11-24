# grid_experiments.py
import os
import csv
import numpy as np
import math
from itertools import combinations

from digit_dataset import load_dataset_from_disk
from features import make_feature_fns, crop_and_resize_to_square
from evaluation import evaluate_clustering_full
from clustering_methods import CLUSTERING_METHODS

from sklearn.preprocessing import StandardScaler  # 可选：特征标准化


# =============== 1. 全局实验配置 ===============

DATASET_CONFIGS = [

    {"name": "noise0_shift2",   "dir": "data/digits/noise0_shift2"},

]

# 所有可用特征名，要和 features.py -> make_feature_fns() 里的 key 对应
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

# 自动生成组合时使用的特征子集
FEATURES_FOR_AUTO_COMBO = [
    "projection",
    "intersections",
    "zoning_4x4",
    "zoning_8x8",
    "global",
    "grad_hist",
    "skeleton",
]

# 限制组合中最多包含多少种特征（None 表示 1~全部）
MAX_FEATURE_COMBO_SIZE = None  # 比如设为 3，则只生成 1/2/3 个特征的组合

# 特征组合是否使用 bbox 归一化（预处理）的选项
PREPROCESS_OPTIONS = [False, True]

# 是否在拼接前按“特征块”缩放，减弱多维特征块的主导作用
USE_BLOCK_SCALING = True

# 手动配置（当 USE_AUTO_FEATURE_CONFIGS=False 时才用）
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

# 聚类方法
METHODS_TO_RUN = ["kmeans", "maxmin", "agglomerative"]  # 你也可以加 "agglomerative"

DIGITS = list(range(10))
IMAGE_SIZE = 32
RANDOM_STATE = 0

# 是否额外对特征做 StandardScaler 标准化
USE_SCALER = False
# USE_SCALER = True

RESULTS_DIR = "results"
SUMMARY_CSV = os.path.join(RESULTS_DIR, "experiment_summary.csv")

# 特征 / 预处理缓存目录
CACHE_DIR = "cache"


# =============== 2. 一些工具函数 ===============

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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

def get_block_weight(feature_name, dim):
    """
    给某个特征块一个权重。
    默认策略：w = 1 / sqrt(dim)
    这样每块特征对距离的平均贡献不会因为“维度多”而过大。
    你也可以根据 feature_name 做手动调整。
    """
    if not USE_BLOCK_SCALING:
        return 1.0

    # 如果想对某些特征特别调权，可以在这里加：
    # if feature_name == "skeleton":
    #     return 2.0 / math.sqrt(dim)
    # 先用简单版：
    return 1.0 / math.sqrt(dim)

def get_preprocessed_images_cached(dataset_name, use_bbox_norm, images, image_size, cache_dir=CACHE_DIR):
    """
    根据 use_bbox_norm 决定是否做 bbox 裁剪 + resize；
    结果缓存在 cache/ 下的 .npy 文件中。
    """
    ensure_dir(cache_dir)
    tag = "bbox" if use_bbox_norm else "raw"
    cache_path = os.path.join(cache_dir, f"{dataset_name}_{tag}_images_{image_size}.npy")

    if os.path.exists(cache_path):
        imgs = np.load(cache_path)
        print(f"[cache] 读取预处理图像: {cache_path} shape={imgs.shape}")
        return imgs

    # 重新计算
    if use_bbox_norm:
        print(f"[prep] 计算预处理图像 (bbox_norm=True) for {dataset_name} ...")
        proc_list = [
            crop_and_resize_to_square(img, out_size=image_size)
            for img in images
        ]
        imgs = np.stack(proc_list).astype(np.float32)
    else:
        print(f"[prep] 使用原始图像 (bbox_norm=False) for {dataset_name} ...")
        # 假定 images 已经是 float32 [0,1]
        imgs = images.astype(np.float32)

    np.save(cache_path, imgs)
    print(f"[cache] 保存预处理图像到: {cache_path}")
    return imgs


def get_feature_matrix_cached(dataset_name, use_bbox_norm, feature_name,
                              images_proc, feature_fn, cache_dir=CACHE_DIR):
    """
    对某个数据集 + 某个预处理方式 + 某种原子特征：
    - 如果缓存存在，直接 load；
    - 否则对所有图像逐个计算特征，stack 后保存。
    """
    ensure_dir(cache_dir)
    tag = "bbox" if use_bbox_norm else "raw"
    cache_path = os.path.join(cache_dir, f"{dataset_name}_{tag}_feat_{feature_name}.npy")

    if os.path.exists(cache_path):
        F = np.load(cache_path)
        print(f"[cache] 读取特征 {feature_name}: {cache_path} shape={F.shape}")
        return F

    print(f"[feat] 计算特征 {feature_name} for {dataset_name}, bbox_norm={use_bbox_norm} ...")
    feats = [feature_fn(img) for img in images_proc]
    F = np.stack(feats).astype(np.float32)
    np.save(cache_path, F)
    print(f"[cache] 保存特征 {feature_name} 到: {cache_path}")
    return F


# =============== 3. 单个方法在某个 X 上跑一次聚类 ===============

def run_single_method(
    dataset_name,
    method_name,
    X,
    labels,
    use_bbox_norm,
    features_to_use,
    all_feature_names
):
    """
    在给定特征矩阵 X + 真实标签 labels 上，使用指定 method 做一次聚类。
    只负责：
      - 可选标准化
      - 调用聚类方法
      - 调用评价函数
      - 打印指标
      - 返回一行 summary（用于写 CSV）
    """
    print("\n" + "-" * 60)
    print(f"Dataset={dataset_name}, Method={method_name}, "
          f"BBoxNorm={'Y' if use_bbox_norm else 'N'}, "
          f"Features={features_to_use}")
    print("-" * 60)

    X_used = X
    if USE_SCALER:
        scaler = StandardScaler()
        X_used = scaler.fit_transform(X_used)
        print("[feat] 使用 StandardScaler 对特征做标准化。")

    if method_name not in CLUSTERING_METHODS:
        print(f"[WARN] 未知聚类方法 {method_name}，跳过。")
        return None

    cluster_fn = CLUSTERING_METHODS[method_name]

    # 聚类
    if "random_state" in cluster_fn.__code__.co_varnames:
        cluster_labels, model = cluster_fn(
            X_used,
            n_clusters=len(DIGITS),
            random_state=RANDOM_STATE
        )
    else:
        cluster_labels, model = cluster_fn(
            X_used,
            n_clusters=len(DIGITS)
        )

    # 评价
    eval_res = evaluate_clustering_full(labels, cluster_labels, DIGITS)
    ari = eval_res["ari"]
    nmi = eval_res["nmi"]
    acc = eval_res["accuracy"]
    f1_list = [eval_res["per_class"][d]["f1"] for d in DIGITS]
    macro_f1 = float(np.mean(f1_list))

    print(f"[result] ARI = {ari:.4f}, NMI = {nmi:.4f}, "
          f"Acc = {acc:.4f}, Macro-F1 = {macro_f1:.4f}")

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

    # 组装 CSV summary 行
    summary = {
        "dataset": dataset_name,
        "method": method_name,
        "bbox_norm": "Y" if use_bbox_norm else "N",
        "ari": ari,
        "nmi": nmi,
        "accuracy": acc,
        "macro_f1": macro_f1,
    }

    # 每个特征列用 Y/N 表示是否使用
    for feat in all_feature_names:
        summary[feat] = "Y" if feat in features_to_use else "N"

    return summary


# =============== 4. 主函数：按数据集 / 预处理 / 特征组合 / 方法遍历 ===============

def main():
    ensure_dir(RESULTS_DIR)
    ensure_dir(CACHE_DIR)

    feature_configs = get_feature_configs()
    all_summaries = []

    # 对每个数据集只加载一次
    for ds_conf in DATASET_CONFIGS:
        ds_name = ds_conf["name"]
        ds_dir = ds_conf["dir"]

        if not os.path.exists(ds_dir) or not any(os.scandir(ds_dir)):
            print(f"[WARN] 数据集目录 {ds_dir} 不存在或为空，跳过 {ds_name}。")
            continue

        print("\n" + "=" * 80)
        print(f"[dataset] 加载数据集: {ds_name} from {ds_dir}")
        print("=" * 80)

        images_raw, labels = load_dataset_from_disk(
            dataset_dir=ds_dir,
            digits=DIGITS,
            image_size=IMAGE_SIZE
        )
        n_samples = len(labels)
        print(f"[data] images: {images_raw.shape}, labels: {labels.shape}, n_samples={n_samples}")

        # 准备所有原子特征函数
        feature_fns = make_feature_fns()

        # 该数据集实际用到了哪些 bbox_norm（从 feature_configs 里统计）
        used_bbox_values = sorted(set(conf["use_bbox_norm"] for conf in feature_configs))

        # 对每种 bbox_norm 预处理：先预处理图像，再预先算出所有原子特征矩阵
        for use_bbox_norm in used_bbox_values:
            # 1) 预处理图像（带缓存）
            images_proc = get_preprocessed_images_cached(
                dataset_name=ds_name,
                use_bbox_norm=use_bbox_norm,
                images=images_raw,
                image_size=IMAGE_SIZE,
                cache_dir=CACHE_DIR
            )

            # 2) 为当前预处理方式计算 / 读取所有原子特征矩阵
            feature_mats = {}
            for feat_name in ALL_FEATURE_NAMES:
                feature_fn = feature_fns[feat_name]
                F = get_feature_matrix_cached(
                    dataset_name=ds_name,
                    use_bbox_norm=use_bbox_norm,
                    feature_name=feat_name,
                    images_proc=images_proc,
                    feature_fn=feature_fn,
                    cache_dir=CACHE_DIR
                )
                # 根据维度计算一个块级权重
                w = get_block_weight(feat_name, F.shape[1])
                F_scaled = F * w
                feature_mats[feat_name] = F_scaled

            # 3) 对所有在该 bbox_norm 下的特征组合 × 聚类方法 进行实验
            for feat_conf in feature_configs:
                if feat_conf["use_bbox_norm"] != use_bbox_norm:
                    continue

                features_to_use = feat_conf["features"]

                # 拼接该组合的特征矩阵
                X_parts = [feature_mats[f] for f in features_to_use]
                X = np.concatenate(X_parts, axis=1)

                for method_name in METHODS_TO_RUN:
                    summary = run_single_method(
                        dataset_name=ds_name,
                        method_name=method_name,
                        X=X,
                        labels=labels,
                        use_bbox_norm=use_bbox_norm,
                        features_to_use=features_to_use,
                        all_feature_names=ALL_FEATURE_NAMES,
                    )
                    if summary is not None:
                        all_summaries.append(summary)

    if not all_summaries:
        print("[main] 没有成功运行的实验，检查配置。")
        return

    # 写 CSV 表头：数据集名 + 每个特征的 Y/N + 预处理 Y/N + 指标
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
