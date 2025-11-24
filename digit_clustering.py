# digit_clustering.py
import os
import numpy as np

from digit_dataset import (
    generate_and_save_dataset,
    load_dataset_from_disk,
)
from clustering_methods import CLUSTERING_METHODS
from features import build_features
from evaluation import evaluate_clustering_full
from visualization import save_pca_scatter, save_cluster_examples


def run_one_method(
    method_name,
    X,
    images,
    labels,
    digits,
    results_dir="results",
    n_clusters=10,
    random_state=0
):
    """
    在同一个特征 X 上，用指定聚类方法跑一次完整实验。
    """
    if method_name not in CLUSTERING_METHODS:
        raise ValueError(f"未知聚类方法: {method_name}")

    cluster_fn = CLUSTERING_METHODS[method_name]

    # 1. 聚类
    print(f"\n===== 使用方法: {method_name} =====")
    # 简单处理一下 random_state 参数是否存在
    if "random_state" in cluster_fn.__code__.co_varnames:
        cluster_labels, model = cluster_fn(
            X,
            n_clusters=n_clusters,
            random_state=random_state
        )
    else:
        cluster_labels, model = cluster_fn(
            X,
            n_clusters=n_clusters
        )
    print(f"[{method_name}] 聚类完成。")

    # 2. 评价
    eval_res = evaluate_clustering_full(labels, cluster_labels, digits)

    print(f"[{method_name}] ARI = {eval_res['ari']:.4f}")
    print(f"[{method_name}] NMI = {eval_res['nmi']:.4f}")
    print(f"[{method_name}] 映射后的准确率 = {eval_res['accuracy']:.4f}")
    print(f"[{method_name}] 每类指标:")
    for d in digits:
        m = eval_res["per_class"][d]
        print(
            f"  数字 {d}: "
            f"P={m['precision']:.3f}, "
            f"R={m['recall']:.3f}, "
            f"F1={m['f1']:.3f}, "
            f"真实数={m['support']}, "
            f"预测数={m['pred_count']}"
        )

    # 3. 可视化保存
    os.makedirs(results_dir, exist_ok=True)

    # (1) PCA 散点图：按簇上色
    pca_path = os.path.join(results_dir, f"{method_name}_pca_clusters.png")
    save_pca_scatter(X, cluster_labels, pca_path, title=f"{method_name} - clusters")

    # (2) PCA 散点图：按真实数字上色
    pca_true_path = os.path.join(results_dir, f"{method_name}_pca_true_labels.png")
    save_pca_scatter(X, labels, pca_true_path, title=f"{method_name} - true digits")

    # (3) 每个簇的样本拼图
    save_cluster_examples(
        images,
        cluster_labels,
        method_name=method_name,
        out_dir=os.path.join(results_dir, "cluster_examples"),
        examples_per_cluster=10
    )


def main():
    # --------- 参数区（以后你只改这里）---------
    dataset_dir = "data/digits/noise01_shift0"
    digits = list(range(10))
    image_size = 32

    # 特征组合（可以随便改）
    # projection / intersections / zoning_4x4 / zoning_8x8 / global
    # FEATURES_TO_USE = ["projection", "intersections", "zoning_4x4", "global"]
    FEATURES_TO_USE = ["intersections"]
    USE_BBOX_NORM = False   # 或 True，增强平移鲁棒性

    # 聚类方法
    methods_to_run = ["maxmin"]   
    n_clusters = 10
    random_state = 0

    results_dir = "results"

    # --------- 1. 确保有数据集 ---------
    if not os.path.exists(dataset_dir) or not any(os.scandir(dataset_dir)):
        print("[main] 没有检测到数据集，自动生成一份 ...")
        generate_and_save_dataset(
            output_dir=dataset_dir,
            digits=digits,
            samples_per_digit=100,
            image_size=image_size,
            font_size=30,
            noise_level=0.15,
            shift_range=3,
            random_seed=random_state,
            overwrite=False
        )

    # --------- 2. 读取图片 ---------
    images, labels = load_dataset_from_disk(
        dataset_dir=dataset_dir,
        digits=digits,
        image_size=image_size
    )

    # --------- 3. 提取特征 ---------
    X = build_features(
        images,
        feature_names = FEATURES_TO_USE,
        use_bbox_norm=USE_BBOX_NORM,
        out_size=image_size
    )

    # --------- 4. 对每种方法跑一遍 ---------
    for method_name in methods_to_run:
        run_one_method(
            method_name=method_name,
            X=X,
            images=images,
            labels=labels,
            digits=digits,
            results_dir=results_dir,
            n_clusters=n_clusters,
            random_state=random_state
        )


if __name__ == "__main__":
    main()
