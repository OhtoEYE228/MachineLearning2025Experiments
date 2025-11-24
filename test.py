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
from digit_clustering import run_one_method




def main():
    dataset_dir = "data/digits/noise0_shift0"
    digits = list(range(10))
    image_size = 32

    # 特征组合
    # projection / intersections / zoning_4x4 / zoning_8x8 / global
    # FEATURES_TO_USE = ["projection", "intersections", "zoning_4x4", "global"]
    FEATURES_TO_USE = ["projection"]
    USE_BBOX_NORM = False   # 预处理

    # 聚类方法
    methods_to_run = ["kmeans"] 
    n_clusters = 10
    random_state = 0

    results_dir = "results"

    # --------- 1. 确保有数据集 ---------
    if not os.path.exists(dataset_dir) or not any(os.scandir(dataset_dir)):
        print("[main] 没有检测到数据集")
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
        use_bbox_norm = USE_BBOX_NORM,
        out_size = image_size
    );

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
