# demo_feature_distribution.py
"""
演示：在使用某种特征提取方式后，各个数字在这些特征上的表现。

功能：
1. 从指定数据集读取所有图片和标签；
2. 选择一种特征提取方式（例如 "projection"、"zoning_4x4"、"grad_hist" 等）；
3. 对所有样本提取该特征；
4. 画出：
   (1) 特征维度上的按数字分组的平均曲线；
   (2) PCA 2D 散点图（颜色区分数字）。

使用方式：
- 修改本文件开头的 CONFIG 区域，然后直接运行该脚本即可。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from digit_dataset import load_dataset_from_disk
from features import make_feature_fns, crop_and_resize_to_square


# =========================
# 配置区：在这里修改参数
# =========================

DATASET_DIR = "data/digits/noise0_shift2"  # 选择要演示的数据集
DIGITS = list(range(10))
IMAGE_SIZE = 32

FEATURE_NAME = "intersections"   # 想要演示的特征名称
USE_BBOX_NORM = False         # 是否先做 bbox 裁剪 + resize 到中心

OUTPUT_DIR = "results/feature_demo"


# =========================
# 1. 特征计算
# =========================

def compute_feature_for_all_samples(
    dataset_dir,
    feature_name,
    digits,
    image_size=32,
    use_bbox_norm=True
):
    """
    读取 dataset_dir 下所有图片，提取指定 feature_name 的特征。

    返回：
      X: (N, D) 特征矩阵
      labels: (N,) 数字标签
    """
    if not os.path.exists(dataset_dir) or not any(os.scandir(dataset_dir)):
        raise FileNotFoundError(f"数据集目录 {dataset_dir} 不存在或为空。")

    images, labels = load_dataset_from_disk(
        dataset_dir=dataset_dir,
        digits=digits,
        image_size=image_size
    )
    print(f"[data] images: {images.shape}, labels: {labels.shape}")

    feature_fns = make_feature_fns()
    if feature_name not in feature_fns:
        raise ValueError(f"未知特征名称: {feature_name}，可用: {list(feature_fns.keys())}")

    feat_fn = feature_fns[feature_name]

    feat_list = []
    for idx, img in enumerate(images):
        if use_bbox_norm:
            img_proc = crop_and_resize_to_square(img, out_size=image_size)
        else:
            img_proc = img

        f = feat_fn(img_proc)
        feat_list.append(f)

    X = np.stack(feat_list).astype(np.float32)
    print(f"[feature] 特征名: {feature_name}, X shape: {X.shape}")
    return X, labels


# =========================
# 2. 可视化：特征维度上的平均曲线
# =========================

def plot_mean_feature_per_digit(
    X, labels, digits, feature_name, out_dir, max_digits_in_legend=10
):
    """
    对每个数字 d：
      - 在所有属于 d 的样本上，计算特征向量的均值 => mean_f_d (D,)
    再在一张图上画出所有 mean_f_d 曲线。

    适合维度不太大的特征（几十~几百维左右）。
    """
    os.makedirs(out_dir, exist_ok=True)

    dim = X.shape[1]
    xs = np.arange(dim)

    plt.figure(figsize=(8, 5))
    for d in digits:
        idx = np.where(labels == d)[0]
        if len(idx) == 0:
            continue
        mean_feat = X[idx].mean(axis=0)
        plt.plot(xs, mean_feat, label=str(d))  # 不用指定颜色，由 Matplotlib 自己配

    plt.xlabel("Feature dimension index")
    plt.ylabel("Mean value")
    plt.title(f"Mean {feature_name} feature per digit")
    if len(digits) <= max_digits_in_legend:
        plt.legend(title="Digit", ncol=5)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{feature_name}_mean_per_digit.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[vis] 保存均值曲线图到 {out_path}")


# =========================
# 3. 可视化：PCA 2D 散点图
# =========================

def plot_pca_scatter_by_digit(
    X, labels, digits, feature_name, out_dir
):
    """
    使用 PCA 将特征降到 2 维，可视化每个样本在该特征空间中的分布。
    用颜色区分数字。
    """
    os.makedirs(out_dir, exist_ok=True)

    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(
        X2[:, 0],
        X2[:, 1],
        c=labels,
        s=5,
        alpha=0.7
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA scatter of {feature_name} feature")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Digit label")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{feature_name}_pca_scatter.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[vis] 保存 PCA 散点图到 {out_path}")


# =========================
# 4. 主流程
# =========================

def run_demo():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) 计算特征
    X, labels = compute_feature_for_all_samples(
        dataset_dir=DATASET_DIR,
        feature_name=FEATURE_NAME,
        digits=DIGITS,
        image_size=IMAGE_SIZE,
        use_bbox_norm=USE_BBOX_NORM
    )

    # 2) 画“按数字的平均特征曲线”
    plot_mean_feature_per_digit(
        X,
        labels,
        digits=DIGITS,
        feature_name=FEATURE_NAME,
        out_dir=OUTPUT_DIR
    )

    # 3) 画“PCA 2D 散点图”
    plot_pca_scatter_by_digit(
        X,
        labels,
        digits=DIGITS,
        feature_name=FEATURE_NAME,
        out_dir=OUTPUT_DIR
    )

    print("[demo] 完成特征可视化。")


if __name__ == "__main__":
    run_demo()
