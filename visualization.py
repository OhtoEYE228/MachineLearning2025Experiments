# visualization.py
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def save_pca_scatter(X, labels, out_path, title="PCA scatter"):
    """
    用 PCA 把特征降到 2D，按 labels 上色。
    labels 可以是真实标签，也可以是簇编号。
    """
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(5, 5))
    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        s=5,
        alpha=0.8
    )
    plt.title(title)
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_cluster_examples(
    images,
    cluster_labels,
    method_name,
    out_dir="results/cluster_examples",
    examples_per_cluster=10
):
    """
    对每个簇，随机取若干张图片，拼在一张大图里保存。
    方便肉眼看“这个簇长得像哪个数字”。
    """
    os.makedirs(out_dir, exist_ok=True)
    n_clusters = int(cluster_labels.max()) + 1
    img_h, img_w = images.shape[1], images.shape[2]

    for c in range(n_clusters):
        idx = np.where(cluster_labels == c)[0]
        if len(idx) == 0:
            continue

        n_show = min(examples_per_cluster, len(idx))
        chosen = np.random.choice(idx, size=n_show, replace=False)

        cols = min(5, n_show)           # 一行最多 5 张
        rows = int(np.ceil(n_show / cols))

        grid = Image.new("L", (cols * img_w, rows * img_h), color=0)

        for k, sample_idx in enumerate(chosen):
            row = k // cols
            col = k % cols
            tile_arr = (images[sample_idx] * 255).astype(np.uint8)
            tile_img = Image.fromarray(tile_arr, mode="L")
            grid.paste(tile_img, (col * img_w, row * img_h))

        filename = os.path.join(
            out_dir,
            f"{method_name}_cluster_{c}.png"
        )
        grid.save(filename)
