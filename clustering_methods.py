# clustering_methods.py

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering


# =========================
# 1. KMeans 聚类（动态聚类 / K-均值）
# =========================

def cluster_kmeans(X, n_clusters=10, random_state=0):
    """
    标准 KMeans 聚类。
    返回:
        labels: (N,) 聚类标签
        model:  sklearn 的 KMeans 对象
    """
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    labels = model.fit_predict(X)
    return labels, model


# =========================
# 2. 层次聚类（系统聚类的一种）
# =========================

def cluster_agglomerative(X, n_clusters=10, linkage="ward"):
    """
    层次聚类（自底向上合并），默认使用 Ward 方法。
    linkage 可以改成 'single' / 'complete' / 'average' / 'ward' 等。
    """
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage
    )
    labels = model.fit_predict(X)
    return labels, model


# =========================
# 3. 最大最小距离算法（max-min distance clustering）
# =========================

def _pairwise_distances_squared(X, Y):
    """
    计算两组样本的欧氏距离平方:
        X: (N, D)
        Y: (M, D)
    返回:
        D2: (N, M)，第 i 行 j 列是 ||X_i - Y_j||^2
    """
    # 利用 (x-y)^2 = x^2 + y^2 - 2xy^T 的展开提高效率
    X2 = np.sum(X**2, axis=1, keepdims=True)      # (N, 1)
    Y2 = np.sum(Y**2, axis=1, keepdims=True).T    # (1, M)
    XY = X @ Y.T                                   # (N, M)
    D2 = X2 + Y2 - 2 * XY
    # 数值上可能会有极小负数，截断到 0
    D2[D2 < 0] = 0.0
    return D2


def cluster_maxmin(X, n_clusters=10, random_state=0):
    """
    最大最小距离聚类算法（课上 PPT 里的“max-min 距离算法”版本）：

    1) 随机选择一个样本作为第一个中心 c1；
    2) 选择距离 c1 最远的样本作为第二个中心 c2；
    3) 对于每一个尚未成为中心的样本 x，
       计算其到当前所有中心的距离，取最近中心距离 d(x)，
       再选择 d(x) 最大的样本作为新的中心；
    4) 重复步骤 3 直到选出 n_clusters 个中心；
    5) 最后，将所有样本归到最近的中心，得到聚类标签。

    返回:
        labels: (N,) 聚类标签
        model:  一个简单的 dict，只保存选出的中心数组 {"centers": centers}
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples = X.shape[0]

    if n_clusters <= 0:
        raise ValueError("n_clusters 必须是正整数")
    if n_clusters > n_samples:
        raise ValueError("n_clusters 不能大于样本数")

    rng = np.random.RandomState(random_state)

    # 1. 选第一个中心：随机选一个样本索引
    first_idx = rng.randint(0, n_samples)
    centers_idx = [first_idx]

    # 2. 选第二个中心：距离第一个中心最远的点
    if n_clusters >= 2:
        c0 = X[first_idx:first_idx+1, :]  # shape (1, D)
        d2 = _pairwise_distances_squared(X, c0).ravel()  # (N,)
        second_idx = int(np.argmax(d2))
        centers_idx.append(second_idx)

    # 3. 继续选剩下的中心
    while len(centers_idx) < n_clusters:
        current_centers = X[centers_idx, :]               # (k, D)
        d2_all = _pairwise_distances_squared(X, current_centers)  # (N, k)
        # 对每个样本，取其到当前任一中心的最近距离
        min_d2 = np.min(d2_all, axis=1)                   # (N,)

        # 对于已经是中心的样本，把距离设为 -1，避免再选
        min_d2[centers_idx] = -1.0

        # 找到 min_d2 最大的那个样本，作为新中心
        new_center_idx = int(np.argmax(min_d2))
        # 如果所有非中心点 min_d2 也都 <= 0，说明全部重合或非常近，可以提前停止
        if min_d2[new_center_idx] <= 0:
            break

        centers_idx.append(new_center_idx)

    centers_idx = np.array(centers_idx, dtype=int)
    centers = X[centers_idx, :]   # (k, D)

    # 4. 把所有样本归到最近中心
    d2_final = _pairwise_distances_squared(X, centers)    # (N, k)
    labels = np.argmin(d2_final, axis=1)

    # 简单的“模型对象”：只保存中心
    model = {
        "centers": centers,
        "centers_idx": centers_idx,
        "n_clusters": len(centers_idx),
        "algorithm": "maxmin"
    }

    return labels, model


# =========================
# 4. 方法映射表
# =========================

CLUSTERING_METHODS = {
    "kmeans":      cluster_kmeans,
    "agglomerative": cluster_agglomerative,
    "maxmin":      cluster_maxmin,
}
