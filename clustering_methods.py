# clustering_methods.py
# 说明：
#  - 不再使用 sklearn.cluster.KMeans / AgglomerativeClustering
#  - 全部聚类算法统一用 numpy 手写：
#       1) KMeans 聚类（动态聚类 / K-均值）
#       2) 层次聚类（系统聚类，自底向上合并，支持 single/complete/average/ward）
#       3) 最大最小距离聚类（max-min 距离算法，和 PPT 中一致）
#
#  接口保持不变：
#    cluster_kmeans(X, n_clusters=10, random_state=0)
#    cluster_agglomerative(X, n_clusters=10, linkage="ward")
#    cluster_maxmin(X, n_clusters=10, random_state=0)
#
#  返回值仍是：
#    labels: (N,) int，0~(k-1) 的簇编号
#    model:  一个简单 dict，保存中心或参数信息

import numpy as np


# =========================
# 公共：两组样本的欧氏距离平方
# =========================

def _pairwise_distances_squared(X, Y):
    """
    计算两组样本的欧氏距离平方:
        X: (N, D)
        Y: (M, D)
    返回:
        D2: (N, M)，第 i 行 j 列是 ||X_i - Y_j||^2

    利用展开式：
        ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    X2 = np.sum(X ** 2, axis=1, keepdims=True)        # (N, 1)
    Y2 = np.sum(Y ** 2, axis=1, keepdims=True).T      # (1, M)
    XY = X @ Y.T                                      # (N, M)

    D2 = X2 + Y2 - 2 * XY
    D2[D2 < 0] = 0.0           # 数值误差保护
    return D2


# =========================
# 1. KMeans 聚类（动态聚类 / K-均值）
# =========================

def cluster_kmeans(X, n_clusters=10, random_state=0, max_iter=100, tol=1e-4):
    """
    手写版 KMeans 聚类（动态聚类）：

    步骤：
      1) 随机选取 n_clusters 个样本作为初始中心；
      2) 迭代：
         - E 步：把每个样本分配给最近的中心；
         - M 步：对每个簇，取该簇样本均值作为新中心；
         - 若中心整体移动量 < tol，则停止。

    参数：
      X: (N, D) 样本特征
      n_clusters: 簇个数
      random_state: 随机种子（控制初始中心）
      max_iter: 最大迭代次数
      tol: 收敛阈值（中心移动的范数）

    返回:
      labels: (N,) int，簇标签 0~(k-1)
      model: dict，包含中心等信息
    """
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape

    if n_clusters <= 0 or n_clusters > n_samples:
        raise ValueError("n_clusters 必须在 [1, 样本数] 范围内")

    rng = np.random.RandomState(random_state)

    # 1. 随机选择初始中心（从样本中无放回采样）
    init_idx = rng.choice(n_samples, size=n_clusters, replace=False)
    centers = X[init_idx].copy()      # (k, D)

    labels = np.zeros(n_samples, dtype=int)

    for it in range(max_iter):
        # E 步：把每个样本分配到最近的中心
        d2 = _pairwise_distances_squared(X, centers)      # (N, k)
        new_labels = np.argmin(d2, axis=1)                # (N,)

        # M 步：根据新的簇划分更新中心
        centers_old = centers.copy()
        for k in range(n_clusters):
            mask = (new_labels == k)
            if np.any(mask):
                centers[k] = X[mask].mean(axis=0)
            else:
                # 出现空簇：把这个中心重新放到“当前最不易归类”的样本位置
                # 这里简单选取：对所有样本到最近中心距离中，最远的那个样本
                far_idx = int(np.argmax(np.min(d2, axis=1)))
                centers[k] = X[far_idx]

        # 计算中心的整体移动量
        shift = np.linalg.norm(centers - centers_old)
        labels = new_labels

        if shift < tol:
            break

    inertia = float(np.sum((X - centers[labels]) ** 2))

    model = {
        "centers": centers,          # (k, D)
        "n_iter": it + 1,
        "inertia": inertia,
        "algorithm": "kmeans",
        "n_clusters": n_clusters,
        "random_state": random_state,
    }

    return labels, model


# =========================
# 2. 层次聚类（系统聚类，自底向上）
# =========================

def cluster_agglomerative(X, n_clusters=10, linkage="ward"):
    """
    手写版系统聚类（层次聚类，自底向上合并）：

    初始时每个样本单独成簇：
      - 维护一个簇间距离矩阵 D (N x N)
      - 每次找到距离最小的两个簇 i, j 合并
      - 按 linkage 规则更新新簇与其它簇之间的距离
      - 重复直到剩下 n_clusters 个簇

    linkage 支持:
      - 'single'   : 最短距离 (single-link)
      - 'complete' : 最长距离 (complete-link)
      - 'average'  : 类平均距离 (average-link)
      - 'ward'     : 离差平方和最小 (Ward 法，基于 Lance-Williams 公式)

    注意：这是一个 O(N^3) 的朴素实现，适合样本数不是特别大的情况。
    """
    X = np.asarray(X, dtype=float)
    N = X.shape[0]

    if n_clusters <= 0 or n_clusters > N:
        raise ValueError("n_clusters 必须在 [1, 样本数] 范围内")

    # 初始距离矩阵：用欧氏距离平方
    D = _pairwise_distances_squared(X, X)   # (N, N)
    np.fill_diagonal(D, np.inf)

    # active[i] = True 表示簇 i 目前仍存在
    active = np.ones(N, dtype=bool)
    # cluster_size[i] = 簇 i 当前包含的样本数
    cluster_size = np.ones(N, dtype=int)

    # labels[s] = 当前样本 s 所在簇的“代表索引”
    labels = np.arange(N, dtype=int)

    num_active = N
    linkage = linkage.lower()
    if linkage not in ("single", "complete", "average", "ward"):
        raise ValueError("不支持的 linkage 类型: " + linkage)

    while num_active > n_clusters:
        # 在所有 active 簇中寻找距离最小的一对 (i, j)
        idx = np.argmin(D)
        i = idx // N
        j = idx % N

        # 理论上 D 中 inactive 行列都设为 inf，所以这里 i, j 应该都是 active
        if not (active[i] and active[j]) or i == j:
            D[i, j] = D[j, i] = np.inf
            continue

        # 约定保持 i < j，合并 j 到 i
        if j < i:
            i, j = j, i

        n_i = cluster_size[i]
        n_j = cluster_size[j]

        # 更新新簇 i 与其他簇 k 的距离
        for k in range(N):
            if not active[k] or k == i or k == j:
                continue

            if linkage == "single":
                # 最短距离
                d_new = min(D[i, k], D[j, k])

            elif linkage == "complete":
                # 最长距离
                d_new = max(D[i, k], D[j, k])

            elif linkage == "average":
                # 类平均距离
                d_new = (n_i * D[i, k] + n_j * D[j, k]) / (n_i + n_j)

            elif linkage == "ward":
                # Ward 法：基于 Lance-Williams 公式
                # d_new(i∪j, k) = α_i d(i,k) + α_j d(j,k) + β d(i,j)
                n_k = cluster_size[k]
                denom = (n_i + n_j + n_k)
                alpha_i = (n_i + n_k) / denom
                alpha_j = (n_j + n_k) / denom
                beta = - n_k / denom
                d_new = alpha_i * D[i, k] + alpha_j * D[j, k] + beta * D[i, j]

            D[i, k] = D[k, i] = d_new

        # 合并完成：簇 j 被吸收进 i
        active[j] = False
        cluster_size[i] = n_i + n_j
        cluster_size[j] = 0

        # 簇 j 的行列不再参与后续最小值搜索
        D[j, :] = np.inf
        D[:, j] = np.inf

        # 更新样本标签：所有原来属于簇 j 的样本并入 i
        labels[labels == j] = i

        num_active -= 1

    # 此时 labels 中的簇编号是原始索引（例如 0, 7, 23 ...），需要压缩到 0 ~ (n_clusters-1)
    unique_clusters = np.unique(labels)
    mapping = {old: idx for idx, old in enumerate(unique_clusters)}
    final_labels = np.array([mapping[c] for c in labels], dtype=int)

    model = {
        "linkage": linkage,
        "n_clusters": n_clusters,
        "cluster_indices_original": unique_clusters,   # 原始簇索引
        "algorithm": "agglomerative",
    }

    return final_labels, model


# =========================
# 3. 最大最小距离算法（max-min distance clustering）
# =========================

def cluster_maxmin(X, n_clusters=10, random_state=0):
    """
    最大最小距离聚类算法（课上 PPT 里的 “max-min 距离算法”）：

    1) 随机选择一个样本作为第一个中心 c1；
    2) 选择距离 c1 最远的样本作为第二个中心 c2；
    3) 对于每一个尚未成为中心的样本 x，
       计算其到当前所有中心的距离，取最近中心距离 d(x)，
       再选择 d(x) 最大的样本作为新的中心；
    4) 重复步骤 3 直到选出 n_clusters 个中心；
    5) 最后，将所有样本归到最近的中心，得到聚类标签。

    返回:
        labels: (N,) 聚类标签
        model:  简单 dict，保存选出的中心
    """
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]

    if n_clusters <= 0:
        raise ValueError("n_clusters 必须是正整数")
    if n_clusters > n_samples:
        raise ValueError("n_clusters 不能大于样本数")

    rng = np.random.RandomState(random_state)

    # 1. 第一个中心：随机选一个样本
    first_idx = rng.randint(0, n_samples)
    centers_idx = [first_idx]

    # 2. 第二个中心：距离第一个中心最远的样本
    if n_clusters >= 2:
        c0 = X[first_idx:first_idx + 1, :]       # (1, D)
        d2 = _pairwise_distances_squared(X, c0).ravel()   # (N,)
        second_idx = int(np.argmax(d2))
        centers_idx.append(second_idx)

    # 3. 继续选剩下的中心
    while len(centers_idx) < n_clusters:
        current_centers = X[centers_idx, :]                  # (k, D)
        d2_all = _pairwise_distances_squared(X, current_centers)  # (N, k)
        # 对每个样本，取其到当前任一中心的最近距离
        min_d2 = np.min(d2_all, axis=1)                      # (N,)

        # 已经是中心的样本不再参与下一轮选取
        min_d2[centers_idx] = -1.0

        new_center_idx = int(np.argmax(min_d2))
        if min_d2[new_center_idx] <= 0:
            # 所有非中心点距离都很小，说明数据点非常接近，可以提前停止
            break

        centers_idx.append(new_center_idx)

    centers_idx = np.array(centers_idx, dtype=int)
    centers = X[centers_idx, :]     # (k, D)

    # 4. 把所有样本归到最近的中心
    d2_final = _pairwise_distances_squared(X, centers)     # (N, k)
    labels = np.argmin(d2_final, axis=1)

    model = {
        "centers": centers,
        "centers_idx": centers_idx,
        "n_clusters": len(centers_idx),
        "algorithm": "maxmin",
    }

    return labels, model


# =========================
# 4. 方法映射表（供主程序统一调用）
# =========================

CLUSTERING_METHODS = {
    "kmeans":       cluster_kmeans,
    "agglomerative": cluster_agglomerative,
    "maxmin":       cluster_maxmin,
}
