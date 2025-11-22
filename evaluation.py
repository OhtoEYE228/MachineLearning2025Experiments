# evaluation.py
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

def map_clusters_to_labels(y_true, cluster_labels, n_classes):
    """
    把“簇编号”映射为“数字类别”，方便计算准确率/召回率。

    做法：
    - 先算 confusion_matrix(true_digit, cluster_id)
    - 对每个 cluster，找该簇中出现最多的真实数字 => 作为该簇的预测标签
    - 得到 y_pred_digit（与 y_true 同维度）
    """
    cm = confusion_matrix(y_true, cluster_labels, labels=range(n_classes))
    cluster_to_label = {}

    for cluster in range(cm.shape[1]):
        col = cm[:, cluster]              # 每个真实数字在该簇的数量
        label = int(np.argmax(col))       # 多数表决
        cluster_to_label[cluster] = label

    y_pred = np.array([cluster_to_label[c] for c in cluster_labels], dtype=int)
    return y_pred, cluster_to_label, cm


def evaluate_clustering_full(y_true, cluster_labels, digits):
    """
    计算：
    - 调整兰德指数 ARI（不需要映射）
    - 归一化互信息 NMI（不需要映射）
    - 映射后的分类准确率
    - 每类的 precision / recall / F1 / support(真实数) / pred_count(预测数)
    - 混淆矩阵（真实数字 vs cluster_id）
    """
    n_classes = len(digits)

    # 不依赖标签编号的无监督指标
    ari = adjusted_rand_score(y_true, cluster_labels)
    nmi = normalized_mutual_info_score(y_true, cluster_labels)

    # 簇 -> 数字 编号映射
    y_pred_digit, cluster_to_label, cm = map_clusters_to_labels(
        y_true, cluster_labels, n_classes
    )

    # 映射后的整体分类指标
    acc = accuracy_score(y_true, y_pred_digit)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred_digit,
        labels=digits,
        zero_division=0
    )

    # 每个数字被预测为该类的总样本数
    pred_counts = {}
    for d in digits:
        pred_counts[d] = int((y_pred_digit == d).sum())

    metrics_per_class = {}
    for i, d in enumerate(digits):
        metrics_per_class[d] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": support[i],           # 真实是 d 的个数
            "pred_count": pred_counts[d],    # 预测为 d 的个数
        }

    return {
        "ari": ari,
        "nmi": nmi,
        "accuracy": acc,
        "per_class": metrics_per_class,
        "cluster_to_label": cluster_to_label,
        "confusion_matrix": cm,
        "y_pred_digit": y_pred_digit,
    }
