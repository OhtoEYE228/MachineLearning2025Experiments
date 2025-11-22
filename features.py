# features.py
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

def crop_and_resize_to_square(img_arr, out_size=32, threshold=0.3, margin=1):
    """
    对单张数字图像做预处理：
    1) 二值化 -> 找前景像素的外接矩形（bounding box）
    2) bbox 周围加一点 margin
    3) 裁剪出 ROI 后 pad 成近似正方形
    4) 缩放到 out_size × out_size
    这样基本消除平移带来的影响
    """
    h, w = img_arr.shape
    bin_img = img_arr > threshold
    ys, xs = np.where(bin_img)

    # 如果整张都是黑的，就直接返回原图（极端情况）
    if len(xs) == 0:
        return img_arr

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    y0 = max(0, y0 - margin)
    y1 = min(h - 1, y1 + margin)
    x0 = max(0, x0 - margin)
    x1 = min(w - 1, x1 + margin)

    roi = img_arr[y0:y1+1, x0:x1+1]

    # pad 成近似正方形
    rh, rw = roi.shape
    pad_top = pad_bottom = pad_left = pad_right = 0
    if rh > rw:
        diff = rh - rw
        pad_left = diff // 2
        pad_right = diff - pad_left
    elif rw > rh:
        diff = rw - rh
        pad_top = diff // 2
        pad_bottom = diff - pad_top

    roi_padded = np.pad(
        roi,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0.0
    )

    roi_img = Image.fromarray((roi_padded * 255).astype(np.uint8), mode="L")
    roi_img = roi_img.resize((out_size, out_size), resample=Image.BILINEAR)
    out = np.array(roi_img, dtype=np.float32) / 255.0
    return out


# --------- 各种特征 ----------

def extract_projection_features(img_arr):
    """
    投影直方图特征：
    - 对每一行求和 -> 水平投影 (H,)
    - 对每一列求和 -> 垂直投影 (W,)
    拼接后按总和归一化。
    """
    row_sum = img_arr.sum(axis=1)
    col_sum = img_arr.sum(axis=0)
    feat = np.concatenate([row_sum, col_sum])
    total = feat.sum() + 1e-8
    feat = feat / total
    return feat.astype(np.float32)

def extract_skeleton_features(img_arr, threshold=0.3):
    """
    Skeleton 特征：
    - 二值化 -> skeletonize 得到 1 像素宽骨架
    - 统计：
        endpoints: 邻居数=1 的骨架点数量
        branchpoints: 邻居数>=3 的骨架点数量
        length: 骨架总像素数
    返回一个 3 维向量，简单归一化。
    """
    bin_img = img_arr > threshold
    if not np.any(bin_img):
        return np.zeros(3, dtype=np.float32)

    skel = skeletonize(bin_img).astype(np.uint8)
    h, w = skel.shape

    endpoints = 0
    branchpoints = 0

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skel[y, x] == 0:
                continue
            # 3x3 邻域
            neighborhood = skel[y-1:y+2, x-1:x+2]
            neighbors = neighborhood.sum() - 1  # 去掉自己
            if neighbors == 1:
                endpoints += 1
            elif neighbors >= 3:
                branchpoints += 1

    length = int(skel.sum())

    if length == 0:
        return np.zeros(3, dtype=np.float32)

    # 简单缩放一下，避免数值太大
    feat = np.array([endpoints, branchpoints, length], dtype=np.float32)
    feat = feat / (length + 1e-6)
    return feat

def extract_zoning_features(img_arr, grid_size=(4, 4)):
    """
    Zoning 特征：把图像分成 grid_size 的网格，每个格子取平均灰度。
    例如 grid_size=(4,4) 时特征维度为 16。
    """
    h, w = img_arr.shape
    gh, gw = grid_size
    cell_h = h // gh
    cell_w = w // gw

    feats = []
    for r in range(gh):
        for c in range(gw):
            block = img_arr[
                r * cell_h:(r + 1) * cell_h,
                c * cell_w:(c + 1) * cell_w
            ]
            feats.append(block.mean())

    return np.array(feats, dtype=np.float32)

def extract_grad_orientation_hist(img_arr, num_bins=8):
    """
    简化版梯度方向直方图（全局版）：
    1) 用 np.gradient 得到每个像素的梯度 gx, gy
    2) 计算梯度幅值 mag 和方向 ori
    3) 方向范围 [0, pi)，分成 num_bins 个区间
    4) 用 mag 作为权重累加到直方图
    5) L1 归一化直方图
    """
    img = img_arr.astype(np.float32)

    # 计算梯度（gy, gx 的顺序注意）
    gy, gx = np.gradient(img)
    mag = np.hypot(gx, gy)          # 梯度幅值
    ori = np.arctan2(gy, gx)        # [-pi, pi]
    ori = np.mod(ori, np.pi)        # 映射到 [0, pi)，方向无符号

    mag_flat = mag.ravel()
    ori_flat = ori.ravel()

    # 去掉几乎没有梯度的像素（纯背景）
    mask = mag_flat > 1e-3
    if not np.any(mask):
        return np.zeros(num_bins, dtype=np.float32)

    mag_flat = mag_flat[mask]
    ori_flat = ori_flat[mask]

    # 按方向划分 bin
    bin_edges = np.linspace(0.0, np.pi, num_bins + 1, endpoint=True)
    hist = np.zeros(num_bins, dtype=np.float32)

    # 找每个像素属于哪个 bin
    inds = np.searchsorted(bin_edges, ori_flat, side="right") - 1
    inds = np.clip(inds, 0, num_bins - 1)

    # 用梯度幅值累加
    for idx, w in zip(inds, mag_flat):
        hist[idx] += w

    # L1 归一化
    s = hist.sum()
    if s > 0:
        hist /= s

    return hist



def extract_global_shape_features(img_arr, threshold=0.3):
    """
    一些全局粗特征：
    - 前景面积占比
    - 上/下、左/右质量分布
    - 垂直、水平“对称性”（1 表示完全对称，0 表示差异很大）
    """
    h, w = img_arr.shape
    bin_img = img_arr > threshold

    area = bin_img.sum() / (h * w)  # 前景面积占比

    rows = bin_img.sum(axis=1)
    cols = bin_img.sum(axis=0)
    total = bin_img.sum() + 1e-8

    top = rows[:h//2].sum()
    bottom = rows[h//2:].sum()
    left = cols[:w//2].sum()
    right = cols[w//2:].sum()

    top_ratio = top / total
    bottom_ratio = bottom / total
    left_ratio = left / total
    right_ratio = right / total

    # 对称性：比较图像与左右翻转/上下翻转的差异
    vert_sym = 1.0 - np.mean(np.abs(img_arr - img_arr[:, ::-1]))   # 左右
    horiz_sym = 1.0 - np.mean(np.abs(img_arr - img_arr[::-1, :])) # 上下

    feat = np.array([
        area,
        top_ratio,
        bottom_ratio,
        left_ratio,
        right_ratio,
        vert_sym,
        horiz_sym,
    ], dtype=np.float32)

    return feat


def extract_line_intersection_features(img_arr, num_lines=8, threshold=0.5):
    """
    交点特征：
    - 二值化 -> bin_img
    - 竖直 + 水平各 num_lines 条扫描线
    - 每条线数 0->1 的跳变次数
    - 共 2*num_lines 维，按最大值归一化。
    """
    bin_img = (img_arr > threshold).astype(np.uint8)
    h, w = bin_img.shape
    feats = []

    # 竖直扫描线
    for i in range(num_lines):
        x = int((i + 0.5) * w / num_lines)
        col = bin_img[:, x]
        count = 0
        for j in range(1, h):
            if col[j-1] == 0 and col[j] == 1:
                count += 1
        feats.append(count)

    # 水平扫描线
    for i in range(num_lines):
        y = int((i + 0.5) * h / num_lines)
        row = bin_img[y, :]
        count = 0
        for j in range(1, w):
            if row[j-1] == 0 and row[j] == 1:
                count += 1
        feats.append(count)

    feats = np.array(feats, dtype=np.float32)
    max_val = feats.max() if feats.max() > 0 else 1.0
    feats = feats / max_val
    return feats


# --------- 特征组合接口 ----------

def make_feature_fns():
    """
    把可用特征封装成一个字典，方便自由组合。
    可以按名字选用：
      projection / intersections / zoning_4x4 / zoning_8x8 / global
    """
    return {
        "projection": lambda img: extract_projection_features(img),
        "intersections": lambda img: extract_line_intersection_features(img, num_lines=8),
        "zoning_4x4": lambda img: extract_zoning_features(img, grid_size=(4, 4)),
        "zoning_8x8": lambda img: extract_zoning_features(img, grid_size=(8, 8)),
        "global": lambda img: extract_global_shape_features(img),
        "grad_hist":    lambda img: extract_grad_orientation_hist(img, num_bins=8),
        "skeleton":     lambda img: extract_skeleton_features(img),
    }


def build_features(
    images,
    feature_names,
    use_bbox_norm=True,
    out_size=None
):
    """
    images: (N, H, W)
    feature_names: 例如 ["projection", "intersections", "zoning_4x4", "global"]
    use_bbox_norm: 是否先做 bbox 裁剪+缩放，增强平移鲁棒性
    """
    if out_size is None:
        out_size = images.shape[1]

    feature_fns = make_feature_fns()
    feat_list = []

    for img in images:
        # 先做平移归一化（可关掉）
        if use_bbox_norm:
            proc = crop_and_resize_to_square(img, out_size=out_size)
        else:
            proc = img

        parts = []
        for name in feature_names:
            fn = feature_fns[name]
            parts.append(fn(proc))
        feat = np.concatenate(parts)
        feat_list.append(feat)

    X = np.stack(feat_list)
    print(f"[features] X shape: {X.shape}, 使用特征: {feature_names}")
    return X
