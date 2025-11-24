# digit_dataset.py
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_PATH = r"C:\Windows\Fonts\timesbd.ttf"   

def get_font(font_size=28, font_path=FONT_PATH):
    """
    优先用 Times New Roman，找不到就退回默认字体。
    """
    try:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, font_size)
        else:
            print(f"[digit_dataset] 找不到字体文件 {font_path}，改用默认字体。")
            return ImageFont.load_default()
    except OSError:
        print(f"[digit_dataset] 打开 {font_path} 失败，改用默认字体。")
        return ImageFont.load_default()

def get_text_size(draw, text, font):
    """
    兼容 Pillow 新旧版本的文字尺寸计算：
    优先用 textbbox，其次 textsize，最后 font.getsize。
    """
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h

    if hasattr(draw, "textsize"):
        return draw.textsize(text, font=font)

    if hasattr(font, "getsize"):
        return font.getsize(text)

    return len(text) * 10, 20

def get_center_xy(draw, text, font, image_size):
    """
    使用 textbbox 精确计算让文字“视觉上”居中的坐标。

    设 textbbox((0,0)) 返回 (x0,y0,x1,y1)，
    那么如果在 (X,Y) 处画字，实际 bbox 是 (X+x0, Y+y0, X+x1, Y+y1)。

    我们要让这个 bbox 的中心 = 图像中心 (image_size/2, image_size/2)。
    推导得：
        X = (image_size - (x1 - x0)) / 2 - x0
        Y = (image_size - (y1 - y0)) / 2 - y0
    """
    if hasattr(draw, "textbbox"):
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
        w = x1 - x0
        h = y1 - y0
    else:
        # 老版本兼容
        w, h = draw.textsize(text, font=font)
        x0, y0 = 0, 0

    x = (image_size - w) / 2 - x0
    y = (image_size - h) / 2 - y0
    return int(x), int(y)


# =========================
# 单张数字图像生成
# =========================

def generate_digit_image(
    digit,
    image_size=32,
    font_size=28,
    noise_level=0.1,
    shift_range=3
):
    img = Image.new("L", (image_size, image_size), color=0)
    draw = ImageDraw.Draw(img)
    font = get_font(font_size)

    text = str(digit)

    # ✅ 精确居中坐标（不带随机位移）
    cx, cy = get_center_xy(draw, text, font, image_size)

    # ✅ 再叠加一点随机平移（如果你想完全居中，就把 shift_range 设为 0）
    dx = np.random.randint(-shift_range, shift_range + 1)
    dy = np.random.randint(-shift_range, shift_range + 1)
    x = cx + dx
    y = cy + dy

    # 画白色数字
    draw.text((x, y), text, fill=255, font=font)

    arr = np.array(img, dtype=np.float32) / 255.0

    # 高斯噪声
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)

    return arr

# =========================
# 生成并保存到 PNG
# =========================

def generate_and_save_dataset(
    output_dir="data/digits",
    digits=range(10),
    samples_per_digit=100,
    image_size=32,
    font_size=28,
    noise_level=0.15,
    shift_range=3,
    random_seed=0,
    overwrite=False
):
    """
    把数字图片生成到磁盘：
    data/digits/0/0_0000.png
    data/digits/0/0_0001.png
    ...
    data/digits/9/9_0099.png
    """
    np.random.seed(random_seed)

    # 如果已经存在且不想覆盖，就直接跳过
    if os.path.exists(output_dir) and not overwrite:
        # 简单判断一下有没有 png 文件
        has_png = False
        for root, _, files in os.walk(output_dir):
            if any(f.lower().endswith(".png") for f in files):
                has_png = True
                break
        if has_png:
            print(f"[digit_dataset] 检测到已有数据集目录 {output_dir}，不重新生成（overwrite=False）")
            return

    print(f"[digit_dataset] 开始生成数据集到 {output_dir} ...")
    for d in digits:
        digit_dir = os.path.join(output_dir, str(d))
        os.makedirs(digit_dir, exist_ok=True)

        for i in range(samples_per_digit):
            arr = generate_digit_image(
                d,
                image_size=image_size,
                font_size=font_size,
                noise_level=noise_level,
                shift_range=shift_range
            )
            # 保存为 PNG（0~255）
            img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
            filename = f"{d}_{i:04d}.png"
            img.save(os.path.join(digit_dir, filename))

    print(f"[digit_dataset] 数据集生成完成。")


# =========================
# 从磁盘读取 PNG，转成数组
# =========================

def load_dataset_from_disk(
    dataset_dir="data/digits",
    digits=range(10),
    image_size=32
):
    """
    从 data/digits 读取所有 png 图片：
    - images: (N, H, W), float32, [0,1]
    - labels: (N,), int，数字标签（0~9）
    """
    images = []
    labels = []

    for d in digits:
        digit_dir = os.path.join(dataset_dir, str(d))
        if not os.path.isdir(digit_dir):
            print(f"[digit_dataset] 警告: 目录 {digit_dir} 不存在，跳过该数字 {d}")
            continue

        file_list = sorted(
            f for f in os.listdir(digit_dir)
            if f.lower().endswith(".png")
        )
        for fname in file_list:
            path = os.path.join(digit_dir, fname)
            img = Image.open(path).convert("L")
            # 确保尺寸一致
            if img.size != (image_size, image_size):
                img = img.resize((image_size, image_size))

            arr = np.array(img, dtype=np.float32) / 255.0
            images.append(arr)
            labels.append(d)

    images = np.stack(images)
    labels = np.array(labels, dtype=np.int64)

    print(f"[digit_dataset] 载入完成: images={images.shape}, labels={labels.shape}")
    return images, labels



if __name__ == "__main__":
    # 你可以在这里调整参数
    generate_and_save_dataset(
        output_dir="data/digits/noise005_shift1",
        digits=range(10),
        samples_per_digit=10,
        image_size=32,
        font_size=28,
        noise_level=0.05,  # 高斯噪声方差, 均值0
        shift_range=1,  # 平移量
        random_seed=0,
        overwrite=True  # 已有就不覆盖
    )
    print("[digit_dataset] 运行结束。")
