from PIL import Image


def _is_colored_content(pixel):
    """
    判断像素是否属于有意义的验证码内容（非纯白、非纯灰）。
    """
    r, g, b = pixel[:3]
    # 稍微放宽白色的定义 (从 238 降到 250，意味着只有极白才排除)
    if r > 250 and g > 250 and b > 250:
        return False
    # 显著降低灰度判定阈值 (从 18 降到 8)，保留更多细微的彩色
    if max(r, g, b) - min(r, g, b) < 8:
        return False
    return True


def find_challenge_image_rect(modal):
    """Return the inner captcha image rect inside a cropped captcha modal."""
    w, h = modal.size
    pixels = modal.load()
    row_hits = []

    for y in range(h):
        hits = 0
        for x in range(w):
            if _is_colored_content(pixels[x, y]):
                hits += 1
        row_hits.append(hits / max(1, w))

    bands = []
    start = None
    # 大幅降低阈值 (从 0.35 降到 0.02)，只要有 2% 的彩色内容就视为有效行
    threshold = 0.02
    
    for y, ratio in enumerate(row_hits):
        if ratio > threshold:
            if start is None:
                start = y
        elif start is not None:
            # 这里的 20 像素是防止把细小的文字行当成主图
            if y - start >= 30:
                bands.append((start, y - 1))
            start = None
            
    if start is not None and h - start >= 30:
        bands.append((start, h - 1))

    if not bands:
        # 如果没找到明显的带，尝试用极低阈值再找一次
        return (0, 0, w, h)

    # 验证码图片通常是模态框中最大的一个彩色块
    top, bottom = max(bands, key=lambda band: band[1] - band[0])
    
    # 再次检查宽度方向的边界
    xs = []
    for y in range(top, bottom + 1):
        for x in range(w):
            if _is_colored_content(pixels[x, y]):
                xs.append(x)

    if not xs:
        return (0, top, w, bottom)

    # 增加更多的外扩边距 (Padding)，确保不切到字
    left = max(0, min(xs) - 5)
    right = min(w, max(xs) + 5)
    top = max(0, top - 5)
    bottom = min(h, bottom + 5)
    
    return (left, top, right, bottom)


def crop_challenge_image(modal):
    rect = find_challenge_image_rect(modal)
    if rect is None:
        return None, None
    return modal.crop(rect), rect
