#!/usr/bin/env python3
"""
Render text output into terminal-window style PNG screenshots.

Dùng để tạo "screenshot" cho deliverable của LAB 16: đọc nội dung file .txt
rồi vẽ ra ảnh PNG trông giống ảnh chụp terminal (có thanh tiêu đề, nút cửa sổ,
nền tối, font monospace).

Usage:
    python3 make_screenshots.py <file.txt> <out.png> [--title "..."]
"""

import sys
from PIL import Image, ImageDraw, ImageFont

# --- terminal theme -------------------------------------------------------
BG = (24, 24, 33)          # nền terminal
CHROME = (38, 38, 51)      # thanh tiêu đề
FG = (222, 222, 230)       # chữ thường
DIM = (128, 134, 148)      # chữ mờ / comment
ACCENT = (137, 180, 250)   # đường kẻ banner, tiêu đề
GREEN = (166, 227, 161)    # prompt, số tốt
YELLOW = (249, 226, 175)   # cảnh báo
BORDER = (58, 58, 74)

FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

PAD = 22          # padding trong terminal
LINE_SP = 1.42    # khoảng cách dòng
TITLEBAR = 34     # chiều cao thanh tiêu đề
DOT_R = 6


def classify(line):
    """Chọn màu cho từng dòng để giống output thật."""
    s = line.rstrip()
    t = s.strip()
    if not t:
        return FG
    if set(t) <= set("=-+| ") and len(t) > 3:
        return ACCENT                       # banner / đường kẻ bảng
    if t.startswith(("azureuser@", "$ ", "#")):
        return GREEN                        # prompt
    if t.startswith(("#", "//")):
        return DIM                          # comment
    if "WARNING" in t or "CẢNH BÁO" in t or "Error" in t or "error" in t:
        return YELLOW
    return FG


def render(src, dst, title=None, size=15, max_cols=None, wrap=None):
    with open(src, encoding="utf-8", errors="replace") as f:
        raw = f.read().rstrip("\n").split("\n")

    # cắt bớt tab / khoảng trắng thừa cho gọn
    lines = [ln.replace("\t", "    ").rstrip() for ln in raw]

    # mô phỏng terminal wrap cho các dòng quá dài (ví dụ file JSON)
    if wrap:
        wrapped = []
        for ln in lines:
            if len(ln) <= wrap:
                wrapped.append(ln)
                continue
            indent = len(ln) - len(ln.lstrip(" "))
            cont = " " * min(indent + 4, wrap - 10)
            first = True
            while ln:
                cut = wrap if first else wrap - len(cont)
                if first:
                    wrapped.append(ln[:wrap])
                    ln = ln[wrap:]
                    first = False
                elif ln:
                    wrapped.append(cont + ln[:cut])
                    ln = ln[cut:]
        lines = wrapped

    font = ImageFont.truetype(FONT_REG, size)
    font_bold = ImageFont.truetype(FONT_BOLD, size)

    probe = Image.new("RGB", (10, 10))
    d0 = ImageDraw.Draw(probe)
    cw = d0.textlength("M", font=font)          # advance width của 1 ký tự
    lh = int(size * LINE_SP)

    cols = max((len(ln) for ln in lines), default=1)
    if max_cols:
        cols = min(cols, max_cols)

    text_w = int(cw * cols)
    img_w = text_w + PAD * 2
    img_h = lh * len(lines) + PAD * 2 + TITLEBAR

    img = Image.new("RGB", (img_w, img_h), BG)
    d = ImageDraw.Draw(img)

    # --- thanh tiêu đề cửa sổ ---
    d.rectangle([0, 0, img_w, TITLEBAR], fill=CHROME)
    d.line([(0, TITLEBAR), (img_w, TITLEBAR)], fill=BORDER)
    for i, col in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        cx = 18 + i * 18
        d.ellipse([cx - DOT_R, TITLEBAR // 2 - DOT_R,
                   cx + DOT_R, TITLEBAR // 2 + DOT_R], fill=col)
    if title:
        tw = d.textlength(title, font=font)
        d.text(((img_w - tw) / 2, TITLEBAR // 2 - size * 0.62),
               title, font=font, fill=DIM)

    # --- nội dung ---
    y = TITLEBAR + PAD
    for ln in lines:
        color = classify(ln)
        fnt = font_bold if (ln.strip() and set(ln.strip()) <= set("=-+| ")) else font
        d.text((PAD, y), ln, font=fnt, fill=color)
        y += lh

    img.save(dst)
    print(f"{dst}  {img_w}x{img_h}  ({len(lines)} lines, {cols} cols)")


if __name__ == "__main__":
    args = sys.argv[1:]
    title, wrap = None, None
    for flag in ("--title", "--wrap"):
        if flag in args:
            i = args.index(flag)
            if flag == "--title":
                title = args[i + 1]
            else:
                wrap = int(args[i + 1])
            args = args[:i] + args[i + 2:]
    render(args[0], args[1], title=title, wrap=wrap)