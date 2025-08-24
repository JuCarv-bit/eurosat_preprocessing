eurosatpath = "/share/DEEPLEARNING/carvalhj/EuroSAT_RGB/"

import os, glob, random, sys, string
from pathlib import Path
from typing import Optional, List
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

seed = 42
save_path = "eurosat_2x5_panels.png"
tile_size = 192
gutter = 6
outer_border = 4
label_fontsize = 14
fig_dpi = 220
save_dpi = 400

random.seed(seed)

def _is_dir_with_subdirs(p: Path) -> bool:
    return p.is_dir() and any(d.is_dir() for d in p.iterdir())

def _class_root(base: Path) -> Path:
    p2750 = base / "2750"
    if _is_dir_with_subdirs(p2750):
        return p2750
    return base

def _list_classes(base: Path) -> List[Path]:
    root = _class_root(base)
    classes = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not classes:
        raise RuntimeError(f"No class folders found under {root}. Expected .../2750/<ClassName>/image.jpg or .../<ClassName>/image.jpg")
    classes = sorted(classes, key=lambda p: p.name.lower())
    return classes

def _list_images(class_dir: Path) -> List[str]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp", "*.npy"]
    files = []
    for pat in exts:
        files.extend(glob.glob(str(class_dir / pat)))
    return sorted(files)

def _load_image_any(path: str) -> Image.Image:
    if path.lower().endswith(".npy"):
        arr = np.load(path)
        if arr.ndim == 3:
            if arr.shape[0] == 3 and (arr.shape[2] not in (3, 4)):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        elif arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        else:
            raise ValueError(f"Unsupported .npy shape {arr.shape} in {path}")
    else:
        with Image.open(path) as im:
            return im.convert("RGB")

def _make_2x2_tile(img_paths: List[str], tile_size: int = 192, gutter: int = 6, outer_border: int = 4, bg=(255, 255, 255)) -> Image.Image:
    imgs = []
    for p in img_paths:
        try:
            im = _load_image_any(p)
            im = im.resize((tile_size, tile_size), Image.LANCZOS)
            imgs.append(im)
        except Exception as e:
            print(f"[ERROR] Failed to load {p}: {e}", file=sys.stderr)
    while len(imgs) < 4 and len(imgs) > 0:
        imgs.append(imgs[len(imgs) % len(imgs)])
    if len(imgs) == 0:
        w = 2*tile_size + 2*outer_border + gutter
        h = 2*tile_size + 2*outer_border + gutter
        return Image.new("RGB", (w, h), bg)
    W = 2*tile_size + 2*outer_border + gutter
    H = 2*tile_size + 2*outer_border + gutter
    canvas = Image.new("RGB", (W, H), bg)
    xs = [outer_border, outer_border + tile_size + gutter]
    ys = [outer_border, outer_border + tile_size + gutter]
    positions = [(xs[0], ys[0]), (xs[1], ys[0]), (xs[0], ys[1]), (xs[1], ys[1])]
    for im, (x, y) in zip(imgs[:4], positions):
        canvas.paste(im, (x, y))
    return canvas

def show_eurosat_2x5_quads(eurosatpath: str,
                           seed: int = 42,
                           save_path: Optional[str] = None,
                           tile_size: int = 192,
                           gutter: int = 6,
                           outer_border: int = 4,
                           label_fontsize: int = 14,
                           fig_dpi: int = 220,
                           save_dpi: int = 400):
    base = Path(eurosatpath)
    if not base.exists():
        raise FileNotFoundError(f"Path does not exist: {base}")
    random.seed(seed)
    class_dirs = _list_classes(base)
    target = 10
    if len(class_dirs) < target:
        print(f"[WARN] Only {len(class_dirs)} classes found; grid shows {target}. Empty slots will be blank.", file=sys.stderr)
    elif len(class_dirs) > target:
        print(f"[INFO] Found {len(class_dirs)} classes; showing first {target} alphabetically.", file=sys.stderr)
    picked = class_dirs[:target]
    items = []
    for cdir in picked:
        imgs = _list_images(cdir)
        if not imgs:
            print(f"[WARN] No images in class '{cdir.name}'.", file=sys.stderr)
            items.append((cdir.name, []))
        else:
            picks = imgs if len(imgs) <= 4 else random.sample(imgs, 4)
            while len(picks) < 4:
                picks.append(picks[len(picks) % len(picks)])
            items.append((cdir.name, picks))
    rows, cols = 2, 5
    fig_w, fig_h = 3.8 * cols, 4.3 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=fig_dpi)
    plt.subplots_adjust(wspace=0.35, hspace=0.55, bottom=0.08)
    letters = list(string.ascii_lowercase)
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.axis("off")
        if idx < len(items):
            cls_name, img_paths = items[idx]
            tile = _make_2x2_tile(img_paths, tile_size=tile_size, gutter=gutter, outer_border=outer_border)
            ax.imshow(tile, interpolation="lanczos")
            tag = f"({letters[idx]}) {cls_name}" if idx < 26 else cls_name
            ax.text(0.5, -0.10, tag, transform=ax.transAxes, ha="center", va="top", fontsize=label_fontsize, fontweight="bold", clip_on=False)
        else:
            ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=18)
    if save_path:
        plt.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved figure to: {save_path} @ {save_dpi} DPI")
    plt.show()

show_eurosat_2x5_quads(eurosatpath, seed=seed, save_path=save_path,
                       tile_size=tile_size, gutter=gutter, outer_border=outer_border,
                       label_fontsize=label_fontsize, fig_dpi=fig_dpi, save_dpi=save_dpi)
