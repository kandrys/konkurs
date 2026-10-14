"""Bar chart of morph features for one processed photo.

Before = pattern spectrum (open/close at 5 radii).
After  = spectrum + std + Laplacian var + gray percentiles.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from train import TARGET_PPM, _close_r, _open_r, load_image_rgb, morph_features

RADII_MM = [0.2, 0.63, 2.0, 6.3, 20.0]
SPECTRUM_LABELS = [f"open {r}" for r in RADII_MM] + [f"close {r}" for r in RADII_MM]
EXTRA_LABELS = ["std", "lap_var", "p5", "p25", "p50", "p75", "p95"]


def leftover_mass(img: np.ndarray) -> tuple[float, float]:
    """Bright / dark mass still there after the 20 mm disk — not in morph_features yet."""
    gray_u8 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = gray_u8.astype(np.float32) / 255.0
    r = max(1, int(round(RADII_MM[-1] * TARGET_PPM)))
    open_rest = float(_open_r(gray_u8, r).mean() / 255.0)
    close_rest = float(_close_r(gray_u8, r).mean() / 255.0)
    return open_rest, close_rest - gray.mean()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).parent / "processed" / "Samsung_A52_H371_04.png",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "morph_spectrum.png",
    )
    args = p.parse_args()

    img = load_image_rgb(args.image)
    feat = morph_features(img)
    if feat.shape[0] != 17:
        raise RuntimeError(f"expected 17 features, got {feat.shape}")

    spectrum, extra = feat[:10], feat[10:]
    open_rest, close_delta = leftover_mass(img)

    print(f"image={args.image.name}  shape={img.shape}  ppm={TARGET_PPM}")
    print("--- spectrum (pattern, should be >= 0) ---")
    for name, val in zip(SPECTRUM_LABELS, spectrum):
        print(f"  {name:12s}  {val:.5f}")
    print("--- extra ---")
    for name, val in zip(EXTRA_LABELS, extra):
        print(f"  {name:12s}  {val:.5f}")
    print(f"--- not in the vector yet ---")
    print(f"  open leftover >20mm (mean after last open)  {open_rest:.5f}")
    print(f"  close leftover vs mean                      {close_delta:.5f}")
    print(f"zeros in spectrum: {int(np.sum(np.abs(spectrum) < 1e-8))} / 10")
    print(f"any NaN: {bool(np.isnan(feat).any())}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), width_ratios=[1.1, 1.4, 1.2])

    axes[0].imshow(img)
    axes[0].set_title(args.image.stem)
    axes[0].axis("off")

    colors = ["#3b82f6"] * 5 + ["#f59e0b"] * 5
    axes[1].bar(range(10), spectrum, color=colors)
    axes[1].set_xticks(range(10), SPECTRUM_LABELS, rotation=45, ha="right")
    axes[1].set_ylabel("Δ mean intensity")
    axes[1].set_title("przed: samo spektrum (open / close)")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].legend(
        handles=[
            mpatches.Patch(color="#3b82f6", label="open (jasne ziarna)"),
            mpatches.Patch(color="#f59e0b", label="close (ciemne ziarna)"),
        ],
        loc="upper right",
    )

    all_labels = SPECTRUM_LABELS + EXTRA_LABELS
    all_colors = colors + ["#22c55e"] * 2 + ["#a78bfa"] * 5
    axes[2].bar(range(17), feat, color=all_colors)
    axes[2].set_xticks(range(17), all_labels, rotation=45, ha="right")
    axes[2].set_title("po: spektrum + std/laplace + percentyle")
    axes[2].axhline(0.0, color="black", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
