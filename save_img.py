from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path

import config

def save_debug_images(items: List[Tuple[str, np.ndarray, np.ndarray]], out_dir: Path,cfg:Config,
                      thumb_h: int = 300, cols: int = 3, rows_per_sheet: int = 6) -> None:
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    tiles = []
    for name, before, after in items:
        cv2.imwrite(str(out_dir / f"{name}.png"), cv2.cvtColor(after, cv2.COLOR_RGB2BGR))

        w_mm, h_mm = after.shape[1] / cfg.target_ppm, after.shape[0] / cfg.targer_ppm
        print(f"{name:40s} {before.shape[1]}x{before.shape[0]} -> "
              f"{after.shape[1]}x{after.shape[0]}  ({w_mm:.0f}x{h_mm:.0f} mm)")

        thumbs = [cv2.resize(im, (int(im.shape[1] * thumb_h / im.shape[0]), thumb_h),
                             interpolation=cv2.INTER_AREA) for im in (before, after)]
        gap = np.full((thumb_h, 10, 3), 255, np.uint8)
        tile = np.hstack([thumbs[0], gap, thumbs[1]])
        cv2.putText(tile, f"{name}  {w_mm:.0f}x{h_mm:.0f} mm", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        tiles.append(tile)

    tile_w = max(t.shape[1] for t in tiles)
    tiles = [np.pad(t, ((0, 0), (0, tile_w - t.shape[1]), (0, 0)), constant_values=255) for t in tiles]
    blank = np.full_like(tiles[0], 255)

    per_sheet = cols * rows_per_sheet
    for sheet_idx, start in enumerate(range(0, len(tiles), per_sheet)):
        chunk = tiles[start:start + per_sheet]
        chunk += [blank] * (-len(chunk) % cols)
        sheet = np.vstack([np.hstack(chunk[r:r + cols]) for r in range(0, len(chunk), cols)])
        cv2.imwrite(str(debug_dir / f"sheet_{sheet_idx:02d}.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
