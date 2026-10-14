from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from config import Config
from train import (
    TARGET_PPM,
    ViT,
    _tiles,
    build_backbone_and_meta,
    build_transforms,
    list_image_path,
    remove_frames,
)

HEADER = [
    "sample_id",
    "0.002",
    "0.0063",
    "0.02",
    "0.063",
    "0.2",
    "0.63",
    "2",
    "6.3",
    "20",
    "63",
    "200",
]


def load_rgb_exif(path: Path) -> np.ndarray:
    """cv2.IMREAD_COLOR ignores EXIF; iPhone 14/16 need orientation 3/6."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        arr = np.asarray(im.convert("RGB"))
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image: {path}")
    return arr


def load_ppm_table(path: Path) -> List[Dict]:
    with path.open(mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["phone"] = row["phone"].lower().replace(" ", "_")
        row["ppm"] = float(row["ppm"])
        row["height"] = int(row["height"])
        row["width"] = int(row["width"])
    rows.sort(key=lambda x: len(x["phone"]), reverse=True)
    return rows


def parse_test_stem(stem: str) -> Tuple[str, str]:
    stem_nfc = unicodedata.normalize("NFC", stem)
    low = stem_nfc.lower()
    if low.startswith("iphone_16_") or low.startswith("iphone16_"):
        phone = "iphone_16"
    elif low.startswith("iphone_14_") or low.startswith("iphone14_"):
        phone = "iphone_14"
    else:
        raise ValueError(f"Unknown phone prefix: {stem}")
    idx = stem_nfc.upper().find("HPC_")
    if idx < 0:
        raise ValueError(f"No HPC_ in stem: {stem}")
    rest = re.sub(r"\s*\(\d+\)\s*$", "", stem_nfc[idx:])
    rest = rest.replace("ü", "ue").replace("Ü", "Ue").replace(",", "_")
    return phone, rest


def resize_to_target_ppm(img: np.ndarray, row: Dict, path: Path) -> np.ndarray:
    height, width = img.shape[:2]
    scale = max(height, width) / max(row["height"], row["width"])
    scale2 = min(height, width) / min(row["height"], row["width"])
    if np.abs((scale - scale2)) / scale >= 0.01:
        raise ValueError(f"Too high scale difference {path} {scale} {scale2}")
    ppm = scale * row["ppm"]
    factor = TARGET_PPM / ppm
    return cv2.resize(
        img,
        (int(round(width * factor)), int(round(height * factor))),
        interpolation=cv2.INTER_AREA,
    )


def pad_if_needed(img: np.ndarray, crop: int) -> np.ndarray:
    h, w = img.shape[:2]
    pad_h = max(0, crop - h)
    pad_w = max(0, crop - w)
    if pad_h or pad_w:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    return img


def load_submission_ids(path: Path) -> List[str]:
    with path.open(mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["sample_id"] for r in rows]


@torch.no_grad()
def predict_image(
    model: ViT,
    img: np.ndarray,
    cfg: Config,
    tfm,
    device: torch.device,
) -> torch.Tensor:
    img = pad_if_needed(img, cfg.crop_size)
    sample = ("", img, 0)
    coords = _tiles(sample, cfg)
    preds = []
    batch_x: List[torch.Tensor] = []

    def flush():
        if not batch_x:
            return
        x = torch.stack(batch_x, dim=0).to(device)
        out = model(x).cpu()
        preds.extend(list(out))
        batch_x.clear()

    for _, y, x0 in coords:
        tile = img[y : y + cfg.crop_size, x0 : x0 + cfg.crop_size]
        batch_x.append(tfm(image=tile)["image"])
        if len(batch_x) >= cfg.batch_size:
            flush()
    flush()
    return torch.stack(preds, dim=0).mean(0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--model_name", type=str, default="facebook/dinov2-with-registers-small")
    args = p.parse_args()

    root = Path(__file__).parent
    ckpt_path = args.ckpt or (root / "best.pt")
    out_path = args.out or (root / "submission.csv")
    sample_csv = root / "data" / "sample_submission.csv"
    test_dir = root / "data" / "Test_All_Photos" / "Test_All_Photos"
    ppm_path = root / "data" / "ppm_updated.csv"

    cfg = Config(
        accum_steps=1,
        crop_size=448,
        batch_size=args.batch_size,
        freeze_backbone=True,
        crops_per_image=1,
        lr=1e-5,
        seed=42,
        epochs=1,
        eval_stride=448,
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tfm = build_transforms(train=False, cfg=cfg)

    wanted = load_submission_ids(sample_csv)
    wanted_set = set(wanted)
    ppm_rows = load_ppm_table(ppm_path)
    by_phone = {r["phone"]: r for r in ppm_rows}

    x_sup = np.array(
        [0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0],
        dtype=np.float32,
    )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    in_dim = int(sd["head.0.weight"].shape[0])
    n_out = int(sd["head.5.weight"].shape[0])
    if in_dim != 1152 or n_out != 2:
        raise RuntimeError(
            f"{ckpt_path} has LayerNorm={in_dim} Linear_out={n_out}, "
            "expected 1152 and 2 (Weibull ViT). Retrain current train.py."
        )

    backbone, meta = build_backbone_and_meta(args.model_name, rank0=True)
    print(f"ckpt={ckpt_path} device={device} backbone={meta['model_id']} in_dim={in_dim} n_out={n_out}")
    model = ViT(backbone, x_sup, freeze_backbone=True).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    by_sample: Dict[str, List[torch.Tensor]] = defaultdict(list)
    n_used = 0
    for path in list_image_path(test_dir):
        if "hammer" in path.stem.lower():
            print(f"skip hammer: {path.name}")
            continue
        phone, sample_id = parse_test_stem(path.stem)
        if sample_id not in wanted_set:
            raise ValueError(f"{path.name} -> '{sample_id}' not in sample_submission")
        row = by_phone.get(phone)
        if row is None:
            raise FileNotFoundError(f"No ppm row for {phone} ({path.name})")
        img = load_rgb_exif(path)
        img = resize_to_target_ppm(img, row, path)
        img = remove_frames(img)
        cdf = predict_image(model, img, cfg, tfm, device)
        by_sample[sample_id].append(cdf)
        n_used += 1
        print(f"{path.name} -> {sample_id} tiles_ok shape={img.shape}")

    missing = [s for s in wanted if s not in by_sample]
    if missing:
        raise RuntimeError(f"No test photos for sample_ids: {missing}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(HEADER)
        for sid in wanted:
            cdf = torch.stack(by_sample[sid], dim=0).mean(0).numpy()
            pct = np.clip(cdf * 100.0, 0.0, 100.0)
            pct = np.maximum.accumulate(pct)
            pct[-1] = 100.0
            writer.writerow([sid] + [f"{x:.4f}" for x in pct.tolist()])
    print(f"wrote {out_path} from {n_used} photos / {len(wanted)} samples")


if __name__ == "__main__":
    main()
