
from dataclasses import dataclass


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

@dataclass
class Config:
    accum_steps: int
    crop_size: int
    batch_size: int
    freeze_backbone: bool
    crops_per_image: int
    lr: float
    seed: int
    epochs: int
    eval_stride: int
    unfreeze_last: int = 0
    lr_backbone: float = 1e-6
    target_ppm: float = 4.55
