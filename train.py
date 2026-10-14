
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from save_img import save_debug_images
from config import IMG_EXTS,Config

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.optimize import curve_fit


import cv2
import csv 
import albumentations as A
from albumentations.pytorch import ToTensorV2

from transformers import AutoModel, AutoImageProcessor


TARGET_PPM = Config.target_ppm
D = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0], dtype=np.float64)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True



def load_images(path: Path,debug: bool = False) -> Dict:
    path_img = path / "Training-All_Photos_updated" / "Training-All_Photos_updated"
    paths = list_image_path(path_img)
    path_ppm = path / "ppm_updated.csv"
    with path_ppm.open(mode="r",encoding="utf-8") as file:
        ppm_data = list(csv.DictReader(file))
    for a in ppm_data:
        a["phone"] = a["phone"].lower().replace(" ","_")
        a["ppm"] = float(a["ppm"])
        a["height"] = int(a["height"])
        a["width"] = int(a["width"])
    ppm_data.sort(key=lambda x:len(x["phone"]),reverse = True)
    data = {}
    images = {}
    to_debug = []
    for img_path in paths:
        img = img_path.stem
        for row in ppm_data:
            if img.lower().startswith(row["phone"]):
                data = row
                break
        else:
            raise FileNotFoundError(f"Unknown Phone: {img}")
        img2 = normalize(img_path,data)
        img1 = remove_frames(img2)
        to_debug.append((img,img2,img1))
        images[img] = img1
    if debug:
        save_debug_images(to_debug,path.parent / "processed")
    return images
def normalize(img_path: Path,data : Dict ) -> np.ndarray:
    img = load_image_rgb(img_path)
    height,width= img.shape[:2]
    scale = max(height,width) / max(data["height"],data["width"])
    scale2 = min(height,width) / min(data["height"],data["width"])
    if np.abs((scale - scale2))/ scale >= 0.01:
        raise ValueError(f"Too high scale difference {img_path} {scale} {scale2}")
    ppm = scale * data["ppm"]
    factor = TARGET_PPM / ppm 
    return cv2.resize(img,(int(round(width * factor)),int(round(height * factor))), interpolation = cv2.INTER_AREA)
def remove_frames(img : np.ndarray,margin_mm: float = 8.0, k :float = 5.0, min_diff: float = 0.2,confirm_mm: float = 6.0) -> np.ndarray:
    height,width = img.shape[:2]
    gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    blur_img = cv2.GaussianBlur(gray_img,(5,5),0).astype(np.float32)
    rows_profile = np.median(blur_img[:, width // 4 : width - width // 4], axis=1)
    cols_profile = np.median(blur_img[height // 4 : height - height // 4, :], axis=0)
    w = int(round(3*TARGET_PPM))
    kernel = cv2.getGaussianKernel(w, TARGET_PPM)
    confirm_px = int(round(confirm_mm * TARGET_PPM))
    max_px = int(round(40 * TARGET_PPM)) 


    def find_edge(profile: np.ndarray,dir: int) -> int:
        n = len(profile)
        profile = np.convolve(np.pad(profile, w//2, mode="edge"), kernel.ravel(), mode="valid")[:n]
        ref = np.median(profile[n//4 : n - n//4])
        grad = np.abs(np.gradient(profile))
        thr = max(k*np.median(grad[n //4 : n -n//4]),0.5)
        end = -1 if dir < 0 else n
        for i in range(n //2,end,dir):
            if grad[i] <= thr or (dir < 0 and i > max_px) or (dir > 0 and i < n - 1 - max_px):
                continue
            lo,hi = (i,min(n,i+confirm_px)) if dir >0 else (max(0,i-confirm_px),i+1)
            if abs(np.median(profile[lo:hi]-ref)) / (ref +1e-6) > min_diff:
                return i 
        return 0 if dir < 0 else n-1
    m = int(round(margin_mm * TARGET_PPM))
    y0 = find_edge(rows_profile, -1) + m
    y1 = find_edge(rows_profile, +1) - m
    x0 = find_edge(cols_profile, -1) + m
    x1 = find_edge(cols_profile, +1) - m
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"Empty crop box: {(x0, y0, x1, y1)}")
    return img[y0:y1, x0:x1]
def list_image_path(img_dir: Path) -> List[Path]:
    paths = []
    for p in sorted(img_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return paths

def load_image_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
def load_image_from_processed(path: Path) -> Dict[str,np.ndarray]:
    paths = list_image_path(path)
    return {p.stem: load_image_rgb(p) for p in paths}
def read_labels(path: Path)-> Dict:
    data = {}
    path_labels = path / "Training_labels_updated.csv"
    with path_labels.open(mode="r",encoding="utf-8") as file:
        a = csv.reader(file)
        one = next(a)
        for row in a:
            sample_id = row[0]
            array = [float(x)/100 for x in row[1:]]
            data[sample_id] = array
    return data
def build_transforms(train: bool,cfg: Config):
    if train: 
        return A.Compose([
            A.PadIfNeeded(min_height=cfg.crop_size, min_width=cfg.crop_size, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
            A.RandomCrop(height=cfg.crop_size, width=cfg.crop_size, p=1.0),
            A.D4(p=1.0),
            A.OneOf([
                A.Illumination(mode="linear",   intensity_range=(0.05, 0.2), effect_type="both"),
                A.Illumination(mode="corner",   intensity_range=(0.05, 0.2), effect_type="both"),
                A.Illumination(mode="gaussian", intensity_range=(0.05, 0.2), effect_type="both",
                               sigma_range=(0.3, 1.0)),
                ], p=0.5),                
            A.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2,p=0.8),
            A.RandomGamma(gamma_limit=(80,120),p=0.3),
            A.PlanckianJitter(p=0.5 ),
            A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(32, 64), 
                            hole_width_range=(16, 32), fill=0, p=0.5),
            A.HueSaturationValue(hue_shift_limit=5,sat_shift_limit=20,val_shift_limit=0,p=0.5),
            A.ToGray(p=0.2),
            A.ImageCompression(quality_range=(60,95),p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0, normalization="standard", p=1.0),
            ToTensorV2(),
            ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0, normalization="standard", p=1.0),
            ToTensorV2(),
            ])

def _tiles(img: Tuple[str,np.ndarray,int],cfg: Config) -> list(Tuple[int,int,int]):

    _,arr,idx = img
    if not isinstance(arr,np.ndarray):
        raise ValueError(f"Wrong img type {type(arr)}")
    H,W = arr.shape[:2]
    def _pos(L: int,C: int,stride: int) ->list[int]:
        if  L<C:
            raise ValueError(" L< C")
        pos = list(range(0,L-C+1,stride))
        if pos[-1] != L-C:
            pos.append(L-C)
        return pos

    ys = _pos(H,cfg.crop_size,cfg.eval_stride)
    xs = _pos(W,cfg.crop_size,cfg.eval_stride)
    return [(idx,y,x) for y in ys for x in xs]

class SoilDataset(Dataset):
    def __init__(self, items: Dict[str,np.ndarray], labels: Dict[str, np.ndarray],
                  train: bool,cfg: Config):
        self.cfg = cfg
        self.train = train
        self.labels = labels
        self.tfm = build_transforms(train=train,cfg = cfg)
        self.samples = [(key[-7:-3], value, i) for i, (key, value) in enumerate(items.items(), start=0)]
        for key,_,_ in self.samples:
            if key in labels:
                continue
            else:
                raise ValueError(f"Sample id not in labels {key}")
        if not train: 
            self.val_samples = [tile for sample in self.samples for tile in _tiles(sample, cfg)]
    def __len__(self) -> int:
        return len(self.samples) * self.cfg.crops_per_image if self.train else len(self.val_samples)

    def __getitem__(self, idx: int):
        if self.train:
            img = self.samples[idx % len(self.samples)]
            out = self.tfm(image=img[1])
            label = self.labels[img[0]]

        else:
            img_val = self.val_samples[idx]
            z,y,x = img_val
            img = self.samples[z]
            tile = img[1][y:y+ self.cfg.crop_size,x:x + self.cfg.crop_size]
            out = self.tfm(image=tile)
            label = self.labels[img[0]]
    
        return out["image"],torch.tensor(label,dtype =torch.float32), img[2]
def get_patch_size_from_hf_config(cfg) -> int:
    ps = getattr(cfg, "patch_size", None)
    if ps is None:
        raise ValueError("HF config has no patch_size; not a ViT patch model.")
    if isinstance(ps, (tuple, list)):
        ps = ps[0]
    return int(ps)


def get_prefix_tokens_from_hf_config(cfg) -> int:
    n_reg = int(getattr(cfg, "num_register_tokens", 0) or 0)
    return 1 + n_reg  # CLS + registers


def _encoder_layers(backbone: nn.Module) -> nn.ModuleList:
    enc = getattr(backbone, "encoder", None)
    layers = getattr(enc, "layer", None) if enc is not None else None
    if layers is None:
        raise RuntimeError("Could not find backbone.encoder.layer (not a DINOv2 HF model?)")
    return layers


def set_backbone_trainable(backbone: nn.Module, unfreeze_last: int) -> None:
    for p in backbone.parameters():
        p.requires_grad = False
    if unfreeze_last <= 0:
        return
    layers = _encoder_layers(backbone)
    k = min(unfreeze_last, len(layers))
    for layer in layers[-k:]:
        for p in layer.parameters():
            p.requires_grad = True
    ln = getattr(backbone, "layernorm", None)
    if ln is not None:
        for p in ln.parameters():
            p.requires_grad = True
    print(f"unfroze last {k}/{len(layers)} encoder blocks + layernorm")
def weibull_np(d, lam, k):
    return 1.0 - np.exp(-np.power(np.clip(d / lam, 1e-12, None), k))
def fit_weibull(cdf01):  # cdf in [0,1], length 11
    y = np.clip(cdf01[:10], 1e-4, 1 - 1e-4) 
    (lam, k), _ = curve_fit(
        weibull_np, D[:10], y,
        p0=(1.0, 1.0),
        bounds=([1e-4, 0.15], [80.0, 8.0]),
        maxfev=20000,
    )
    return lam, k
def weibull_cdf(ln_lam: torch.Tensor, k: torch.Tensor, x_sup: torch.Tensor) -> torch.Tensor:
    lam = torch.exp(ln_lam).clamp(1e-4, 80.0)
    k = torch.nn.functional.softplus(k) + 0.2  
    d = x_sup.to(ln_lam.device).unsqueeze(0)   
    log_arg = k.unsqueeze(1) * (torch.log(d) - torch.log(lam).unsqueeze(1))
    log_arg = log_arg.clamp(max=20.0)
    F = 1.0 - torch.exp(-torch.exp(log_arg))
    F = F.clamp(0.0, 1.0)
    F[:, -1] = 1.0
    return F  # [B, 11]
class ViT(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        x_sup: np.ndarray,
        freeze_backbone: bool = False,
        unfreeze_last: int = 0,
    ):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("x_sup", torch.as_tensor(x_sup, dtype=torch.float32))
        self.unfreeze_last = unfreeze_last
        self.backbone_no_grad = bool(freeze_backbone) and unfreeze_last <= 0
        cfg = backbone.config
        self.patch_size = get_patch_size_from_hf_config(cfg)
        self.num_prefix_tokens = get_prefix_tokens_from_hf_config(cfg)
        self.embed_dim = int(getattr(cfg,"hidden_size",getattr(cfg,"embed_dim",0)))
        if self.embed_dim <=0:
            raise ValueError("could not infer hidden_size/embed_dim from HF config")
        self.head = nn.Sequential(
                nn.LayerNorm(1152),
                nn.Dropout(),
                nn.Linear(1152,256),
                nn.GELU(),
                nn.Dropout(p=0.5),
                nn.Linear(256,2 ),
                )
        if freeze_backbone:
            set_backbone_trainable(self.backbone, unfreeze_last)
            if self.backbone_no_grad:
                self.backbone.eval()
    def train(self,mode:bool = True):
        super().train(mode)
        if self.backbone_no_grad:
            self.backbone.eval()
        return self
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.backbone_no_grad:
            with torch.no_grad():
                out = self.backbone(
                    pixel_values=x, return_dict=True, interpolate_pos_encoding=True
                )
        else:
            out = self.backbone(
                pixel_values=x, return_dict=True, interpolate_pos_encoding=True
            )

        t = out.last_hidden_state
        patch_tokens =t[:,self.num_prefix_tokens:,:]
        cls_token = t[:,0,:]
        b,n,c = patch_tokens.shape
        g = int(math.sqrt(n))
        if g*g !=n:
            raise RuntimeError(f"Expected square patch grid, got n={n}")
        mean_patch = patch_tokens.mean(dim=1)
        std_patch = patch_tokens.std(dim=1)
        global_feat = torch.cat([cls_token,mean_patch,std_patch],dim=1)#
        raw = self.head(global_feat)          # [B, 2]
        ln_lam, k_raw = raw[:, 0], raw[:, 1]
        return weibull_cdf(ln_lam, k_raw, self.x_sup)

def _hf_from_pretrained(model_id: str, *, attn_impl: str) -> nn.Module:
    """
    Transformers recently started warning that torch_dtype is deprecated in favor of dtype.
    We'll try dtype first, fall back to torch_dtype for older versions.
    """
    kwargs = dict(attn_implementation=attn_impl)
    try:
        return AutoModel.from_pretrained(model_id, dtype=torch.float32, **kwargs)
    except TypeError:
        return AutoModel.from_pretrained(model_id, torch_dtype=torch.float32, **kwargs)


def build_backbone_and_meta(model_id: str, rank0: bool) -> Tuple[nn.Module, Dict]:
    processor = AutoImageProcessor.from_pretrained(model_id)
    mean = tuple(float(x) for x in processor.image_mean)
    std = tuple(float(x) for x in processor.image_std)

    backbone = None
    attn_used = None
    for attn in ("sdpa",):
        try:
            backbone = _hf_from_pretrained(model_id, attn_impl=attn)
            attn_used = attn
            break
        except Exception as e:
            if rank0:
                print(f"[WARN] Failed to load attn_implementation='{attn}': {type(e).__name__}: {e}")

    if backbone is None:
        raise RuntimeError("Failed to load backbone with either flash_attention_2 or sdpa.")

    # Remove pooler to prevent unused-param DDP issues.
    if hasattr(backbone, "pooler") and backbone.pooler is not None:
        if rank0:
            print("[INFO] Removing HF pooler to avoid unused parameter grads under DDP.")
        backbone.pooler = None

    patch = get_patch_size_from_hf_config(backbone.config)
    prefix = get_prefix_tokens_from_hf_config(backbone.config)
    emb = int(getattr(backbone.config, "hidden_size", getattr(backbone.config, "embed_dim", 0)))

    meta = {
        "model_id": model_id,
        "attn_implementation": attn_used,
        "mean": mean,
        "std": std,
        "patch_size": int(patch),
        "num_prefix_tokens": int(prefix),
        "embed_dim": int(emb),
    }
    return backbone, meta

def report_weibull_label_fit(labels: Dict, x_sup: np.ndarray) -> float:
    scores = []
    for sid, cdf in labels.items():
        y = np.asarray(cdf, dtype=np.float64)
        try:
            lam, k = fit_weibull(y)
        except Exception as e:
            print(f"  weibull fit {sid}: FAILED {type(e).__name__}: {e}")
            continue
        pred = torch.tensor(weibull_np(D, lam, k), dtype=torch.float32).unsqueeze(0)
        tgt = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        emd = EMD_loss(pred, tgt, x_sup).item() * 100
        scores.append(emd)
        print(f"  weibull fit {sid}: λ={lam:.4f} k={k:.4f} emd={emd:.2f}")
    if not scores:
        raise RuntimeError("Weibull fit failed for every sample")
    mean_emd = float(sum(scores) / len(scores))
    print(f"weibull label-fit mean EMD={mean_emd:.2f}  (n={len(scores)}/{len(labels)})")
    return mean_emd


def EMD_loss(pred:torch.Tensor,targets:torch.Tensor,x_sup:np.ndarray) -> torch.Tensor:
    x1 = torch.tensor(x_sup,dtype = torch.float32)
    w = torch.log10(x1[1:]) - torch.log10(x1[:-1])
    w = w.to(pred.device)
    pred = pred[:,:10]
    targets = targets[:,:10]
    abs_error = torch.abs(targets - pred) *w 
    return torch.mean(torch.sum(abs_error, dim=1))

def train_one_epoch(
        model:nn.Module,
        loader:DataLoader,
        optimizer:torch.optim.Optimizer,
        cfg:Config,
        epoch: int,
        x_sup:np.ndarray,
        device:torch.device
        )-> float:
    model.train()
    accum = max(1,cfg.accum_steps)
    optimizer.zero_grad(set_to_none=True)
    total, n = 0.0,0
    for step, (img,targets,nr) in enumerate(loader):
        img = img.to(device)
        targets = targets.to(device)
        pred = model(img)
        loss = EMD_loss(pred,targets,x_sup)
        loss = loss/ accum
        loss.backward()
        if (step+1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total += float(loss.detach().item()) * accum
        n+=1
    if n%accum !=0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    lrs = ", ".join(f"{g['lr']:.2e}" for g in optimizer.param_groups)
    print(f"LR na koniec epoki: {lrs}")
    return total / max(n,1)
def sample_id_of(stem: str) -> str:
    return stem.split("_")[-2]


def split_sample_ids(labels: Dict, seed: int, n_val: int = 5) -> Tuple[set, set]:
    families = [
        ["H181", "H183"],
        ["H366", "H367", "H368", "H371", "H372", "H374"],
    ]
    in_family = {sid for family in families for sid in family}
    groups = [list(family) for family in families]
    for sid in labels:
        if sid not in in_family:
            groups.append([sid])

    random.Random(seed).shuffle(groups)

    val_ids: List[str] = []
    train_ids: List[str] = []
    for group in groups:
        if len(val_ids) < n_val and not (len(group) > 3 and val_ids):
            val_ids.extend(group)
        else:
            train_ids.extend(group)
    return set(train_ids), set(val_ids)


def filter_images(imgs: Dict[str, np.ndarray], ids: set) -> Dict[str, np.ndarray]:
    return {stem: img for stem, img in imgs.items() if sample_id_of(stem) in ids}


def create_dataloaders(imgs: Dict[str, np.ndarray], labels: Dict, cfg: Config
                       ) -> Tuple[DataLoader, DataLoader]:
    train_ids, val_ids = split_sample_ids(labels, cfg.seed)
    train_ds = SoilDataset(filter_images(imgs, train_ids), labels, train=True, cfg=cfg)
    val_ds = SoilDataset(filter_images(imgs, val_ids), labels, train=False, cfg=cfg)

    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader, val_ds
def validate(model:nn.Module,loader:DataLoader,dataset:SoilDataset,x_sup:np.ndarray,device:torch.device)->float:
    model.eval()
    by_img = {}
    label_img = {}
    with torch.no_grad():
        for imgs,targets,nrs in loader:
            pred = model(imgs.to(device)).cpu()
            for i, nr in enumerate(nrs.tolist()):
                by_img.setdefault(nr,[]).append(pred[i])
                label_img[nr] = targets[i]
        by_smpl = {}
        for nr,preds in by_img.items():
            sid = dataset.samples[nr][0]
            by_smpl.setdefault(sid, {"pred":[],"label":label_img[nr]})
            by_smpl[sid]["pred"].append(torch.stack(preds).mean(0))
        scores= []
        for pack in by_smpl.values():
            pred = torch.stack(pack["pred"]).mean(0,keepdim= True)
            target = pack["label"].unsqueeze(0)
            scores.append(EMD_loss(pred,target,x_sup).item()*100)
        return sum(scores)/ len(scores)

    
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--crops_per_image", type=int, default=16)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr_backbone", type=float, default=1e-6)
    p.add_argument("--unfreeze_last", type=int, default=0)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--model_name", type=str, default="facebook/dinov2-with-registers-small")
    args = p.parse_args()

    cfg = Config(
        accum_steps=args.accum_steps,
        crop_size=448,
        batch_size=args.batch_size,
        freeze_backbone=args.freeze_backbone,
        crops_per_image=args.crops_per_image,
        lr=args.lr,
        seed=args.seed,
        epochs=args.epochs,
        eval_stride=448,
        unfreeze_last=args.unfreeze_last,
        lr_backbone=args.lr_backbone,
    )
    seed_everything(cfg.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x_sup = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0],dtype=np.float32,)

    root = Path(__file__).parent
    imgs = load_image_from_processed(root / "processed")
    labels = read_labels(root / "data")
    report_weibull_label_fit(labels, x_sup)

    backbone, meta = build_backbone_and_meta(args.model_name, rank0=True)
    print(f"backbone={meta['model_id']} attn={meta['attn_implementation']} dim={meta['embed_dim']}")
    model = ViT(
        backbone,
        x_sup,
        freeze_backbone=cfg.freeze_backbone,
        unfreeze_last=cfg.unfreeze_last,
    ).to(device)
    if args.ckpt is not None:
        sd = torch.load(args.ckpt, map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=True)
        print(f"loaded ckpt {args.ckpt}")

    dl_train, dl_eval, val_ds = create_dataloaders(imgs, labels, cfg)
    print(
        f"train images={len(dl_train.dataset.samples)} "
        f"val images={len(val_ds.samples)} val tiles={len(val_ds)}"
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params={n_train} unfreeze_last={cfg.unfreeze_last}")

    head_params = [p for p in model.head.parameters() if p.requires_grad]
    backbone_params = [
        p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")
    ]
    param_groups = [{"params": head_params, "lr": cfg.lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg.lr_backbone})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=dl_train,
            optimizer=optimizer,
            cfg=cfg,
            epoch=epoch,
            x_sup=x_sup,
            device=device,
        )
        val_loss = validate(model, dl_eval, val_ds, x_sup, device)
        scheduler.step(val_loss)
        lrs = "/".join(f"{g['lr']:.2e}" for g in optimizer.param_groups)
        print(
            f"Epoch {epoch:02d} | lr={lrs} | "
            f"train_loss={train_loss:.5f} | val_emd={val_loss:.2f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), root / "best.pt")
            print(f"  saved best.pt (val_emd={best_val:.2f})")


if __name__ == "__main__":
    main()
