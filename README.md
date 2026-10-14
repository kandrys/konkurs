# Soil grain size from photos

Kaggle solution for [Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos) (GRID / BOKU).

The task is to predict an 11-point cumulative grain-size curve (DIN EN ISO 14688-1, 0.002–200 mm) from smartphone photos of a soil surface. The metric is a log-weighted Earth Mover’s Distance on that CDF, scaled to `[0, 500]` (lower is better).

## Approach

- **Backbone:** frozen [`facebook/dinov2-with-registers-small`](https://huggingface.co/facebook/dinov2-with-registers-small) (patch 14, 448×448 crops).
- **Head:** LayerNorm → MLP → two Weibull parameters \((\ln\lambda, k)\). The 11-bin CDF is reconstructed as \(F(d)=1-\exp(-(d/\lambda)^k)\), not 11 independent softmax bins.
- **Scale:** every photo is resized to **4.55 pixels/mm** using `data/ppm_updated.csv`, then tray/ruler frames are cropped. Training uses `PadIfNeeded` + `RandomCrop` so the physical scale stays intact (`RandomResizedCrop` would break ppm).
- **Split:** group-aware holdout (related sample IDs stay together), 19 train / 5 val soils at seed 42.
- **Train vs test:** training photos are Android (Motorola / Samsung); Kaggle test is iPhone 14/16. In-domain val is much more optimistic than the public leaderboard.

Optional: `--unfreeze_last N` with a lower `--lr_backbone` (last encoder blocks + two AdamW groups). On this dataset that overfit immediately; the working run is fully frozen.

## Results

| Setup | Score |
| --- | --- |
| Weibull fit on labels only (parameterization ceiling) | ~8 EMD |
| Frozen DINOv2-S + Weibull, Android val (5 soils) | **31.94** |
| Same checkpoint, Kaggle public LB (3 iPhone soils) | **~64** |
| Mean train CDF as a constant predictor (train soils) | ~86 |

Public LB is 3 of 10 test soils. The jump 32 → 64 is mainly the phone domain gap, not a broken submission file.

## Repository layout

```
train.py          # training, Weibull head, EMD, group split
predict.py        # iPhone EXIF, ppm resize, tiles → submission.csv
config.py         # crop size, ppm, optional unfreeze LRs
save_img.py       # debug sheets when building processed/
plot_morph.py     # morphology spectrum helper (experiment)
data/             # labels, ppm table, sample_submission (photos not in git)
processed/        # local ppm-normalized training PNGs (gitignored)
submission.csv    # latest Kaggle-format predictions
```

Raw photos and `*.pt` checkpoints are gitignored (too large). Put contest images at:

- `data/Training-All_Photos_updated/Training-All_Photos_updated/`
- `data/Test_All_Photos/Test_All_Photos/`

Training reads **`processed/`**, not the raw folder. Build it once by running `load_images(..., debug=True)` from `train.py` (writes PNGs via `save_img.py`).

## Setup

Python 3.10+ with PyTorch (CUDA or Apple MPS), then:

```text
torch
transformers
albumentations
opencv-python
numpy
scipy
Pillow
```

Device is `mps` if available, otherwise CPU.

## Train

```bash
python train.py --freeze_backbone --lr 1e-5 --epochs 8 --batch_size 8
```

Saves `best.pt` on the best val EMD. Useful flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--freeze_backbone` | off | freeze DINOv2 (use this) |
| `--unfreeze_last` | 0 | last N encoder blocks trainable |
| `--lr` / `--lr_backbone` | 1e-5 / 1e-6 | head vs backbone AdamW groups |
| `--ckpt` | none | load a previous `state_dict` |
| `--crops_per_image` | 16 | random crops per photo per epoch |

## Predict and submit

```bash
python predict.py --ckpt best.pt --out submission.csv

kaggle competitions submit -c soil-grain-size-from-photos -f submission.csv -m "frozen dinov2-s weibull"
```

`predict.py` matches iPhone 14/16 stems (including the `iPhone_16_` filename typo), applies EXIF orientation, resizes to 4.55 ppm, drops frames, averages 448 tiles, writes percent CDFs that are clipped, made non-decreasing, and forced to 100 at 200 mm.

## Metric

For CDFs in `[0, 1]` on support \(d_i\):

\[
\mathrm{EMD}=\sum_{i=1}^{10} \lvert F_{\mathrm{pred}}(d_i)-F_{\mathrm{true}}(d_i)\rvert \cdot \log_{10}(d_{i+1}/d_i)
\]

Training/validation multiply by 100 to match Kaggle’s reported scale.
