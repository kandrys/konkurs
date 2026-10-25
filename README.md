# Soil Grain Size from Photos — Kaggle (GRID / BOKU)

Predicting an 11-point cumulative grain-size distribution (DIN EN ISO 14688-1, 0.002–200 mm) from smartphone
photos of a soil surface, for the Kaggle competition
[Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos).
Metric: log-weighted Earth Mover's Distance on the CDF, scaled to `[0, 500]` (lower is better).

**Validation score: 31.94 EMD** (frozen DINOv2-S + Weibull head, in-domain Android val, 5 soils).

## Approach and design decisions

### Weibull parameterization instead of 11 independent bins

The most consequential architectural choice. A naive head would predict 11 softmax bins and reconstruct the CDF
by cumsum — but a grain-size distribution is not arbitrary: it is unimodal, monotone, and physically constrained
to follow a roughly log-normal shape over several decades of particle diameter.

Instead, the head outputs two Weibull parameters (ln λ, k), and the CDF is reconstructed analytically as:

$$F(d) = 1 - \exp\left(-(d/\lambda)^k\right)$$

This reduces the prediction to a 2D manifold inside the 11-dimensional output space, which is a much tighter
inductive bias.

To validate the ceiling of this approach, I first fit Weibull parameters directly to the ground-truth labels —
the resulting EMD was **~8**, meaning the parameterization itself can represent the true distributions well.
Everything above that is the model's vision error, not a modeling artifact.

### Physical scale preservation

Each photo is resized to a canonical **4.55 pixels/mm** using per-image resolution metadata
(`data/ppm_updated.csv`), and tray/ruler frames are cropped before any augmentation.

This rules out `RandomResizedCrop`, which would silently change the physical scale and destroy the texture
frequency signal the model relies on. Instead, training uses `PadIfNeeded` + `RandomCrop` at 448×448, which
samples different regions of the same physical-scale image.

### Frozen backbone

Backbone: [`facebook/dinov2-with-registers-small`](https://huggingface.co/facebook/dinov2-with-registers-small)
(patch 14, 448×448), fully frozen.

DINOv2 was self-supervised on diverse visual data and its patch features already encode fine-grained texture
well. I experimented with `--unfreeze_last N` (last N encoder blocks + two separate AdamW groups with a lower
backbone LR), but on a dataset this small — 19 training soils — unfreezing caused immediate overfitting with no
val improvement. Frozen + head-only training is the stable regime here.

### Group-aware split

Soil samples share related photos across the dataset. A naive random split leaks: photos from the same soil
appear in both train and val. The split (`seed=42`) keeps related IDs together, leaving 19 soils for training
and 5 for validation.

### Domain gap: Android train → iPhone test

Training photos are from Android devices (Motorola / Samsung). Kaggle's test set is iPhone 14/16. The in-domain
Android val (31.94 EMD) is substantially more optimistic than what the public leaderboard reflects — the gap is
expected given different sensor characteristics, JPEG processing pipelines, and image sharpening.

`predict.py` handles iPhone-specific quirks: EXIF orientation, a filename typo (`iPhone_16_`), and averages
predictions across multiple 448-px tiles per image.

## Results

| Setup | EMD (val) |
| --- | --- |
| Weibull fit on ground-truth labels (parameterization ceiling) | ~8 |
| Frozen DINOv2-S + Weibull head, Android val (5 soils) | **31.94** |

## Repository

```
train.py          # training loop, Weibull head, EMD loss, group split
predict.py        # iPhone EXIF handling, ppm resize, tile averaging → submission.csv
config.py         # crop size, ppm target, optional unfreeze LRs
save_img.py       # debug sheets for processed/
plot_morph.py     # morphology spectrum visualisation (experimental)
data/             # labels, ppm table, sample_submission
processed/        # ppm-normalised training PNGs (gitignored)
submission.csv    # latest Kaggle-format predictions
```

Raw photos and `*.pt` checkpoints are gitignored. Place contest images at:

- `data/Training-All_Photos_updated/Training-All_Photos_updated/`
- `data/Test_All_Photos/Test_All_Photos/`

Build `processed/` once by calling `load_images(..., debug=True)` from `train.py`.

## Setup

Python 3.10+, PyTorch (CUDA or Apple MPS):

```text
torch
transformers
albumentations
opencv-python
numpy
scipy
Pillow
```

## Train

```bash
python train.py --freeze_backbone --lr 1e-5 --epochs 8 --batch_size 8
```

Saves `best.pt` on the best val EMD.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--freeze_backbone` | off | freeze DINOv2 (recommended) |
| `--unfreeze_last` | 0 | last N encoder blocks trainable |
| `--lr` / `--lr_backbone` | 1e-5 / 1e-6 | head vs backbone LR |
| `--ckpt` | none | resume from checkpoint |
| `--crops_per_image` | 16 | random crops per photo per epoch |

## Predict and submit

```bash
python predict.py --ckpt best.pt --out submission.csv
kaggle competitions submit -c soil-grain-size-from-photos -f submission.csv -m "frozen dinov2-s weibull"
```

Predictions are clipped to `[0, 100]`, made non-decreasing, and forced to 100 at 200 mm.

## Metric

$$\mathrm{EMD} = \sum_{i=1}^{10} \left|F_\mathrm{pred}(d_i) - F_\mathrm{true}(d_i)\right| \cdot \log_{10}\left(\frac{d_{i+1}}{d_i}\right)$$

Multiplied by 100 to match Kaggle's reported scale.
