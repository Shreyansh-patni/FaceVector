# Dataset Documentation — AT&T / ORL Face Dataset

## Overview

FaceVector utilizes the **AT&T / ORL (Olivetti Research Laboratory) Face Dataset**, a standard benchmark dataset for face recognition research.

## Dataset Specifications

| Metric | Value |
|---|---|
| Total Subjects | 40 subjects (`s1` through `s40`) |
| Images per Subject | 10 grayscale images per subject |
| Total Dataset Images | 400 images |
| Image Dimensions | 92 × 112 pixels |
| Pixels per Image | 10,304 grayscale pixels |
| Format | Portable Gray Map (`.pgm`) |
| Pixel Intensity Range | 0 to 255 (8-bit grayscale) |

## Training and Testing Split

This project uses a fixed, reproducible split:

- **Training Set (`data/att_faces/Training`)**:
  - 9 images per subject × 40 subjects = **360 training images**
  - Shape of Training Matrix $X_{\text{train}}$: `(10304, 360)`
- **Testing Set (`data/att_faces/Testing`)**:
  - 1 image per subject × 40 subjects = **40 testing images**
  - Shape of Testing Matrix $X_{\text{test}}$: `(10304, 40)`

## Directory Structure

```text
data/
└── att_faces/
    ├── Training/
    │   ├── s1/ (9 images: 1.pgm ... 9.pgm)
    │   ├── s2/ (9 images)
    │   └── ... s40/
    └── Testing/
        ├── s1/ (1 image: 10.pgm)
        ├── s2/ (1 image)
        └── ... s40/
```

## How to Obtain / Restore the Dataset

If setting up the repository in a clean environment without pre-existing data:

1. Download the AT&T Face Dataset from the official archive or Kaggle:
   - AT&T Laboratories Cambridge / Cambridge University Computer Laboratory
2. Place the subject folders (`s1` through `s40`) into `data/att_faces/Training` (first 9 images per subject) and `data/att_faces/Testing` (10th image per subject).
3. Run `python src/dataset_check.py` to verify image shapes and count.
